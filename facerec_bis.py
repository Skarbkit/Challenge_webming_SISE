import cv2
import streamlit as st
import numpy as np
import threading
import face_recognition
import os
import pickle

#faire une fonction qui va charger les photos stockées dans le fichier storage_photo 
def storage():
    with open("storage_photo", "rb") as f:
        data = pickle.load(f)
    
    known_faces =data["encodings"]
    known_names = data["names"]
    return(known_faces, known_names)

# Fonction pour la détection de visage, d'âge, de genre et d'émotions
def detect_face(img, known_faces, known_names):
    # Convertir l'image en RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Détecter les visages dans l'image
    faces = face_recognition.face_locations(rgb, model='hog')

    # Pour chaque visage détecté, détecter l'âge, le genre et les émotions
    for (top, right, bottom, left) in faces:
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

        # Détecter la personne
        for i in range(len(known_names)):
            
            # Comparer chaque visage trouvé avec tous les visages connus
            matches = face_recognition.compare_faces(known_faces[i], faces, tolerance=0.7)
            # Trouver le nom correspondant pour chaque visage correspondant
            name = "Inconnu"
            if True in matches:
                # Trouver les index de toutes les correspondances
                match_indexes = [i for (i, b) in enumerate(matches) if b]
                # Compter toutes les correspondances et trouver l'index avec le plus grand nombre de correspondances
                counts = {}
                for i in match_indexes:
                    name = known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)


        cv2.putText(img, f"Nom: {name}", (left, bottom+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img



# Fonction pour la lecture en continu de la webcam
def video_capture(known_faces, known_names):
    cap = cv2.VideoCapture(0) # 0 pour la webcam intégrée, 1 pour une webcam externe
    stframe = st.empty() # Créer un espace vide pour afficher l'image

    # Paramètres de la webcam
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    count = 0
    while True:
        count += 1
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Appeler la fonction de détection de visage dans un thread séparé seulement une image sur deux pour accélérer le traitement
            #if count % 2 == 0:
            thread = threading.Thread(target=detect_face, args=(frame,known_faces,known_names))
            thread.start() # Lancer le thread
            thread.join() # Attendre la fin du thread pour continuer

            # Afficher l'image dans Streamlit
            stframe.image(frame, channels="BGR")
            
        if cv2.waitKey(1) == ord('a'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Lancer l'application
def main():
    known_faces, known_names = storage()
    st.title("Détection de visage en temps réel")
    st.text("Caméra en cours d'utilisation...")

    video_capture(known_names, known_faces)

if __name__ == '__main__':
    main()
            

 
