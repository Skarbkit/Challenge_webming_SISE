import cv2
import streamlit as st
import numpy as np
import threading
import face_recognition
import os
import pickle


with open("storage_photo", "rb") as f:
    data = pickle.load(f)
    
known_faces =data["encodings"]
known_names = data["names"]

# Fonction pour la détection de visage, d'âge, de genre et d'émotions
def detect_face(img):
    # Convertir l'image en RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Détecter les visages dans l'image
    faces = face_recognition.face_locations(rgb, model='hog')

    # Pour chaque visage détecté, détecter l'âge, le genre et les émotions
    for (top, right, bottom, left) in faces:
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

        # Détecter la personne
        for i in range(len(known_faces)):
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



        # Détecter l'âge
        age_model = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = round(np.sum(age_preds[0] * np.arange(0, 8)) * 4)


        # Détecter le genre
        gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        # changer le genre en fonction de la prédiction avec < alors que c'est > à l'origine
        gender = 'Homme' if gender_preds[0][0] > gender_preds[0][1] else 'Femme'

        # # Détecter les émotions
        # emotion_model = cv2.dnn.readNetFromTensorflow('emotion_frozen.pb')
        # face = img[top:bottom, left:right]
        # blob = cv2.dnn.blobFromImage(face, 1.0, (64, 64), (104.0, 177.0, 123.0), False, False)
        # emotion_model.setInput(blob)
        # emotion_preds = emotion_model.forward()
        # emotion_labels = ['Enervé', 'Dégouté', 'Apeuré', 'Heureux', 'Neutre', 'Triste', 'Surpris']
        # emotion = emotion_labels[np.argmax(emotion_preds)]


        # Afficher les résultats
        cv2.putText(img, f'Age: {age}', (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f'Genre: {gender}', (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #cv2.putText(img, f'Emotion: {dominant_emotion}', (right-30, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #cv2.putText(img, f'Score: {emotion_score}', (right, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, f"Nom: {name}", (left, bottom+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img



# Fonction pour la lecture en continu de la webcam
def video_capture():
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
            thread = threading.Thread(target=detect_face, args=(frame,))
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
    st.title("Détection de visage en temps réel")
    st.text("Caméra en cours d'utilisation...")

    video_capture()

if __name__ == '__main__':
    main()
            

 
