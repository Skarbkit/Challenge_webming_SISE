import cv2
import streamlit as st
import numpy as np
import face_recognition
import threading

# Fonction pour la détection de visage et des attributs
def detect_face(img):
    # Détecter les visages dans l'image
    face_locations = face_recognition.face_locations(img)
    
    # Trouver les attributs de chaque visage détecté
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # Boucler sur chaque visage détecté
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        # Trouver le nom de la personne correspondante
        # Charger une image connue pour l'exemple
        known_image = face_recognition.load_image_file("elon_musk.jpg")
        known_encoding = face_recognition.face_encodings(known_image)[0]

        # Comparer les encodages de visage pour voir s'il y a une correspondance
        results = face_recognition.compare_faces([known_encoding], face_encoding)
        name = "Inconnu"

        if results[0]:
            name = "Elon Musk"

        # Afficher le nom de la personne correspondante sur le visage détecté
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

# Fonction pour la lecture en continu de la webcam
def video_capture():
    cap = cv2.VideoCapture(1)
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # Appeler la fonction de détection de visage dans un thread séparé
            thread = threading.Thread(target=detect_face, args=[frame])
            thread.start()
            thread.join()

            # Afficher l'image dans Streamlit
            stframe.image(frame, channels="BGR")
            
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Lancer l'application
def main():
    st.title("Détection de visage avec face_recognition et Streamlit")
    st.text("En direct depuis votre webcam !")

    video_capture()

if __name__ == '__main__':
    main()
