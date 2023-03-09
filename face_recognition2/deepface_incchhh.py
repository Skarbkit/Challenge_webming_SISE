import cv2
import streamlit as st
import numpy as np
import threading
from deepface import DeepFace

# Fonction pour la détection de visage
def detect_face(img):
    # Charger le classifieur cascade pour la détection de visage
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Dessiner des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Détecter l'âge, le genre et les émotions
        face_img = img[y:y+h, x:x+w]
        result = DeepFace.analyze(face_img, actions=['age', 'gender', 'emotion'])
        age = result['age']
        gender = result['gender']
        emotion = max(result['emotion'], key=result['emotion'].get)
        
        # Afficher les informations sur l'image
        cv2.putText(img, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Gender: {gender}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Emotion: {emotion}", (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img

# Fonction pour la lecture en continu de la webcam
def video_capture():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # Appeler la fonction de détection de visage dans un thread séparé
            thread = threading.Thread(target=detect_face, args=[frame])
            thread.start()

            # Afficher l'image dans Streamlit
            st.image(frame, channels="BGR", use_column_width=True)
            
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Lancer l'application
def main():
    st.title("Détection de visage avec OpenCV et Streamlit")
    st.text("En direct depuis votre webcam !")

    video_capture()

if __name__ == '__main__':
    main()