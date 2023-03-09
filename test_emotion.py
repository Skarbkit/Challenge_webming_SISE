import cv2
import facial_emotion_recognition
import streamlit as st

def main():
        # initilisation de la web cam et du modèle de reconnaissance d'émotions
    cap = cv2.VideoCapture(0)
    fer = facial_emotion_recognition.FER()

    # créer une fenêtre Streamlit pour afficher les résultats
    st.set_page_config(page_title="Face and Emotion Detection", page_icon=":smiley:")
    st.title("Détection d'émotions")
    emplacement = st.empty()

    # détecter les visages et les émotions en temps réel
    while True:
        # lire la vidéo en temps réel
        ret, frame = cap.read()

        # convertir l'image de BGR à RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # détecter les visages et les émotions
        emotions = fer.predict_emotions(rgb_frame)

        # dessiner un rectangle autour des visages et les identifier avec leur émotion
        for (x, y, w, h), emotion in emotions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{emotion.capitalize()}"
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame in the Streamlit window
        emplacement.image(frame, channels="BGR")

        # Check if the user has clicked the "Stop" button
        if st.button("Stop"):
            break

    # Release the webcam and close the Streamlit window
    cap.release()
    st.stop()

if __name__ == "__main__":
    main()