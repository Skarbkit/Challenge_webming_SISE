import streamlit as st
import cv2
import numpy as np
import pyaudio
import speech_recognition as sr
import threading

# Initialisation de la webcam
video_capture = cv2.VideoCapture(0)

# Initialisation de la reconnaissance vocale
r = sr.Recognizer()

# Variables pour l'enregistrement vidéo
recording = False
stop_threads = False

# Fonctions pour la webcam et l'enregistrement vidéo
def start_webcam():
    global video_capture
    video_capture = cv2.VideoCapture(0)

def stop_webcam():
    video_capture.release()

def start_recording():
    global recording
    recording = True

def stop_recording():
    global recording
    recording = False

def record_video():
    global stop_threads
    # .avi et fichier de sortie
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while not stop_threads:
        ret, frame = video_capture.read()

        if recording:
            out.write(frame)

        # Affiche la webcam pendant que ça record
        cv2.imshow('Video', frame)

        # Sortie de secours avec la touche 'S'
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Ferme la capture vidéo 
    video_capture.release()
    if recording:
        out.release()
    cv2.destroyAllWindows()

# Création du thread pour l'enregistrement vidéo
record_thread = threading.Thread(target=record_video)

# Fonction pour la reconnaissance vocale
def recognize_audio():
    global stop_threads
    while True:
        with sr.Microphone() as source:
            st.write("Dites quelque chose :")
            audio = r.listen(source)

        try:
            commande = r.recognize_google(audio, language='fr-FR')
            st.write("Vous avez dit : " + commande)

            if "démarrer webcam" in commande:
                start_webcam()
            elif "stopper webcam" in commande:
                stop_webcam()
            elif "démarrer enregistrement" in commande:
                start_recording()
                # Démarrage du thread d'enregistrement vidéo
                stop_threads = False
                record_thread.start()
            elif "stopper enregistrement" in commande:
                stop_recording()
                # Arrêt du thread d'enregistrement vidéo
                stop_threads = True
            elif "stopper tout" in commande:
                stop_webcam()
                stop_recording()
                stop_threads = True
                cv2.destroyAllWindows()
                break
        except sr.UnknownValueError:
            st.write("Je n'ai pas compris ce que vous avez dit.")
        except sr.RequestError as e:
            st.write("Impossible d'effectuer la requête ; {0}".format(e))

# Création du thread pour la reconnaissance vocale
audio_thread = threading.Thread(target=recognize_audio)

# Définition de l'application Streamlit
def app():
    st.title("Application de reconnaissance vocale")

    # Affichage de la webcam
    if video_capture.isOpened():
        ret, frame = video_capture.read()
        st.image(frame, channels="BGR", use_column_width=True)

    # Démarrage du thread de reconnaissance vocale
    stop_threads = False
    audio_thread.start()

if __name__ == "__main__":
    app()