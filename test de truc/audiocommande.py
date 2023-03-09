
import cv2
import numpy as np
import pyaudio
import speech_recognition as sr
import threading

r = sr.Recognizer()
video_capture = cv2.VideoCapture(0)
recording = False
stop_threads = False

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

        # affiche la cam pendant que ça record
        cv2.imshow('Video', frame)

        # sortie de secours avec S mais ca marche pas je crois
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Ferme la capture vidéo 
    video_capture.release()
    if recording:
        out.release()
    cv2.destroyAllWindows()

# Création du thread pour l'enregistrement vidéo
record_thread = threading.Thread(target=record_video)

while True:
    with sr.Microphone() as source:
        print("Dit un truc mon reuf")
        audio = r.listen(source)

    try:
        commande = r.recognize_google(audio, language='fr-FR')
        print("Vous avez dit : " + commande)
        
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
        print("répète on comprend pas")
    except sr.RequestError as e:
        print("Impossible d'effectuer la requête ; {0}".format(e))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
