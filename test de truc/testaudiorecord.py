import cv2
import numpy as np
import pyaudio
import speech_recognition as sr

r = sr.Recognizer()
video_capture = cv2.VideoCapture(0)
recording = False

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

while True:
    with sr.Microphone() as source:
        print("Dites quelque chose...")
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
        elif "stopper enregistrement" in commande:
            stop_recording()
        elif "stopper tout" in commande:
            stop_webcam()
            stop_face_detection()
            stop_recording()
            cv2.destroyAllWindows()
            break
        
    except sr.UnknownValueError:
        print("Je n'ai pas compris ce que vous avez dit")
    except sr.RequestError as e:
        print("Impossible d'effectuer la requête ; {0}".format(e))
    
    ret, frame = video_capture.read()
    
    if recording:
                # .avi et fichier de sortie
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

        while True:
                ret, frame = video_capture.read()

                if recording:
                    out.write(frame)

                # affiche la cam pendant que ca record
                cv2.imshow('Video', frame)

                # sortie de secours avec S
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

            # Ferme la capture vidéo 
        video_capture.release()
        if recording:
                out.release()
        cv2.destroyAllWindows()
        cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()