import streamlit as st
import cv2
import numpy as np
import pyaudio
import wave
import face_recognition
import threading
import cv2
import streamlit as st
import numpy as np
import threading
import time
import pickle
from keras.models import load_model
from fonctions_reco import *


def main():
    st.title("Enregistrement audio et vidéo")

    # Initialiser la webcam
    cap = cv2.VideoCapture(0)

    # Initialiser l'enregistreur vocal
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 10
    p = pyaudio.PyAudio()

    audio = st.sidebar.button("Enregistrement audio")
    
    video = st.sidebar.button("Enregistrement vidéo")

    coef_res = st.sidebar.select_slider("Coefficient de résolution", [1,2,4,5,8],5)

    col1,col2=st.columns([2,10])

    with col1:
        if audio :
            st.write("Enregistrement audio d 10 secondes en cours...")
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            wf = wave.open("enregistrement.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            st.audio("enregistrement.wav")
            st.write("Fin de l'enregistrement audio")

    with col2:
        if video :
            cap = cv2.VideoCapture(0) # 0 pour la webcam intégrée, 1 pour une webcam externe
            stframe = st.empty() # Créer un espace vide pour afficher l'image
            start_time = time.time() # Début du chronomètre
            num_frames = 0 # Nombre d'images traitées

            # Paramètres de la webcam
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 60)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480)) # Créer l'objet VideoWriter

            # Charger le modèle de détection de visage
            model_age, model_genre, emotion_model = load_all_models()
            known_names, known_encodings = load_known_data()
            count = 0
            while True:
                count += 1
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)

                    # Appeler la fonction de détection de visage dans un thread séparé seulement une image sur deux pour accélérer le traitement
                    #if count % 2 == 0:
                    thread = threading.Thread(target=detect_face_opti, args=(frame,model_age,model_genre,emotion_model, known_names, known_encodings,coef_res))
                    thread.start() # Lancer le thread
                    thread.join() # Attendre la fin du thread pour continuer

                    # Afficher l'image dans Streamlit
                    num_frames+=1
                    elapsed_time = time.time() - start_time
                    fps = num_frames / elapsed_time
                    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    stframe.image(frame, channels="BGR")

                    # Écrire l'image dans le fichier vidéo
                    out.write(frame)
            
                    if cv2.waitKey(1) == ord('a'):
                        break
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()