import streamlit as st
import cv2
import numpy as np
import pyaudio
import wave

def start_camera():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    return cap, out

def stop_camera(cap, out):
    cap.release()
    out.release()

def main():
    st.title("Webcam et enregistrement vocal")

    # Initialiser la webcam
    cap = cv2.VideoCapture(0)

    # Initialiser l'enregistreur vocal
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 10
    p = pyaudio.PyAudio()

    col1,col2=st.columns(2)

    
    Audio = st.sidebar.radio(
        "Enregistrement audio",
        ("On","Stop"))
    

    col1,col2=st.columns([2,10])

    stream = None

    with col1:
        if Audio == "On":
            st.write("Enregistrement audio en cours...")
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
            stream=None

        else:
            if stream==None :
                stream.stop_stream()
                stream.close()
                p.terminate()
                st.write("Fin de l'enregistrement audio")
            wf = wave.open("enregistrement.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            st.audio("enregistrement.wav")
            st.write(f"Enregistrement audio terminé. Fichier enregistré sous enregistrement.wav")

    with col2:
        # Paramètres de la webcam
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = 20

        # Paramètres d'enregistrement
        filename = "recording.avi"
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        # Bouton pour démarrer l'enregistrement
        if st.sidebar.button("Démarrer l'enregistrement"):
            st.write("Enregistrement en cours...")
            st.write("Appuyer sur la lettre Q du clavier pour arrêter la webcam")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                cv2.imshow('Webcam', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            st.write(f"Enregistrement terminé. Fichier enregistré sous {filename}")

        #if st.sidebar.button("Arrêter l'enregistrement"):
         #   st.stop()




if __name__ == "__main__":
    main()