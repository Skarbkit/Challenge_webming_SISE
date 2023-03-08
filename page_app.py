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
    RECORD_SECONDS = 30
    p = pyaudio.PyAudio()

    col1,col2=st.columns(2)

    
    Audio = st.sidebar.radio(
        "Enregistrement audio",
        ("On","Stop"))
    
    Video = st.sidebar.radio(
            "Enregistrement vidéo",
            ("Enregistrement","Arrêt"))

    col1,col2=st.columns([2,10])

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

        if Audio == "Stop":
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            st.write("Fin de l'enregistrement audio")

        if Video == "Enregistrement":
            st.write("Enregistrement vidéo en cours...")
            st.write("Appuyer sur la lettre A du clavier pour arrêter la webcam puis sur Arrêt")
            cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

            while(True):
                # Afficher le flux vidéo de la webcam
                ret, frame = cap.read()

                # Display the resulting frame
                cv2.imshow('frame',frame)
                out.write(frame)
            # Arrêter la boucle en appuyant sur la touche 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

        if Video == "Arrêt":
            # Libérer la webcam et fermer les fenêtres
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            st.write("Fin de l'enregistrement vidéo")

    with col2:
        if Video == "Enregistrement":
            cap, out = start_camera()
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    out.write(frame)
                    cv2.imshow('frame',frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            
        if Video =="Arrêt" :
            stop_camera(cap, out)
            cv2.destroyAllWindows()
            st.write("Fin de l'enregistrement")



if __name__ == "__main__":
    main()