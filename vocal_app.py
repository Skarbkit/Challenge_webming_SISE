import streamlit as st
import cv2
import speech_recognition as sr
import threading
import face_recognition
import pyaudio

# Create a flag to track whether the webcam is running
#run = st.checkbox("Run Webcam")

# Create a button to stop the webcam
stop_button = st.sidebar.button("Stop Webcam")


def detect_face(img):
    # Convertir l'image en RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Détecter les visages dans l'image
    faces = face_recognition.face_locations(rgb, model='hog')

    # Pour chaque visage détecté, détecter l'âge, le genre et les émotions
    for (top, right, bottom, left) in faces:
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

    return img


#fonction qui capture la video en utilisant la camera
def capture_video():
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
            thread = threading.Thread(target=detect_face, args=(frame))
            thread.start() # Lancer le thread
            thread.join() # Attendre la fin du thread pour continuer

            # Afficher l'image dans Streamlit
            stframe.image(frame, channels="BGR")
        #if cv2.waitKey(1) == ord('a'):
           # break

def record_video():

    cap = cv2.VideoCapture(0) # 0 pour la webcam intégrée, 1 pour une webcam externe
    stframe = st.empty() # Créer un espace vide pour afficher l'image

    cap.set(cv2.CAP_PROP_FPS, 60)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    out = cv2.VideoWriter('record.avi',fourcc, 20.0, (640,480))

    count = 0
    while True:
        count += 1
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Appeler la fonction de détection de visage dans un thread séparé seulement une image sur deux pour accélérer le traitement
            #if count % 2 == 0:
            thread = threading.Thread(target=detect_face, args=(frame))
            thread.start() # Lancer le thread
            thread.join() # Attendre la fin du thread pour continuer

            # Afficher l'image dans Streamlit
            stframe.image(frame, channels="BGR")
            out.write(frame) 
        else:
            break
        if st.button("Arrêter l'enregistrement"):
            is_recording = False     
        #if cv2.waitKey(1) == ord('a'):
           # break
    #cap.release()
    #cv2.destroyAllWindows()
        return(out)


def start_stop_camera():
    while True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Dites 'Démarrer' pour démarrer la caméra. Stop arrête la reconnaissance vocale.")
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio, language='fr-FR')
                st.write("Vous avez dit : " + text)
                if "démarrer" in text.lower():
                    st.write("Démarrage de la caméra...")
                    capture_video()
                elif "arrêter" in text.lower():
                    st.write("Arrêt de la caméra...")
                    #stop_camera()
                    #cap.out.release()
                elif "audio" in text.lower():
                   record_video()
                else:
                    st.write("Commande non reconnue.")
            except "stop" in text.lower():
                st.write("Arrêt de la reconnaissance vocale.")
        

def main():
    st.title("Contrôle de la caméra par commande vocale")
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)

    # Initialiser l'enregistreur vocal
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 30
    p = pyaudio.PyAudio()

    start_stop_camera()
    # Check if the stop button has been clicked
    if stop_button:
        # Stop the webcam video stream
        cap.release()
        #cap.out.release()

if __name__ == "__main__":
    main()

