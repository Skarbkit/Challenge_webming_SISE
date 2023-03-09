import cv2
import streamlit as st
import numpy as np
import threading
import face_recognition
import time
#import torch


# Fonction pour charger les modèles de détection de genre, d'âge et d'émotions

# fonction qui va charger le modèle de détection de genre
def load_genre_detection_model():
    model = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
    return model
# fonction qui va charger le modèle de détection d'émotions
# def load_emotion_detection_model():
#     model = cv2.dnn.readNetFromTensorflow("emotion_frozen.pb")
#     return model
# fonction qui va charger le modèle de détection de genre et d'âge
def load_age_detection_model():
    model = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
    return model
# fonction qui va charger les 3 modèles
def load_all_models():
    age_model = load_age_detection_model()
    #emotion_model = load_emotion_detection_model()
    genre_model = load_genre_detection_model()
    #emotion_model
    return age_model, genre_model



# Fonction pour la détection de visage, d'âge, de genre et d'émotions
def detect_face(img, model_age, model_genre):
    # Convertir l'image en RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Détecter les visages dans l'image
    faces = face_recognition.face_locations(rgb, model='hog')

    # Pour chaque visage détecté, détecter l'âge, le genre et les émotions
    for (top, right, bottom, left) in faces:
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

        # Détecter l'âge
        #age_model = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
        age_model = model_age
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = round(np.sum(age_preds[0] * np.arange(0, 8)) * 4)


        # Détecter le genre
        #gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
        gender_model = model_genre
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        # changer le genre en fonction de la prédiction avec < alors que c'est > à l'origine
        gender = 'Homme' if gender_preds[0][0] > gender_preds[0][1] else 'Femme'

        # # Détecter les émotions
        # emotion_model = cv2.dnn.readNetFromTensorflow('emotion_frozen.pb')
        # face = img[top:bottom, left:right]
        # blob = cv2.dnn.blobFromImage(face, 1.0, (64, 64), (104.0, 177.0, 123.0), False, False)
        # emotion_model.setInput(blob)
        # emotion_preds = emotion_model.forward()
        # emotion_labels = ['Enervé', 'Dégouté', 'Apeuré', 'Heureux', 'Neutre', 'Triste', 'Surpris']
        # emotion = emotion_labels[np.argmax(emotion_preds)]


        # Afficher les résultats
        cv2.putText(img, f'Age: {age}', (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f'Genre: {gender}', (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #cv2.putText(img, f'Emotion: {dominant_emotion}', (right-30, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #cv2.putText(img, f'Score: {emotion_score}', (right, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img


def detect_face_opti(img, age_model, gender_model):
    # Convertir l'image en RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Réduire la résolution de l'image pour accélérer le traitement
    small_rgb = cv2.resize(rgb, (0, 0), fx=0.20, fy=0.20)

    # Détecter les visages dans l'image
    
    faces = face_recognition.face_locations(small_rgb, model='hog')

    # Créer le blob une seule fois pour toutes les détections
    blob = cv2.dnn.blobFromImage(small_rgb, scalefactor=1.0, size=(227, 227),
                                 mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Pour chaque visage détecté, détecter l'âge et le genre
    for (top, right, bottom, left) in faces:
        # Mettre à l'échelle les coordonnées des visages détectés pour correspondre à l'image d'origine
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5
        # Dans la recherche de l'optimisation de la fonction, on a réduit en 1/5 la résolution de l'image
        # Il faut donc multiplier les coordonnées des visages détectés par 5 pour les remettre à l'échelle
        # Au delà de 5 on perd en précision et on ne détecte plus les visages pour des caméras de faible résolution


        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)


        # Détecter l'âge
        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = round(np.sum(age_preds[0] * np.arange(0, 8)) * 4)

        # Détecter le genre
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = 'Homme' if gender_preds[0][0] > gender_preds[0][1] else 'Femme'

        # Afficher les résultats
        cv2.putText(img, f'Age: {age}', (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f'Genre: {gender}', (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img


# Fonction pour la lecture en continu de la webcam
def video_capture(model_age, model_genre):
    # Charger les modèles
    model_age = model_age
    model_genre = model_genre




    cap = cv2.VideoCapture(0) # 0 pour la webcam intégrée, 1 pour une webcam externe
    stframe = st.empty() # Créer un espace vide pour afficher l'image
    start_time = time.time() # Début du chronomètre
    num_frames = 0 # Nombre d'images traitées

    # Paramètres de la webcam
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
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
            thread = threading.Thread(target=detect_face_opti, args=(frame,model_age,model_genre))
            thread.start() # Lancer le thread
            thread.join() # Attendre la fin du thread pour continuer
             # Attendre la fin du thread pour continuer

            # Afficher le nombre d'images traitées par seconde
            num_frames += 1
            elapsed_time = time.time() - start_time
            fps = num_frames / elapsed_time
            cv2.putText(frame, "IPS: {:.2f}".format(fps), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            

            # Afficher l'image dans Streamlit
            stframe.image(frame, channels="BGR")
            
        if cv2.waitKey(1) == ord('a'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Lancer l'application
def main():
    st.title("Détection de visage en temps réel")
    st.text("Caméra en cours d'utilisation...")

    # Charger les modèles
    model_age, model_genre = load_all_models()


    video_capture(model_age, model_genre)

if __name__ == '__main__':
    main()
            

 
