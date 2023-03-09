import cv2
import streamlit as st
import numpy as np
import threading
import face_recognition
import time
import pickle
from keras.models import load_model
#import torch

def load_known_data():
    data = pickle.loads(open("face_enc", "rb").read())
    return (
        data["names"], 
        data["encodings"]
        )


def load_emotion_model():
    model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
    return model

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
    emotion_model = load_emotion_model()
    return age_model, genre_model, emotion_model



def detect_face_opti(img, age_model, gender_model,emotion_model, known_names, known_encodings):
    # Convertir l'image en RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype('float') / 255.0

    # Ajouter une dimension pour le modèle
    reshaped = np.reshape(normalized, (1, 64, 64, 1))

    # Faire la prédiction des émotions
    predictions = emotion_model.predict(reshaped)

    # Obtenir l'indice de l'émotion prédite
    emotion_index = np.argmax(predictions)

    # Obtenir la liste des émotions
    emotions = ['Colère', 'Dégoût', 'Peur', 'Heureux', 'Neutre', 'Triste', 'Surpris']

    # Obtenir le nom de l'émotion prédite
    emotion_name = emotions[emotion_index]
    # Réduire la résolution de l'image pour accélérer le traitement
    small_rgb = cv2.resize(rgb, (0, 0), fx=0.20, fy=0.20)

    # Détecter les visages dans l'image
    
    faces = face_recognition.face_locations(small_rgb, model='hog')
    encode_faces = face_recognition.face_encodings(img, faces)


    #print("Nombre de visages détectés: ", len(faces))
    #print(len(encode_faces))
    names = []

    for face_encoding in encode_faces:
        face_encoding = np.reshape(face_encoding, (-1,128))
        matches = [face_recognition.compare_faces(known_encoding, face_encoding, tolerance=0.7) for known_encoding in known_encodings]
        matches_count = [sum(match) for match in matches]
        if max(matches_count) > 0:
            # Trouver le nom avec le plus de correspondances
            index = matches_count.index(max(matches_count))
            names.append(known_names[index])
        else:
            names.append("Inconnu")

    # Créer le blob une seule fois pour toutes les détections
    blob = cv2.dnn.blobFromImage(small_rgb, scalefactor=1.0, size=(227, 227),
                                 mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Pour chaque visage détecté, détecter l'âge et le genre
    for (top, right, bottom, left), name in zip(faces, names):
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

        # Récupérer les coordonnées du visage détecté
        # for i in range(len(known_encodings)):
        #     # Comparer chaque visage trouvé avec tous les visages connus
        #     encode_face = face_recognition.face_encodings(img, faces)
        #     #print(type(encode_face))
        #     matches = face_recognition.compare_faces(known_encodings[i], encode_face, tolerance=0.7)
        #     # Trouver le nom correspondant pour chaque visage correspondant
        #     name = "Inconnu"
        #     if True in matches:
        #         # Trouver les index de toutes les correspondances
        #         match_indexes = [i for (i, b) in enumerate(matches) if b]
        #         # Compter toutes les correspondances et trouver l'index avec le plus grand nombre de correspondances
        #         counts = {}
        #         for i in match_indexes:
        #             name = known_names[i]
        #             counts[name] = counts.get(name, 0) + 1
        #             name = max(counts, key=counts.get)


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
        cv2.putText(img, f'{gender}', (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"{name}", (left, bottom+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"{emotion_name}", (right, top -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    return img


# Fonction pour la lecture en continu de la webcam
def video_capture(model_age, model_genre,emotion_model,know_names,known_encodings):
    # Charger les modèles
    model_age = model_age
    model_genre = model_genre
    emotion_model = emotion_model
    known_names = know_names
    known_encodings = known_encodings




    cap = cv2.VideoCapture(0) # 0 pour la webcam intégrée, 1 pour une webcam externe
    stframe = st.empty() # Créer un espace vide pour afficher l'image
    start_time = time.time() # Début du chronomètre
    num_frames = 0 # Nombre d'images traitées

    # Paramètres de la webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    #cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    count = 0
    while True:
        count += 1
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Appeler la fonction de détection de visage dans un thread séparé seulement une image sur deux pour accélérer le traitement
            #if count % 2 == 0:
            thread = threading.Thread(target=detect_face_opti, args=(frame,model_age,model_genre,emotion_model, know_names, known_encodings))
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
    model_age, model_genre, emotion_model = load_all_models()
    known_names, known_encodings = load_known_data()


    video_capture(model_age, model_genre,emotion_model, known_names, known_encodings)

if __name__ == '__main__':
    main()
            

 
