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

def age_detection(age_model, blob):
            age_model.setInput(blob)
            age_preds = age_model.forward()
            age = round(np.sum(age_preds[0] * np.arange(0, 8)) * 4)
            return age

def gender_detection(gender_model, blob):
            gender_model.setInput(blob)
            gender_preds = gender_model.forward()
            gender = 'Homme' if gender_preds[0][0] > gender_preds[0][1] else 'Femme'
            return gender

def name_detection(known_names, known_encodings, encode_faces):
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
        return names

def emotion_detection( resized, emotion_model):
    # Ajouter une dimension pour le modèle
    normalized = resized.astype('float') / 255.0
    reshaped = np.expand_dims(normalized, axis=0)
    # Faire la prédiction des émotions
    predictions = emotion_model.predict(reshaped)
    # Obtenir l'indice de l'émotion prédite
    emotion_index = np.argmax(predictions)
    # Obtenir la liste des émotions
    emotions = ['Colère', 'Dégoût', 'Peur', 'Heureux', 'Neutre', 'Triste', 'Surpris']
    # Obtenir le nom de l'émotion prédite
    emotion_name = emotions[emotion_index]
    return emotion_name

def detect_face_opti(img, age_model, gender_model,emotion_model, known_names, known_encodings,coef=5):
    # Convertir l'image en RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    
    emotion_name = emotion_detection(resized, emotion_model)
    # Réduire la résolution de l'image pour accélérer le traitement
    small_rgb = cv2.resize(rgb, (0, 0), fx=1/coef, fy=1/coef)

    # Détecter les visages dans l'image
    
    faces = face_recognition.face_locations(small_rgb, model='hog')
    encode_faces = face_recognition.face_encodings(img, faces)

    names = name_detection(known_names, known_encodings, encode_faces)

    # Créer le blob une seule fois pour toutes les détections
    blob = cv2.dnn.blobFromImage(small_rgb, scalefactor=1.0, size=(227, 227),
                                 mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Pour chaque visage détecté, détecter l'âge et le genre
    for (top, right, bottom, left), name in zip(faces, names):
        # Mettre à l'échelle les coordonnées des visages détectés pour correspondre à l'image d'origine
        top *= coef
        right *= coef
        bottom *= coef
        left *= coef
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
        def age_thread_fun(blob):
             global age_result
             age_result = age_detection(age_model, blob)


        def gender_thread_fun(blob):
             global gender_result
             gender_result = gender_detection(gender_model, blob)

        age_thread = threading.Thread(target=age_thread_fun, args=(blob,))

        gender_thread = threading.Thread(target = gender_thread_fun, args = (blob,))

        # Détecter le genre
        age_thread.start()
        gender_thread.start()

        age_thread.join()
        gender_thread.join()

        age = age_result
        gender = gender_result
        

        # Afficher les résultats
        cv2.putText(img, f'Age: {age}', (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f'{gender}', (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"{name}", (left, bottom+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"{emotion_name}", (right, top -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    return img