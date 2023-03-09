import cv2
import streamlit as st
import numpy as np
import threading
import face_recognition
import os
import pickle

#sélection du directory de travail et liste vides pour stockage photos et noms
known_faces = []
known_names = []
known_dir = "C:/Users/cornuch/Documents/GitHub/Challenge_webming_SISE/photos/"

# Charger toutes les images connues dans le dossier 'known_faces'
for file in os.listdir(known_dir):
    img = face_recognition.load_image_file(os.path.join(known_dir, file))
    # Extraire l'encodage du visage de chaque image
    encoding = face_recognition.face_encodings(img)[0]
    known_faces.append(encoding)
    known_names.append(file.split('.')[0])

#save emcodings along with their names in dictionary data
data = {"encodings": known_faces, "names": known_names}
#use pickle to save data into a file for later use
filename = "C:/Users/cornuch/Documents/GitHub/Challenge_webming_SISE"

# Enregistrer les données dans le fichier spécifié
with open(filename, "wb") as f:
    f.write(pickle.dump(data, f))
    f.close()

# f = open("storage_photo", "wb")
# f.write(pickle.dumps(data))
# f.close()