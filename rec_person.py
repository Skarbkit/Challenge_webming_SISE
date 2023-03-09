import streamlit as st
import cv2
import face_recognition
import os

def main(): 
    picdata_dir = "C:/Users/cornuch/Documents/GitHub/Challenge_webming_SISE/face_recognition2/picdata"
    
    # enregistrer les visages connus
    known_encodings = []
    known_names = []
    for filename in os.listdir(picdata_dir):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"): 
            nom = str.split(filename,".")[0]
            known_image = face_recognition.load_image_file(os.path.join(picdata_dir, filename))
            known_encoding = face_recognition.face_encodings(known_image)[0]
            known_encodings.append(known_encoding)
            known_names.append(nom)

    # initialiser la capture vidéo
    video_capture = cv2.VideoCapture(0)

    # créer une boucle while pour détecter et reconnaître les visages en temps réel
    while True:
        # lire la vidéo en temps réel
        ret, frame = video_capture.read()

        # détecter les visages dans l'image
        face_locations = face_recognition.face_locations(frame)

        # extraire les visages de l'image
        faces = []
        for top, right, bottom, left in face_locations:
            face_image = frame[top:bottom, left:right]
            faces.append(face_image)

        # permettre la reconnaissance faciale
        for face in faces:

            face_encodings = face_recognition.face_encodings(face)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
        
                # comparer le visage encodé avec tous les visages connus
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
            
                # rechercher le nom correspondant au visage correspondant
                nom = "Inconnu"
                for i, match in enumerate(matches):
                    if match:
                        nom = known_names[i]

                # dessiner un rectangle autour du visage et l'identifier
                top, right, bottom, left = face_recognition.face_locations(face)[0]
                if nom == "Inconnu":
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Inconnu", (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, nom, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # afficher l'image résultante
                st.image(frame, channels="BGR")
            else :
                st.write("Pas de visage détecté")

if __name__ == "__main__":
    main()