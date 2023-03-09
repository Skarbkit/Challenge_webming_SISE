import streamlit as st
import cv2
import face_recognition
import os

def main(): 
    filename = "Christelle Cornu.jpeg"
    nom = str.split(filename,".")[0]
    

    #enregistrer les visages connus
    known_image = face_recognition.load_image_file("C:/Users/cornuch/Documents/GitHub/Challenge_webming_SISE/face_recognition2/picdata/Christelle Cornu.jpeg")
    known_encoding = face_recognition.face_encodings(known_image)[0]

    #initialiser la capture vidéo
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

                # comparer le visage encodé avec le visage connu
                results = face_recognition.compare_faces([known_encoding], face_encoding)
                if results[0]:
                # si match, dessiner un rectangle autour du visage et l'identifier
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, nom, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                # sinon dessiner un rectangle rouge autour du visage et l'identifier comme inconnu
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Inconnu", (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # afficher l'image résultante
                st.image(frame, channels="BGR")
            else:
                st.write("Aucun visage détecté")
    
    # fermer la caméra après avoir analysé tous les visages
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()

if __name__ == "__main__":
    main()
