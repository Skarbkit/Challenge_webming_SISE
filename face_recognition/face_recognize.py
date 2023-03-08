import cv2
import sys
import time



# Charger le classificateur de cascade de Haar pour la détection de visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ouvrir la webcam
cap = cv2.VideoCapture(1)


while True:
    # Capturer un frame de la webcam
    ret, frame = cap.read()

    # Convertir le frame en niveau de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans le frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Dessiner un rectangle autour de chaque visage détecté
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Afficher le frame avec les rectangles autour des visages détectés
    cv2.imshow('frame', frame)

    # Si la touche 'a' est pressée, quitter la boucle
    if cv2.waitKey(1) == ord('a'):
        break

time.Wait(5)
# Libérer la webcam et détruire les fenêtres d'affichage
cap.release()
cv2.destroyAllWindows()
