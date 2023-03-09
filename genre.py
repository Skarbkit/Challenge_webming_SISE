
import cv2

def load_genre_detection_model():
    model = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
    return model
# fonction qui va charger le modèle de détection d'émotions
# def load_emotion_detection_model():
#     model = cv2.dnn.readNetFromTensorflow("emotion_frozen.pb")
#     return model
# fonction qui va charger le modèle de détection de genre et d'âge

load_genre_detection_model()