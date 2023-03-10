# Challenge Web mining SISE

Ce projet consistait à effectuer une reconnaissance faciale et identification sur des vidéos WEBCAM des étudiants du Master SISE avec commande vocale dans une application streamlit.

![](image/detect_2_visages.png)

## Description du dépôt Git
Le dossier contient modele_propre contient tous les modèles pré-entraînés, le fichier des photos encodées ainsi que le fichier contenant toutes les fonctions nécessaires pour l'application streamlit.

Le dossier audio contient des essais sur les commandes vocales et leur intégration dans streamlit.

Le dossier image contient des images des rendus visuels de l'application.

**Afin de lancer l'application il faut :**

Créer un environnement et réaliser un git clone du dépôt dans un dossier de votre choix.

Dans la console bash :

```
git clone https://github.com/Skarbkit/Challenge_webming_SISE.git
```
-aller dans un invite de commande -aller dans le chemin du projet
```
cd .../Challenge_webming_SISE/modele_propre
```

Installer le fichier requirements.txt :
```
$pip install -r requirements.txt
```
Lancer l'application Streamlit :
```
python -m streamlit run application.py
```
Pour lancer l'appli avec docker:
```
cd .../Challenge_webming_SISE/modele_propre
```
-construire l'image docker:
```
docker build -t nomchoisidelimage .
```
Dans un invite de commande, run l'image:
```
docker run nomchoisidelimage 
```
Les liens url ne sont pas valides car l'application n'est pas hébergée en ligne, il faut ouvrir un nouvel onglet et aller à l'url localhost:8501

## Comment utiliser l'application

Lorsque vous avez lancer l'application streamlit, vous pouvez enregistrer un audio de 10 secondes. Vous pouvez directement écouter votre enregistrement sur l'application.

De plus, vous pouvez aussi lancer l'enregistrement d'une vidéo que vous stopper avec le bouton stop situé en haut à droite de la fenêtre de l'application. Votre enregistrement vidéo est ensuite disponible dans votre dossier. 

Dans l'application_res, vous pouvez en plus modifier la résolution.
