# Challenge Web mining SISE

Ce projet consistait à effectuer une reconnaissance faciale et identification sur des vidéos WEBCAM des étudiants du Master SISE
avec commande vocale avec le choix d'une application streamlit.

![](images/detect_face.png)

## Description du dépôt Git:
Le dossier contient 

**Afin de lancer l'application il faut :**

Créer un environnement et réaliser un git clone du dépôt dans un dossier de votre choix.

Dans la console bash :

```
git clone https://github.com/Skarbkit/Challenge_webming_SISE.git
```

Installer le fichier requirements.txt :
```
$pip install -r requirements.txt
```
Lancer l'application Streamlit :
```
python -m streamlit run Accueil.py
```
Pour lancer l'appli avec docker:

-aller dans un invite de commande -aller dans le chemin du projet
```
cd .../Challenge_OPSIE-SISE
```
-construire l'image docker:
```
docker build -t nomchoisidelimage .
```
Dans un invite de commande, run l'image:

docker run nomchoisidelimage 

Les liens url ne sont pas valides car l'application n'est pas hébergée en ligne, il faut ouvrir un nouvel onglet et aller à l'url localhost:8501

## Comment utiliser l'application:
