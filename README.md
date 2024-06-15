# Projet de classification des Espèces Animales à partir de leur cri

Ce projet utilise la recherche par similarité pour classifier des sons émis par différentes espèces animales. Actuellement, le projet se concentre sur la distinction entre trois catégories principales : les chiens, les chats, et les oiseaux. Le but est de permettre une identification rapide et précise de l'espèce à partir d'un enregistrement audio, ce qui peut être utile pour des applications écologiques, éducatives, ou de loisir.


## Installation
Vous devez avoir la version 3.9 de Python installé sur votre ordinateur.
Pour éviter des problèmes de dépendances, il est conseillé de travailler dans un environnement virtuel et d'effectuer l'installation de toutes les dépendances dans celui-ci.
Pour utiliser un environnement virtuel, installer anaconda sur votre ordinateur. Ensuite ouvrez l'Anaconda prompt et executez la commande suivante pour créer un environnement virtuel:
```
conda create -n env_name python=3.9
```
avec env_name, le nom de votre environnement.

Puis executez la commande suivante pour activer l'environnement virtuel.
```
conda activate env_name
```

Naviguez jusque dans le dossier du projet.
Installer les dépendances nécessaires à la bonne exécution du projet via pip en exécutant :

```
pip install -r requirements.txt
```



## Utilisation
Tout d'abord, assurez-vous que les chemins d'accès au fichier "noms_animaux.txt" et à l'index "animals.index" sur votre ordinateur en local sont correctement configurés dans le script au niveau des variables "chemin_noms_animaux" et "index_path".
Pour démarrer l'interface utilisateur Gradio et classer les sons d'animaux, exécutez la commande:

```
python hubert.py
```
Une fois sur l'interface utilisateur, téléchargez un fichier audio de l'animal que vous souhaitez classer.
Le système vous retournera la classification de l'espèce avec la probabilité associée.


### Ajout de Nouveaux Sons

Pour ajouter un nouveau son à la base de données dans le cas ou le cri animal n'est pas correctement classé:
1. Utilisez le  fichier `add_index` en executant la commande:
```
python add_index.py  chemin_de_l_audio_à_ajouter
```
2. Le script mettra à jour l'index et la liste des noms d'animau


## Explication des différentes fonctions 
Il y a deux fonctions principales dans ce script, la fonction animal_classification et la fonction add_in_index.
La fonction animal_classification est la fonction de classification des animaux à partir des sons. Elle prend en paramètre un audio au format .wav .
Pour l'utiliser , executez la commande suivante:
```
python animal_classification.py  chemin_de_l_audio_à_ajouter
```
La fonction add_in_index est la fonction qui permet d'ajouter un nouveau son à la base de données afin d'améliorer la qualité de notre qualification.
Pour l'utiliser , executez la commande suivante:
```
python add_index.py "chemin_de_l_audio_à_ajouter"
```
