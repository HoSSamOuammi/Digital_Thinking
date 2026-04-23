# Studio génératif interactif

Application Flask réalisée pour un projet étudiant de créativité numérique.

Le projet permet de :

- générer des visuels artistiques avec des paramètres simples ;
- transformer un jeu de données CSV en visualisation ;
- appliquer des effets à une image ;
- traiter un fichier audio quand `ffmpeg` est disponible ;
- consulter et télécharger les fichiers produits depuis une galerie.

## Objectif

L’objectif n’est pas de produire une application très avancée, mais un projet clair, présentable et facile à expliquer.

La structure a donc été retravaillée pour éviter un gros fichier unique. Le code est maintenant séparé en petites couches :

```text
interactive-generative-studio/
|-- app.py
|-- README.md
|-- REPORT.md
|-- ANALYSE_PROJET.md
|-- requirements.txt
|-- studio/
|   |-- app_factory.py
|   |-- config.py
