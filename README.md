# Studio génératif interactif

Application Flask réalisée pour le module de créativité numérique. Le projet prend la forme d’un petit studio web où l’on peut générer des visuels, transformer des données, appliquer des effets à une image et retrouver les exports dans une galerie.

L’idée n’était pas de construire une plateforme trop lourde, mais un projet propre, présentable et facile à expliquer à l’oral. On a donc gardé une architecture simple, avec des fichiers séparés par rôle au lieu d’un seul gros `app.py`.

## Ce que fait l’application

- générer des compositions visuelles à partir de paramètres réglables ;
- convertir un fichier CSV, ou un jeu de données de démonstration, en visualisation ;
- appliquer des effets à une image importée ;
- proposer quelques traitements audio lorsque `ffmpeg` est disponible ;
- afficher les fichiers produits dans une galerie avec téléchargement ;
- présenter les membres du groupe et leur rôle.

## Organisation du code

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
|   |-- forms.py
|   |-- labels.py
|   |-- security.py
|   |-- storage.py
|   |-- team.py
|   `-- routes/
|       |-- pages.py
|       |-- generative_routes.py
|       |-- data_routes.py
|       `-- media_routes.py
|-- modules/
|   |-- generative_art.py
|   |-- data_visualization.py
|   |-- image_processing.py
|   `-- audio_processing.py
|-- templates/
|-- static/
`-- tests/
```

## Installation et lancement

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Ouvrir ensuite :

```text
http://127.0.0.1:5000/
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Livrables finaux

- Code complet : dépôt GitHub avec `app.py`, `studio/`, `modules/`, `templates/`, `static/` et `tests/`.
- Application Flask fonctionnelle : lancement avec `python app.py`.
- Rapport technique détaillé illustré : `docs/rapport_technique/rapport_technique.pdf`.
- Version Markdown du rapport : `docs/rapport_technique/rapport_technique.md`.

## Répartition du travail

- Aya EL Amrani : structure Flask, configuration, formulaires, stockage et routes.
- Khadija Baskar : textes français, libellés, cohérence des intitulés et contenus visibles.
- Hossam OUammi : intégration Flask, interface, pages de présentation, médias et galerie.
- Abderrahmane El Garti : tests fonctionnels, documentation, rapport et analyse technique.

## Notes techniques

- Les exports sont enregistrés dans `static/generated`.
- Les fichiers importés temporairement sont nettoyés après traitement.
- Le module audio dépend de `pydub` et de `ffmpeg`; l’application reste utilisable si l’audio n’est pas disponible.
- Le projet n’utilise ni base de données ni authentification, volontairement, pour rester lisible dans un cadre scolaire.
