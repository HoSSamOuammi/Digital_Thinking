# Rapport technique détaillé - Studio génératif interactif

- Date du rapport : 26/06/2026
- Période Git observée : 2026-03-04 au 2026-06-26
- Dépôt GitHub : https://github.com/HoSSamOuammi/Digital_Thinking

## 1. Idée générale et direction artistique

Le projet prend la forme d'un petit studio numérique. L'utilisateur peut générer des images, transformer des données, appliquer des effets médias puis retrouver les exports dans une galerie. Nous avons voulu garder une interface claire, avec un style sobre et des textes français, pour que le projet soit agréable à présenter et facile à tester.

## 2. Modules réalisés

### Tableau de bord

La page d'accueil donne le ton du projet. On y retrouve les compteurs, les accès aux ateliers et les derniers exports. C'est volontairement simple: quelqu'un qui découvre le projet doit comprendre en quelques secondes où cliquer pour tester l'application.

### Atelier génératif

L'atelier génératif est la partie la plus créative. L'utilisateur règle une série visuelle, une palette, un fond, une graine et plusieurs paramètres de densité ou de taille. La prévisualisation aide à tester rapidement une idée avant de lancer l'export final.

### Données visuelles

Ce module transforme un CSV, ou un jeu de démonstration, en image. Les valeurs numériques sont nettoyées puis utilisées pour produire une visualisation. L'intérêt est de montrer que le studio ne génère pas seulement des formes abstraites: il peut aussi partir de données.

### Outils médias

La page médias permet d'importer une image et d'appliquer des effets visibles: sépia, contours, rotation, glitch, palette dominante, etc. Le traitement audio reste prévu, mais il dépend de ffmpeg, donc l'application affiche clairement l'état de disponibilité au lieu de planter.

### Galerie

La galerie ferme le parcours utilisateur. Après une génération ou un traitement, les fichiers sont listés avec pagination et liens de téléchargement. Cette page prouve que le flux complet fonctionne: créer, sauvegarder, retrouver, télécharger.

### Équipe

La page équipe présente les membres, leurs rôles, leurs emails et leurs photos. Elle rend le projet plus humain et permet de relier les parties techniques à la répartition réelle du travail.

## 3. Architecture et pipeline

- `app.py` lance l'application.
- `studio/app_factory.py` crée Flask et enregistre les routes.
- `studio/routes/` sépare les vues par domaine.
- `studio/forms.py`, `storage.py` et `security.py` isolent les tâches répétitives.
- `modules/` contient les traitements de génération, données, image et audio.

Pipeline :
- L'utilisateur choisit un atelier depuis le tableau de bord.
- Le formulaire est envoyé à Flask avec un jeton CSRF.
- Les paramètres sont lus et normalisés dans studio/forms.py.
- Le module métier correspondant génère ou transforme le contenu.
- Le fichier obtenu est sauvegardé dans static/generated.
- La page affiche le résultat et propose le téléchargement.
- Les tests rejouent les parcours importants pour éviter les régressions.

## 4. Outils utilisés

| Outil | Utilisation |
| --- | --- |
| Flask | Routage, formulaires, sessions et rendu des pages. |
| Jinja2 | Templates HTML avec données envoyées par Flask. |
| Pillow | Lecture, transformation et export d'images. |
| Pandas / NumPy | Préparation des données CSV et calculs numériques. |
| Matplotlib | Création des visualisations exportées en image. |
| PyDub / ffmpeg | Traitement audio lorsque l'environnement le permet. |
| unittest | Tests fonctionnels des pages et traitements principaux. |
| Git / GitHub | Historique de travail, dépôt final et preuves de collaboration. |

## 5. Challenges et solutions

| Challenge | Solution |
| --- | --- |
| Le fichier app.py était trop chargé | La logique a été déplacée vers une fabrique Flask, des fichiers de routes et des services simples. Le projet est plus facile à lire et à expliquer. |
| L'interface devait paraître terminée | Les textes ont été harmonisés en français et le design a été rendu plus sobre pour ressembler à un vrai outil étudiant. |
| Les exports pouvaient encombrer le dossier static | Le stockage a été isolé, les imports temporaires sont supprimés et la galerie utilise une pagination. |
| Le module audio dépend de ffmpeg | L'application détecte l'état de l'audio et reste utilisable même quand ffmpeg n'est pas disponible. |
| Les tests devaient suivre la nouvelle architecture | Les chemins de patch et les assertions ont été adaptés après la séparation des routes. |

## 6. Équipe

| Photo | Membre | Rôle |
| --- | --- | --- |
| ![Aya EL Amrani](../../static/Admins/aya.jpeg) | Aya EL Amrani | Structure Flask, configuration, formulaires, stockage et routes. |
| ![Khadija Baskar](../../static/Admins/khadija.jpeg) | Khadija Baskar | Textes français, libellés, cohérence des intitulés et contenu des pages. |
| ![Hossam OUammi](../../static/Admins/hossam.jpeg) | Hossam OUammi | Intégration Flask, interface, médias, galerie et pages de présentation. |
| ![Abderrahmane El Garti](../../static/Admins/abdo.jpg) | Abderrahmane El Garti | Tests fonctionnels, documentation, rapport et analyse technique. |

## 7. Extraits du suivi Git

### Aya EL Amrani
- 2026-03-04 - `38dcb88` - Configurer le studio 1
- 2026-03-09 - `e9530e0` - Configurer le studio 2
- 2026-03-13 - `554b29d` - Configurer le studio 3
- 2026-03-18 - `a916a2c` - Configurer le studio 4
- 2026-03-21 - `36b6cd5` - Structurer les formulaires 1

### Khadija Baskar
- 2026-03-05 - `f070d4a` - Renseigner les libelles 1
- 2026-03-10 - `7a89d74` - Renseigner les libelles 2
- 2026-03-15 - `ed9c1e3` - Renseigner les libelles 3
- 2026-03-20 - `5572214` - Renseigner les libelles 4
- 2026-03-27 - `534b7c2` - Renseigner les libelles 5

### Hossam OUammi
- 2026-03-05 - `3abd34a` - Monter la fabrique Flask 1
- 2026-03-06 - `318e017` - Monter la fabrique Flask 2
- 2026-03-12 - `d1afc55` - Monter la fabrique Flask 3
- 2026-03-14 - `e0362ff` - Monter la fabrique Flask 4
- 2026-03-19 - `1713a6b` - Monter la fabrique Flask 5

### Abderrahmane El Garti
- 2026-03-05 - `4435dcb` - Rediger analyse projet 1
- 2026-03-11 - `e4ff2a0` - Rediger analyse projet 2
- 2026-03-17 - `81813a2` - Rediger analyse projet 3
- 2026-03-24 - `e1b497b` - Rediger analyse projet 4
- 2026-03-28 - `c4d2401` - Rediger analyse projet 5
