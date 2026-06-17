# Rapport technique

## Projet

Studio génératif interactif est une application Flask qui regroupe plusieurs ateliers numériques.
L’utilisateur peut générer des images, transformer des données, traiter des fichiers médias et consulter les exports dans une galerie.

## Fonctionnalités

### Tableau de bord

La page d’accueil donne une vue rapide sur l’état du studio : nombre d’images générées, fichiers audio disponibles, palettes intégrées et accès directs vers les ateliers. Elle sert de point d’entrée pour présenter le projet sans obliger l’utilisateur à connaître la structure interne.

### Atelier génératif

L’atelier génératif permet de produire des visuels à partir de paramètres contrôlés : série visuelle, fond, palette, graine, nombre de formes, densité, taille du canevas et dessin d’accents sur l’aperçu. La prévisualisation donne un retour rapide avant l’export final.

### Visualisation de données

Le module de données accepte un fichier CSV ou utilise un jeu de données de démonstration. Les colonnes numériques sont nettoyées, lissées et transformées en visuels : vue complète, paysage, carte thermique, barres graduées ou rayonnement circulaire.

### Outils médias

La partie médias permet de traiter une image avec plusieurs effets : noir et blanc, sépia, inversion, flou, contours, pixelisation, miroir, rotation, glitch, aquarelle et palette dominante. Le traitement audio est prévu quand l’environnement dispose de PyDub et ffmpeg.

### Galerie

La galerie regroupe les fichiers générés dans l’application. Les images et fichiers audio sont listés séparément, avec pagination et liens de téléchargement. Cette page sert aussi de preuve visuelle du flux complet : génération, sauvegarde, consultation et export.

### Équipe

La page équipe présente les membres, leurs rôles et leurs emails. Elle relie la partie fonctionnelle du projet à la répartition du travail visible dans l’historique Git.

## Architecture

- `app.py` : lancement de l’application.
- `studio/app_factory.py` : création Flask et enregistrement des routes.
- `studio/routes/` : pages séparées par fonctionnalité.
- `studio/forms.py` : lecture et validation des formulaires.
- `studio/storage.py` : gestion des fichiers.
- `modules/` : logique métier.

## Répartition

| Membre | Commits | Partie principale |
| --- | ---: | --- |
| Hossam OUammi | 25 | Intégration Flask, design administratif, module médias, galerie et pages de présentation. |
| Aya EL Amrani | 21 | Architecture applicative, configuration, formulaires, stockage, sécurité et routes principales. |
| Khadija Baskar | 16 | Traduction française, libellés, textes visibles et cohérence des intitulés. |
| Abderrahmane El Garti | 15 | Tests fonctionnels, documentation, rapport et analyse technique. |

## Suivi Git

- Dépôt : https://github.com/HoSSamOuammi/Digital_Thinking
- Période des commits : 2026-03-04 au 2026-06-12
- Total : 77 commits

### Hossam OUammi
- 2026-03-05 - `3abd34a` - Monter la fabrique Flask 1
- 2026-03-06 - `318e017` - Monter la fabrique Flask 2
- 2026-03-12 - `d1afc55` - Monter la fabrique Flask 3
- 2026-03-14 - `e0362ff` - Monter la fabrique Flask 4
- 2026-03-19 - `1713a6b` - Monter la fabrique Flask 5
- 2026-03-23 - `f1c519f` - Simplifier le point entree
- 2026-03-27 - `ede08c9` - Construire les routes medias 1
- 2026-03-30 - `7ce9f2a` - Construire les routes medias 2
- 2026-04-04 - `4c4a001` - Construire les routes medias 3
- 2026-04-05 - `e1c9ffa` - Construire les routes medias 4
- 2026-04-09 - `553e636` - Construire les routes medias 5
- 2026-04-13 - `d051fec` - Construire les routes medias 6
- 2026-04-18 - `a0131fe` - Construire equipe 1
- 2026-04-20 - `d1867ae` - Construire equipe 2
- 2026-04-23 - `2ed962b` - Construire equipe 3
- 2026-04-28 - `f2bf359` - Construire equipe 4
- 2026-05-04 - `8938d70` - Exposer create_app
- 2026-05-08 - `c9f8836` - Appliquer le design administratif
- 2026-05-16 - `d70d214` - Revoir ecran generatif
- 2026-05-20 - `111a810` - Revoir ecran medias
- 2026-05-23 - `729bebb` - Revoir galerie
- 2026-05-30 - `f9461b5` - Revoir page equipe
- 2026-06-05 - `8abba0c` - Ajouter paquet routes
- 2026-06-09 - `388da0c` - Clarifier effets image
- 2026-06-12 - `89aa4f7` - Clarifier traitement audio

### Aya EL Amrani
- 2026-03-04 - `38dcb88` - Configurer le studio 1
- 2026-03-09 - `e9530e0` - Configurer le studio 2
- 2026-03-13 - `554b29d` - Configurer le studio 3
- 2026-03-18 - `a916a2c` - Configurer le studio 4
- 2026-03-21 - `36b6cd5` - Structurer les formulaires 1
- 2026-03-26 - `00b8c30` - Structurer les formulaires 2
- 2026-03-31 - `b533c98` - Structurer les formulaires 3
- 2026-04-04 - `12e982d` - Structurer les formulaires 4
- 2026-04-07 - `0f27827` - Structurer les formulaires 5
- 2026-04-11 - `f7ff5c8` - Structurer les formulaires 6
- 2026-04-15 - `efe9e0b` - Structurer les formulaires 7
- 2026-04-21 - `f63f63e` - Organiser le stockage 1
- 2026-04-26 - `449e829` - Organiser le stockage 2
- 2026-05-01 - `461fda6` - Organiser le stockage 3
- 2026-05-05 - `251f961` - Organiser le stockage 4
- 2026-05-13 - `eba4fa8` - Proteger les formulaires 1
- 2026-05-21 - `6d5e5aa` - Proteger les formulaires 2
- 2026-05-28 - `db4f380` - Deplacer les pages 1
- 2026-06-01 - `f5797fd` - Deplacer les pages 2
- 2026-06-07 - `46db537` - Extraire la route generative
- 2026-06-12 - `ac9dada` - Extraire la route donnees

### Khadija Baskar
- 2026-03-05 - `f070d4a` - Renseigner les libelles 1
- 2026-03-10 - `7a89d74` - Renseigner les libelles 2
- 2026-03-15 - `ed9c1e3` - Renseigner les libelles 3
- 2026-03-20 - `5572214` - Renseigner les libelles 4
- 2026-03-27 - `534b7c2` - Renseigner les libelles 5
- 2026-04-03 - `9ee7ff4` - Renseigner les libelles 6
- 2026-04-06 - `8174b48` - Renseigner les libelles 7
- 2026-04-12 - `5b6bbd4` - Renseigner les libelles 8
- 2026-04-19 - `40633d6` - Renseigner les libelles 9
- 2026-04-25 - `dd310ca` - Renseigner les libelles 10
- 2026-05-03 - `ab25599` - Renseigner les libelles 11
- 2026-05-12 - `f6b2d87` - Traduire les descriptions generatives
- 2026-05-19 - `723b3ed` - Traduire les descriptions donnees
- 2026-05-29 - `2ecce19` - Traduire la navigation
- 2026-06-02 - `96ea7d3` - Traduire accueil
- 2026-06-11 - `a1eb399` - Traduire donnees visuelles

### Abderrahmane El Garti
- 2026-03-05 - `4435dcb` - Rediger analyse projet 1
- 2026-03-11 - `e4ff2a0` - Rediger analyse projet 2
- 2026-03-17 - `81813a2` - Rediger analyse projet 3
- 2026-03-24 - `e1b497b` - Rediger analyse projet 4
- 2026-03-28 - `c4d2401` - Rediger analyse projet 5
- 2026-04-05 - `b16866a` - Rediger analyse projet 6
- 2026-04-10 - `1a3f0c7` - Rediger analyse projet 7
- 2026-04-14 - `2a6c7d3` - Rediger analyse projet 8
- 2026-04-23 - `5581031` - Mettre a jour README 1
- 2026-04-30 - `7561dfd` - Mettre a jour README 2
- 2026-05-07 - `eb8616b` - Mettre a jour README 3
- 2026-05-17 - `0d132c0` - Mettre a jour rapport 1
- 2026-05-24 - `872d35a` - Mettre a jour rapport 2
- 2026-06-02 - `7e48c80` - Mettre a jour rapport 3
- 2026-06-10 - `dc55887` - Actualiser les tests
