# Rapport du projet

## 1. Idée générale

Le projet est un studio numérique en Flask. Il regroupe plusieurs petits ateliers :

- art génératif ;
- visualisation de données ;
- traitement d’image ;
- traitement audio optionnel ;
- galerie des fichiers générés.

Le choix principal a été de garder un niveau étudiant : des fonctions nommées clairement, des fichiers courts et des pages faciles à expliquer.

## 2. Architecture

Le point d’entrée `app.py` est volontairement minimal. Il importe la fonction `create_app()` depuis `studio/app_factory.py`.

Les responsabilités sont séparées ainsi :

- `studio/config.py` : chemins, constantes et configuration Flask ;
- `studio/forms.py` : lecture et validation simple des formulaires ;
- `studio/storage.py` : import, suppression, pagination et nettoyage des fichiers ;
- `studio/security.py` : protection CSRF ;
- `studio/labels.py` : libellés français de l’interface ;
- `studio/routes/` : routes Flask séparées par fonctionnalité ;
- `modules/` : logique métier de génération, données, image et audio.

Cette organisation évite que toute la logique soit concentrée dans un seul fichier.

## 3. Interface

L’interface a été retravaillée en français avec des accents corrects. Le style visuel est devenu plus administratif :

- fond clair ;
- navigation simple ;
- cartes sobres ;
- boutons sans effets exagérés ;
- rayon de bordure réduit ;
- couleurs neutres avec quelques accents bleu, vert et orange.

Le résultat est plus proche d’un outil de projet étudiant que d’une page promotionnelle.

## 4. Fonctionnalités

### Atelier génératif

L’utilisateur choisit une série, une palette, un fond, une graine, une taille de canevas et plusieurs paramètres visuels.

Trois séries sont disponibles :

- constellation ;
- mosaïque ;
- cinétique.

### Données visuelles

Le module accepte un CSV ou utilise un jeu de données de démonstration. Les données numériques sont nettoyées, lissées et transformées en image.

### Outils médias

Le module image applique des effets comme noir et blanc, sépia, contours, glitch, rotation ou palette dominante.

Le module audio reste optionnel, car il dépend de `ffmpeg`.

### Galerie

La galerie liste les images et fichiers audio générés avec une pagination simple et des liens de téléchargement.

## 5. Tests

Les tests vérifient :

- le rendu des pages principales ;
- la génération d’un visuel ;
- l’API de prévisualisation ;
- la création d’une visualisation de données ;
- le traitement d’image ;
- la pagination de la galerie ;
- la protection CSRF ;
- le nettoyage des fichiers temporaires.

Commande utilisée :

```powershell
python -m unittest discover -s tests -v
```

## 6. Limites assumées

Le projet reste volontairement simple :

- pas de comptes utilisateurs ;
- pas de base de données ;
- pas de stockage permanent avancé ;
- pas de système de rôles ;
- pas de dépendance lourde côté frontend.

Ces limites sont cohérentes avec un projet scolaire : l’objectif est de montrer la compréhension de Flask, des formulaires, des fichiers, de la génération visuelle et de l’organisation du code.
