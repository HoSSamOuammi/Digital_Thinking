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
