# Analyse approfondie du projet

## Résumé

Le projet avait déjà une base fonctionnelle : les pages existaient, les modules de génération étaient présents et les tests couvraient les principaux parcours.

Le problème principal était la lisibilité. Le fichier `app.py` contenait trop de responsabilités : configuration, routes, validation de formulaires, sécurité, manipulation de fichiers et logique de rendu. Pour un projet étudiant, cela rendait l’explication difficile.

La refactorisation a donc gardé les fonctionnalités, mais a rendu l’organisation plus pédagogique.

## Diagnostic initial

### 1. Fichier principal trop dense

`app.py` mélangeait :

- la configuration Flask ;
- les constantes de chemins ;
- les fonctions de validation ;
- les fonctions de nettoyage ;
- les routes ;
- la logique de génération ;
