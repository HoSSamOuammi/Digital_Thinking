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
- les messages utilisateur.

Ce type de fichier fonctionne, mais il est difficile à présenter devant un jury parce qu’il n’y a pas de séparation claire.

### 2. Interface en anglais

Les pages étaient majoritairement en anglais. Pour un projet présenté en français, cela pose deux problèmes :

- les textes ne correspondent pas au contexte de présentation ;
- les accents français risquent d’être oubliés si la traduction est faite trop tard.

L’interface est maintenant en français, avec des textes plus naturels.

### 3. Design trop créatif

L’ancien style utilisait des effets décoratifs, des grands titres et des gradients. Cela donnait une impression de page créative, mais moins d’outil administratif.

Le nouveau design est plus simple :

- grille claire ;
- cartes sobres ;
