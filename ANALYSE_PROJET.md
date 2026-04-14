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
- boutons plats ;
- couleurs modérées ;
- meilleure lisibilité.

### 4. Tests liés à l’ancienne structure

Certains tests utilisaient des chemins de patch comme `app.create_generative_art`. Après la séparation des routes, ces chemins devaient pointer vers les nouveaux modules.

Les tests ont été mis à jour sans réduire la couverture.

## Architecture finale

### Couche 1 : point d’entrée

`app.py` reste très court. Il sert seulement à créer et lancer l’application.

### Couche 2 : création de l’application

`studio/app_factory.py` crée l’application Flask, charge la configuration, active la sécurité CSRF et enregistre les routes.

### Couche 3 : routes

Les routes sont séparées par usage :

- `pages.py` : accueil, équipe, galerie, téléchargement ;
- `generative_routes.py` : atelier génératif et prévisualisation ;
- `data_routes.py` : visualisation de données ;
- `media_routes.py` : image et audio.

Chaque fichier reste plus facile à lire qu’un grand fichier central.

### Couche 4 : services simples

Les fichiers suivants évitent la répétition :

- `forms.py` : convertir et valider les valeurs des formulaires ;
- `storage.py` : sauvegarder, lister, paginer et nettoyer les fichiers ;
- `security.py` : gérer le jeton CSRF ;
- `labels.py` : centraliser les libellés français ;
- `team.py` : préparer les profils des membres.

### Couche 5 : modules métier

Le dossier `modules/` garde les traitements principaux :

- génération artistique ;
- visualisation de données ;
- traitement d’image ;
- traitement audio.

Cette séparation est importante : les routes Flask ne devraient pas contenir les algorithmes artistiques eux-mêmes.

## Pourquoi cette version est plus adaptée au niveau étudiant

Le projet évite les abstractions trop avancées. Il n’utilise pas de base de données, de système d’authentification ou de framework frontend complexe.

Les choix sont simples à expliquer :

- une fonction pour créer l’application ;
- un fichier par groupe de routes ;
- un fichier pour les formulaires ;
- un fichier pour les fichiers ;
- des modules métier séparés.

Cela montre une bonne organisation sans donner l’impression d’un projet trop sophistiqué.

## Parcours utilisateur

### Accueil

La page d’accueil sert de tableau de bord avec quelques compteurs et des accès rapides.

### Atelier génératif

L’utilisateur choisit les paramètres puis génère un visuel. La prévisualisation permet de vérifier rapidement le rendu.

### Données

L’utilisateur peut importer un CSV ou utiliser les données de démonstration. Le résultat est exporté en image.

### Médias

L’utilisateur peut importer une image et appliquer un effet. L’audio est proposé seulement si l’environnement le permet.

### Galerie

Les fichiers générés sont listés avec pagination et téléchargement.

## Répartition en quatre parties

### Aya EL Amrani

Travail sur la structure Flask :

- extraction de la configuration ;
- extraction des formulaires ;
- extraction du stockage ;
- séparation des routes.

### Khadija Baskar

Travail sur la langue et les contenus :

- libellés français centralisés ;
- traduction des templates ;
- correction des messages visibles ;
- respect des accents.

### Hossam OUammi

Travail sur l’interface :

- passage à un style administratif ;
- réduction des effets décoratifs ;
- navigation plus sobre ;
- amélioration de la lisibilité.

### Abderrahmane El Garti

Travail sur la qualité :

- adaptation des tests ;
- vérification de la suite ;
- documentation ;
- analyse finale.

## Conclusion

La version finale garde l’idée créative du projet, mais elle est plus propre à expliquer. Le code est séparé, l’interface est en français, le style est plus administratif et les tests confirment que les parcours principaux fonctionnent.
