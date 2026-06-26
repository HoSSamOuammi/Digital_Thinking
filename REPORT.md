# Rapport du projet

## 1. Idée générale

Nous avons construit un studio numérique avec Flask. L’application regroupe plusieurs ateliers dans une même interface : création de visuels génératifs, visualisation de données, traitement d’image, traitement audio optionnel et galerie des fichiers produits.

Le but n’était pas d’impressionner avec une architecture compliquée. Le projet devait surtout être clair, fonctionner correctement et pouvoir être expliqué sans se perdre dans des détails inutiles.

## 2. Architecture

Le fichier `app.py` est volontairement court. Il crée l’application avec `create_app()` puis lance Flask. Le reste est rangé dans des fichiers plus spécialisés :

- `studio/config.py` contient les chemins, constantes et paramètres Flask ;
- `studio/forms.py` lit et vérifie les valeurs envoyées par les formulaires ;
- `studio/storage.py` gère les fichiers, la pagination et le nettoyage ;
- `studio/security.py` s’occupe du jeton CSRF ;
- `studio/labels.py` garde les libellés français au même endroit ;
- `studio/routes/` sépare les pages par fonctionnalité ;
- `modules/` contient la logique métier: génération, données, image et audio.

Cette séparation rend le projet plus simple à défendre. Quand on ouvre un fichier, on comprend assez vite ce qu’il est censé faire.

## 3. Interface

L’interface a été ramenée vers quelque chose de sobre: fond clair, navigation lisible, cartes simples, boutons sans effets excessifs. On a préféré une présentation de type outil de travail plutôt qu’une page trop décorative.

Les textes visibles sont en français. Ce point compte beaucoup pour la présentation, parce que l’utilisateur ne devrait pas avoir à deviner le rôle d’un bouton ou d’un réglage.

## 4. Fonctionnalités

### Atelier génératif

L’utilisateur choisit une série visuelle, une palette, un fond, une graine et plusieurs réglages. Il peut prévisualiser le rendu puis exporter l’image finale.

### Données visuelles

Le module accepte un CSV ou utilise des données de démonstration. Les valeurs numériques sont nettoyées puis transformées en image.

### Outils médias

La partie image applique des effets comme noir et blanc, sépia, contours, glitch, rotation ou palette dominante. L’audio reste optionnel, car il dépend de `ffmpeg`.

### Galerie

La galerie liste les images et les fichiers audio générés. Elle sert de trace concrète du parcours complet: créer, sauvegarder, retrouver, télécharger.

## 5. Tests

Les tests couvrent les routes principales, la génération, la prévisualisation, le traitement d’image, la visualisation de données, la pagination de la galerie, la protection CSRF et le nettoyage des fichiers temporaires.

Commande utilisée :

```powershell
python -m unittest discover -s tests -v
```

## 6. Limites assumées

Le projet reste volontairement simple :

- pas de comptes utilisateurs ;
- pas de base de données ;
- pas de système de rôles ;
- pas de framework JavaScript côté client ;
- pas de stockage permanent avancé.

Ces limites ne sont pas des oublis. Elles permettent de concentrer le projet sur Flask, les formulaires, les fichiers, la génération visuelle et l’organisation du code.
