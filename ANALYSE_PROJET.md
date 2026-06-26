# Analyse approfondie du projet

## Résumé

La base du projet fonctionnait déjà: les pages existaient, les modules produisaient des résultats et les tests validaient les grands parcours. Le vrai sujet était ailleurs. Le code avait besoin d’être plus lisible, et l’interface devait donner une impression de projet terminé plutôt que de prototype assemblé rapidement.

La version finale garde les mêmes idées de départ, mais elle les présente mieux: routes séparées, textes en français, pages plus sobres, rapport plus court et tests faciles à lancer.

## Diagnostic initial

### 1. Un fichier principal trop chargé

Au départ, `app.py` portait beaucoup trop de responsabilités: configuration, routes, formulaires, sécurité, manipulation de fichiers et appels aux modules. C’est un classique dans un petit projet Flask: ça marche au début, puis ça devient difficile à expliquer.

La correction a consisté à déplacer chaque responsabilité vers un fichier clair. `app.py` sert maintenant de point d’entrée, et la logique applicative vit dans `studio/` et `modules/`.

### 2. Une interface à rendre plus cohérente

L’application avait besoin d’une voix unique. Les libellés ont été repris en français et les intitulés ont été harmonisés. Le style visuel a aussi été simplifié: moins d’effets décoratifs, plus d’espace, des cartes lisibles et une navigation qui reste stable.

Ce choix colle mieux au contexte du projet: on présente un outil scolaire fonctionnel, pas une page publicitaire.

### 3. Des tests à adapter à la nouvelle structure

Après la séparation des routes, certains tests ne pouvaient plus pointer vers les anciens chemins. Ils ont été ajustés pour suivre la nouvelle architecture sans réduire la couverture.

La suite vérifie notamment les pages principales, les formulaires protégés, la génération, la galerie et le nettoyage des fichiers temporaires.

## Architecture finale

L’organisation suit une logique simple:

- `app.py` lance l’application ;
- `studio/app_factory.py` crée Flask et enregistre les routes ;
- `studio/routes/` contient les vues par domaine ;
- `studio/forms.py` lit les paramètres ;
- `studio/storage.py` s’occupe des fichiers ;
- `studio/security.py` protège les formulaires ;
- `studio/team.py` prépare les profils de l’équipe ;
- `modules/` garde les traitements eux-mêmes.

Cette séparation évite de mélanger interface, validation et génération artistique. Elle rend aussi la soutenance plus confortable: chaque fichier a une raison d’être.

## Parcours utilisateur

L’utilisateur arrive sur le tableau de bord, choisit un atelier, règle quelques paramètres, lance une génération ou un traitement, puis retrouve le résultat dans la galerie. Les fichiers sont sauvegardés dans `static/generated`, et les fichiers importés temporairement sont supprimés après usage.

Le parcours reste volontairement court. C’est ce qui permet de tester et de présenter l’application sans préparation compliquée.

## Répartition du travail

### Aya EL Amrani

Travail sur la structure Flask: configuration, formulaires, stockage et séparation des routes.

### Khadija Baskar

Travail sur les textes visibles: libellés français, cohérence des intitulés et contenu des pages.

### Hossam OUammi

Travail sur l’intégration, le design administratif, les pages de présentation, les médias et la galerie.

### Abderrahmane El Garti

Travail sur les tests, la documentation, le rapport et l’analyse technique.

## Points maîtrisés

- L’application se lance avec `python app.py`.
- Les dépendances sont listées dans `requirements.txt`.
- Le README contient les étapes d’installation et de test.
- Le rapport final respecte la limite de 2-3 pages.
- Les photos de l’équipe sont chargées automatiquement depuis `static/Admins`.
- La suite `unittest` valide les parcours principaux.

## Conclusion

La version finale est plus propre, mais elle reste à taille humaine. C’est important pour ce type de projet: le code doit montrer que l’on comprend Flask, les formulaires, les fichiers et la génération de contenu, sans donner l’impression d’avoir caché la logique dans une architecture trop lourde.
