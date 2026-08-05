# Commencer le projet sans jargon

Ce document explique quoi faire, dans quel ordre, pour que tout le groupe avance au même rythme.

## L'idée générale

On part d'un jeu de données public déjà annoté. On en garde une petite partie pour travailler vite en local, on l'importe dans le format du projet, puis on compare deux modèles YOLO :

1. le modèle déjà entraîné sur COCO ;
2. le même modèle après entraînement sur notre sous-ensemble.

Le but n'est pas d'avoir le plus grand jeu possible. Le but est de montrer si le modèle s'améliore sur les véhicules petits et éloignés.

## Ce que chacun peut faire

| Travail | Résultat attendu |
|---|---|
| 1 | Télécharger le sous-ensemble BMD-45 et vérifier qu'il existe bien dans `data/raw/bmd45_subset/` |
| 2 | Lancer l'import et l'entraînement |
| 3 | Faire les tests, les tableaux de résultats et les images pour la présentation |

## Commandes principales

Depuis la racine du projet :

```bash
python scripts/download_bmd45_subset.py --train 2400 --val 600 --seed 42
python scripts/import_dataset.py --source data/raw/bmd45_subset
python scripts/evaluate.py --model models/yolov8n.pt --split test
python scripts/train_yolo.py
python scripts/compare_models.py
```

## Ce qu'il faut regarder

- Les dossiers `data/raw/`, `data/dataset/`, `models/finetuned/` et `results/`.
- Le fichier `configs/import.yaml` pour dire quelles classes du jeu source correspondent à nos classes.
- Le fichier `configs/entrainement.yaml` pour régler l'entraînement.

## Ce qu'il ne faut pas faire

- Ne pas modifier `data/dataset/` à la main.
- Ne pas écraser `models/yolov8n.pt`.
- Ne pas garder des classes sans équivalent dans notre comparaison si elles ne sont pas traitées dans `configs/import.yaml`.

## Si vous voulez publier sur GitHub

- Les fichiers de code et de documentation doivent être commités.
- Les gros fichiers de données et les poids de modèles ne doivent pas être poussés.
- Le dossier `data/` reste local.

Commandes simples pour publier votre travail :

```bash
git status
git add README.md docs/COMMENCER.md docs/PROCEDURE.md docs/STRUCTURE.md configs/import.yaml scripts/download_bmd45_subset.py
git commit -m "Add BMD-45 subset workflow and beginner guide"
git push origin main
```

Si votre branche ne s'appelle pas `main`, remplacez-la par le nom réel de votre branche.

## Le message important à retenir

On n'essaie pas de prouver que YOLOv8 est parfait.
On essaie de montrer, avec des chiffres, si un entraînement sur un petit sous-ensemble aide à mieux détecter les véhicules difficiles à voir.
