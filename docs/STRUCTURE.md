# Structure du dépôt : rôle de chaque dossier et de chaque fichier

Ce document répond à une seule question : à quoi sert ce fichier, et qui doit y
toucher.

Convention utilisée dans les tableaux :

- **Vous** : fichier que vous remplissez ou modifiez à la main.
- **Script** : fichier généré automatiquement, à ne pas éditer.
- **Versionné** : suivi par Git, part avec le dépôt.
- **Local** : ignoré par Git, reste sur votre machine.

---

## Vue d'ensemble

```
Projet1/
├── README.md               Présentation du projet
├── requirements.txt        Dépendances Python
├── .gitignore              Ce que Git doit ignorer
├── configs/                Réglages
├── data/                   Images, à toutes les étapes
├── models/                 Poids des modèles
├── scripts/                Le code
├── results/                Métriques et tableau comparatif
└── docs/                   Documentation et documents sources
```

Le flux général va de gauche à droite :

```
data/raw       ->  data/dataset  ->  models/finetuned  ->  results
(jeu téléchargé)   (converti)        (modèle entraîné)     (chiffres)
```

---

## Racine

| Fichier | Rôle | Qui | Statut |
|---|---|---|---|
| `README.md` | Présentation du projet : problématique, objectif, périmètre, commandes principales. Le premier fichier que lit une personne extérieure. | Vous | Versionné |
| `requirements.txt` | Liste figée des versions de bibliothèques. Permet de reproduire l'environnement à l'identique sur une autre machine ou sur Colab. | Vous | Versionné |
| `.gitignore` | Empêche Git de versionner les images et les poids de modèles. Un jeu de données téléchargé pèse souvent plusieurs giga-octets et n'a pas à entrer dans le dépôt. | Vous | Versionné |

---

## `configs/` : les réglages

Aucun seuil, aucun hyperparamètre ne doit être écrit en dur dans le code. Tout
vit ici. C'est ce qui rend un entraînement reproductible : un run est
entièrement décrit par son fichier de configuration.

| Fichier | Rôle | Qui | Statut |
|---|---|---|---|
| `import.yaml` | Le chemin du jeu téléchargé, et la correspondance entre ses classes et nos 4 classes. **Le fichier le plus important du dossier** : c'est lui qui garantit que la comparaison avec le modèle pré-entraîné est valide. | Vous | Versionné |
| `dataset.yaml` | Décrit le jeu converti à Ultralytics : où sont les images d'entraînement, de validation et de test, et quelles sont les 4 classes. **Régénéré à chaque import**, ne l'éditez pas à la main. | Script | Versionné |
| `entrainement.yaml` | Tous les hyperparamètres du fine-tuning : nombre d'epochs, taille d'image, batch, augmentation des données. C'est le fichier que vous modifiez pour tenter un réglage différent. | Vous | Versionné |

---

## `data/` : les images à chaque étape

| Dossier | Rôle | Qui | Statut |
|---|---|---|---|
| `raw/` | Le jeu de données téléchargé et décompressé, tel quel. On n'y touche plus une fois déposé. Contient aussi les quelques photos d'embouteillage servant aux visuels. | Vous | Local |
| `dataset/` | Le jeu converti : classes remappées vers les nôtres, splits train / val / test normalisés. Construit automatiquement à partir de `raw/`. | Script | Local |
| `outputs/` | Images annotées produites par l'inférence, pour les visuels de la présentation. | Script | Local |

**Ne modifiez jamais `dataset/` à la main.** Il est effacé et reconstruit à
chaque exécution de `import_dataset.py`. Tout ce que vous y ajouteriez serait
perdu à l'import suivant.

Rien de ce dossier n'est versionné : tout se régénère en relançant l'import,
à condition d'avoir conservé `configs/import.yaml`.

---

## `models/` : les poids

| Élément | Rôle | Qui | Statut |
|---|---|---|---|
| `yolov8n.pt` | Le modèle YOLOv8 pré-entraîné sur COCO. C'est votre **point de comparaison**, celui dont vous devez démontrer les limites. Ne l'écrasez jamais. | Fourni | Local |
| `finetuned/` | Les poids issus de vos entraînements. `train_yolo.py` y copie automatiquement le meilleur modèle de chaque run, sous un nom stable. | Script | Local |

Les poids ne sont pas versionnés : un fichier `.pt` pèse plusieurs méga-octets
et se régénère par entraînement.

---

## `scripts/` : le code

Les cinq scripts correspondent aux étapes de la procédure. Chacun s'exécute
seul, en ligne de commande, et accepte `--help`.

| Script | Ce qu'il fait | Ce qu'il lit | Ce qu'il écrit |
|---|---|---|---|
| `detect_image.py` | Détection sur une image ou un dossier. Sert aux visuels qualitatifs : le même embouteillage vu par les deux modèles. | `data/raw/` | `data/outputs/` |
| `import_dataset.py` | Convertit le jeu téléchargé : remappe ses classes vers les nôtres, normalise les splits, garantit un jeu de test isolé. | `data/raw/`, `configs/import.yaml` | `data/dataset/`, `configs/dataset.yaml` |
| `train_yolo.py` | Le fine-tuning. Tous ses réglages viennent de `configs/entrainement.yaml`. | `data/dataset/` | `models/finetuned/`, `results/entrainements/` |
| `evaluate.py` | Calcule les métriques d'un seul modèle, y compris le rappel ventilé par taille d'objet. | `data/dataset/`, un modèle | `results/eval_*.json` |
| `compare_models.py` | Évalue les deux modèles sur le même jeu de test et produit le tableau comparatif. **C'est le livrable du projet.** | `data/dataset/`, deux modèles | `results/comparaison.*` |

Deux points de conception qui répondent à des questions de jury probables :

- `compare_models.py` importe `evaluate.py` : les deux modèles passent donc
  exactement par le même code de mesure.
- `import_dataset.py` refuse d'importer si une classe du jeu source n'est pas
  explicitement traitée. Une classe oubliée fausserait silencieusement toutes
  les métriques ; mieux vaut un arrêt qu'un résultat faux.

---

## `results/` : les chiffres

| Fichier | Rôle |
|---|---|
| `comparaison.md` | Le tableau comparatif, prêt à être repris dans le rapport et les slides. |
| `comparaison.csv` | Les mêmes valeurs, pour retraitement dans un tableur. |
| `comparaison.json` | Toutes les valeurs brutes, y compris le détail par taille d'objet. |
| `eval_<modele>_<split>.json` | Le rapport détaillé d'un modèle évalué seul. |
| `entrainements/` | Les journaux et courbes produits par Ultralytics pendant l'entraînement. Volumineux, non versionné. |

Les trois fichiers `comparaison.*` sont versionnés : ce sont les résultats du
projet, ils doivent survivre à un changement de machine.

---

## `docs/` : la documentation

| Élément | Rôle |
|---|---|
| `STRUCTURE.md` | Ce document. |
| `PROCEDURE.md` | La marche à suivre, étape par étape, pour mener le projet à son terme. |
| `DATASETS.md` | Les jeux de données candidats, leurs caractéristiques et leurs licences. |
| `COMMENCER.md` | Guide simple pour débutants : quoi faire, dans quel ordre, sans jargon compliqué. |
| `ressources/` | Les documents sources : sujet, rapport technique, présentations. Documents de référence, pas de la documentation de code. |

---

## Ce qu'il ne faut pas perdre

Le projet repose sur des données publiques : les images, les labels et les poids
se retéléchargent ou se régénèrent. Trois fichiers, en revanche, ne se
reconstituent pas tout seuls.

| Fichier | Pourquoi il compte |
|---|---|
| `configs/import.yaml` | Contient la correspondance de classes que vous avez établie. Le perdre oblige à refaire ce travail et, surtout, à risquer un mapping différent qui rendrait vos anciens résultats incomparables aux nouveaux. |
| `configs/entrainement.yaml` | Décrit entièrement le run qui a produit vos chiffres. Sans lui, vos résultats ne sont pas reproductibles. |
| `results/comparaison.*` | Les résultats eux-mêmes. Un réentraînement ne redonne pas exactement les mêmes valeurs. |

Ces trois-là sont versionnés, précisément pour cette raison. Vérifiez qu'ils
sont bien commités avant la soutenance.
