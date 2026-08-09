# Détection des véhicules éloignés, de petite taille et dans les embouteillages

**Scène de trafic urbain appliquée au contexte béninois : proposition d'amélioration basée sur YOLOv8**

**AMA, Cohorte 2, Projet Intégrateur 1, Groupe 5**
Aïchatou TRAORE, Benoît DJOSSOU, Andréa AFOUDA

Document de référence : [Rapport technique](docs/ressources/Rapport_technique_v2.pptx)

---

## 1. Problématique

Les modèles YOLO offrent une détection rapide et performante, mais leurs
performances diminuent lorsque les véhicules sont éloignés, de petite taille, ou
masqués dans les embouteillages. Dans les scènes de trafic urbain au Bénin :

- les véhicules éloignés occupent peu de pixels ;
- plusieurs véhicules sont regroupés ou se chevauchent ;
- les motos-taxis circulent de manière moins prévisible.

Ces situations réduisent la précision de YOLOv8 et entraînent des véhicules non
détectés.

## 2. Objectif

Améliorer YOLOv8 par un fine-tuning sur un jeu de données de trafic urbain
dense, puis **mesurer l'écart** avec le modèle pré-entraîné sur le même jeu de
test, en distinguant les véhicules selon leur taille apparente.

Le livrable est le tableau comparatif produit par `scripts/compare_models.py`.

## 3. Données

Le projet utilise des **jeux de données publics déjà annotés**. Il n'y a ni
collecte ni annotation à faire.

Aucun jeu public de détection de véhicules spécifique au Bénin n'existe. Le
travail porte donc sur du trafic urbain dense d'un contexte comparable, et cette
limite doit être annoncée explicitement plutôt que contournée. Les critères de
choix et les sources candidates sont dans [docs/DATASETS.md](docs/DATASETS.md).

## 4. Périmètre

| Dans le périmètre | Hors périmètre |
|---|---|
| Détection de véhicules (voiture, moto, bus, camion) | Lecture de plaques (ANPR, OCR) |
| Fine-tuning de YOLOv8 | Suivi multi-objets et ID persistants |
| Évaluation chiffrée et comparaison | Règles d'infraction, estimation de vitesse |
| Analyse par taille d'objet | Génération de procès-verbaux, interface web |

---

## 5. Organisation du dépôt

```
Projet1/
├── README.md
├── app.py                      # application Streamlit de démonstration
├── requirements.txt            # dépendances de l'application (lues au déploiement)
├── requirements-dev.txt        # dépendances de la chaîne complète
├── .gitignore
├── assets/
│   └── exemples/               # scènes du jeu de test embarquées dans l'application
├── configs/
│   ├── import.yaml             # jeu source et correspondance des classes
│   ├── dataset.yaml            # classes et splits (régénéré automatiquement)
│   └── entrainement.yaml       # hyperparamètres du fine-tuning
├── data/
│   ├── raw/                    # jeu téléchargé, non versionné
│   ├── dataset/                # train / val / test, généré par l'import
│   └── outputs/                # images annotées par inférence
├── models/
│   ├── yolov8n.pt              # référence pré-entraînée COCO
│   └── finetuned/              # poids issus du fine-tuning
├── scripts/
│   ├── detect_image.py         # inférence qualitative sur image ou dossier
│   ├── import_dataset.py       # conversion du jeu téléchargé vers nos classes
│   ├── train_yolo.py           # fine-tuning
│   ├── evaluate.py             # métriques d'un modèle
│   └── compare_models.py       # pré-entraîné contre fine-tuné
├── notebooks/
│   └── colab_entrainement.ipynb  # chaîne complète sur Google Colab (GPU gratuit)
├── results/                    # métriques et tableaux comparatifs
└── docs/
    ├── STRUCTURE.md            # rôle de chaque dossier et de chaque fichier
    ├── PROCEDURE.md            # marche à suivre, étape par étape
    ├── DATASETS.md             # jeux de données candidats et critères de choix
    ├── GUIDE_STREAMLIT.md      # construire l'application de démonstration
    ├── GUIDE_ARTICLE_LATEX.md  # rédiger l'article scientifique avec LaTeX
    └── ressources/             # rapport technique, sujet, présentations
```

Deux documents à lire en premier : [docs/STRUCTURE.md](docs/STRUCTURE.md) pour
savoir où va quoi, [docs/PROCEDURE.md](docs/PROCEDURE.md) pour savoir quoi faire
et dans quel ordre.

---

## 6. Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

Deux fichiers de dépendances, et la distinction compte au déploiement :

| Fichier | Contenu | Quand |
|---|---|---|
| `requirements.txt` | ce dont `app.py` a besoin | installé automatiquement par Streamlit Community Cloud |
| `requirements-dev.txt` | le précédent, plus le téléchargement et l'import du jeu | travail local et Colab |

Installez `requirements-dev.txt` en local : il inclut l'autre. Sur le cloud,
seul `requirements.txt` est lu, ce qui évite d'y installer `datasets` et
`huggingface_hub`, inutiles à la démonstration.

Le modèle `yolov8n.pt` est déjà présent dans `models/`. Ultralytics le
télécharge sinon au premier appel, ce qui nécessite une connexion.

---

## 7. Utilisation

### Télécharger un sous-ensemble BMD-45 pour travailler en local

Pour éviter de télécharger tout le jeu, utilisez ce script qui crée un sous-ensemble local d'environ 3 000 images :

```bash
python scripts/download_bmd45_subset.py --train 2400 --val 600 --seed 42
```

Il écrit dans `data/raw/bmd45_subset/`, puis vous pouvez lancer l'import du projet avec `scripts/import_dataset.py`. Interrompu puis relancé, le script reprend là où il s'était arrêté sans retélécharger les images déjà présentes.

Pour un guide très simple, destiné à des débutants en vision par ordinateur, voir [docs/COMMENCER.md](docs/COMMENCER.md).

### Tout exécuter sur Google Colab (recommandé si machine modeste)

Le notebook [notebooks/colab_entrainement.ipynb](notebooks/colab_entrainement.ipynb)
déroule toute la chaîne (téléchargement, import, fine-tuning sur GPU T4,
comparaison) et sauvegarde poids et résultats sur Google Drive. L'ouvrir via
`colab.research.google.com` → GitHub → ce dépôt, puis activer le GPU
(*Exécution → Modifier le type d'exécution*).

### Vérifier l'installation, détection qualitative

```bash
python scripts/detect_image.py data/dataset/test/images/
```

Écrit une image annotée par entrée dans `data/outputs/`, suffixée du nom du
modèle. C'est ce qui permet de montrer côte à côte le même embouteillage vu par
le modèle pré-entraîné puis par le modèle fine-tuné.

### Chaîne complète

Après avoir téléchargé un jeu de données dans `data/raw/` et renseigné
`configs/import.yaml` :

```bash
python scripts/import_dataset.py
python scripts/train_yolo.py
python scripts/compare_models.py
```

`compare_models.py` écrit `results/comparaison.md`, `.csv` et `.json`.

### Évaluer un seul modèle

```bash
python scripts/evaluate.py --model models/yolov8n.pt --split test
```

### Application de démonstration

```bash
streamlit run app.py
```

L'application [app.py](app.py) compare les deux modèles côte à côte sur une
image ou une vidéo, avec une ventilation des détections par classe et par
taille apparente d'objet. Trois scènes du jeu de test sont embarquées dans
`assets/exemples/` pour que la démo fonctionne sans photo sous la main.

**Déploiement sur Streamlit Community Cloud** : sur
[share.streamlit.io](https://share.streamlit.io), se connecter avec GitHub,
« New app », choisir ce dépôt, la branche `main` et le fichier `app.py`.
Tout le nécessaire est versionné (poids fine-tunés, `requirements.txt` avec
`opencv-python-headless`, `.python-version`) ; chaque `git push` redéploie
automatiquement. L'inférence se fait sur CPU : compter 1 à 2 s par image.

---

## 8. Classes et comparabilité

Le projet retient 4 classes, volontairement alignées sur des classes COCO :

| Classe locale | Indice local | Classe COCO | Indice COCO |
|---|---|---|---|
| voiture | 0 | car | 2 |
| moto | 1 | motorcycle | 3 |
| bus | 2 | bus | 5 |
| camion | 3 | truck | 7 |

Cet alignement est ce qui rend la comparaison possible, et il se joue à deux
endroits :

- **À l'import.** Un jeu téléchargé a ses propres classes, dans son propre
  ordre. `import_dataset.py` les remappe vers les nôtres et refuse d'importer
  tant qu'une classe source n'est pas explicitement traitée, en cible ou en
  `ignorer`. Une classe oubliée fausserait silencieusement toutes les métriques.
- **À l'évaluation.** Le modèle pré-entraîné raisonne sur 80 classes indexées
  différemment ; `evaluate.py` réindexe les labels du jeu de test à la volée
  vers l'espace de classes du modèle évalué, de sorte que les deux modèles sont
  mesurés sur exactement les mêmes images et les mêmes boîtes de référence.

Sans ces deux précautions, les indices ne correspondent pas et les métriques du
modèle pré-entraîné sont fausses.

---

## 9. Métriques

Celles annoncées dans le rapport technique :

| Métrique | Ce qu'elle mesure |
|---|---|
| mAP@0.5 | Qualité de détection à un recouvrement permissif |
| mAP@0.5:0.95 | Qualité moyenne sur des recouvrements de plus en plus stricts |
| Precision | Part des détections qui sont justes |
| Recall | Part des véhicules réels effectivement trouvés |
| F1-score | Moyenne harmonique des deux précédentes |
| FPS | Débit d'inférence, mesuré sur le jeu de test |

**Plus la mesure propre au sujet : le rappel par taille d'objet.** La
problématique porte sur les véhicules éloignés et de petite taille ; une moyenne
globale masque précisément cet effet. Un gain de mAP peut très bien provenir des
seuls gros véhicules au premier plan, auquel cas le projet n'a rien résolu.

Les seuils suivent la convention COCO (petit sous 32x32 pixels, moyen sous
96x96, grand au-delà), mais les aires sont **ramenées à la résolution d'entrée
du réseau** avant d'être classées. Sans cette normalisation, des photos de
plusieurs milliers de pixels de large rangent la totalité des véhicules dans la
catégorie « grand » et l'analyse ne mesure plus rien. Ce point est vérifié :
c'est exactement ce qui s'est produit lors du premier essai.

---

## 10. Comment lire les résultats

La ligne décisive du tableau comparatif est **le rappel sur les objets petits**.
C'est elle qui répond à la question posée par le sujet.

Trois lectures possibles, toutes présentables :

- Le rappel des petits objets augmente nettement, celui des grands bouge peu :
  c'est le résultat attendu, le fine-tuning a bien corrigé le point visé.
- Toutes les tailles progressent de manière comparable : le gain vient de
  l'adaptation au domaine (couleurs, cadrage, densité) plus que de la taille.
  C'est un résultat honnête, il faut le dire ainsi.
- Rien ne progresse : le jeu retenu ressemble déjà beaucoup aux données
  d'entraînement de COCO, ou il est trop petit. Rapportez le nombre d'images et
  d'instances par classe, ainsi que la nature du jeu.

Un résultat mesuré et expliqué vaut mieux qu'un résultat impressionnant sans
protocole. Si les chiffres sont mauvais, présentez-les avec le diagnostic.

---

## 11. Licences et données

- Vérifiez la licence du jeu de données retenu et citez-la dans le rapport. Une
  licence non spécifiée n'est pas une autorisation.
- Le jeu téléchargé n'est pas versionné, le `.gitignore` l'exclut : il pèse
  souvent plusieurs giga-octets et se retélécharge à l'identique.
- Les images de trafic montrent des plaques et des visages, donc des données à
  caractère personnel. Floutez-les sur toute capture destinée aux slides.
