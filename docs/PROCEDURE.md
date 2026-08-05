# Procédure : mener le projet à son terme

Marche à suivre concrète, du dépôt actuel jusqu'aux chiffres à mettre dans la
présentation.

Le projet repose sur des **jeux de données publics déjà annotés**. Aucune
collecte ni annotation n'est à faire.

Chaque étape indique ce qu'il faut faire, la commande, le temps à prévoir, et
**la diapositive du rapport technique qu'elle alimente**.

---

## Ce que la présentation affirme, et ce qu'il faut produire pour le soutenir

Le rapport technique avance cinq affirmations. Chacune doit être adossée à un
résultat, sinon elle ne tient pas devant un jury.

| Diapositive | Affirmation | Ce qui doit la soutenir | Étape |
|---|---|---|---|
| 2 et 3 | Les performances chutent sur les véhicules éloignés, petits, en embouteillage | Une mesure du modèle pré-entraîné, ventilée par taille d'objet | 3 |
| 4 | YOLOv8 offre le meilleur compromis précision / rapidité | Un argumentaire sourcé, plus votre propre mesure de FPS | 3 et 7 |
| 5 | « Nos observations montrent que YOLOv8 présente plusieurs limites » | Les chiffres de l'étape 3, pas une impression visuelle | 3 |
| 6 | Comparaison pré-entraîné contre fine-tuné sur 5 métriques | Le tableau de `compare_models.py` | 5 |
| 7 | Meilleure détection des véhicules éloignés et petits | La ligne « rappel objets petits » de ce tableau | 5 |

**Le point de vigilance principal est la diapositive 5.** Elle annonce des
observations. Tant que l'étape 3 n'est pas faite, ces observations sont des
impressions. C'est la première question qu'un jury technique posera.

---

## Une limite à annoncer vous-mêmes

Le rapport parle d'un jeu de données « représentatif du trafic urbain
béninois ». **Aucun jeu public béninois n'existe**, et vous n'en constituez pas.
Vous travaillez donc sur des jeux de trafic dense d'autres pays, choisis pour
leur proximité de conditions.

Ce n'est pas un problème, à condition de le dire. Deux formulations, l'une
défendable et l'autre non :

> À éviter : « nous avons adapté YOLOv8 au trafic béninois ».

> À dire : « faute de jeu de données béninois public, nous avons travaillé sur
> du trafic urbain dense d'un contexte comparable. Nous mesurons le gain sur les
> véhicules de petite taille, qui est le verrou identifié. La validation sur
> données béninoises est la suite naturelle de ce travail. »

Annoncer cette limite en conclusion coûte quinze secondes et vous évite d'être
pris en défaut. C'est aussi une perspective de travail futur, ce qu'un jury
apprécie.

---

## Étape 0 : préparer l'environnement

À faire une fois, par chaque membre du groupe.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Vérification immédiate :

```bash
python scripts/detect_image.py data/raw/
```

Trois images annotées doivent apparaître dans `data/outputs/`. Si c'est le cas,
l'installation est bonne.

**Durée** : 20 minutes.

---

## Étape 1 : choisir et télécharger le jeu de données

**Diapositive 6, étapes 1 et 2.** C'est ici que se joue la qualité du projet.
Le choix du jeu remplace le travail de terrain, il mérite le même soin.

Les sources candidates et leurs caractéristiques sont détaillées dans
[DATASETS.md](DATASETS.md). Les quatre critères qui comptent pour ce sujet
précisément :

| Critère | Pourquoi | Comment vérifier |
|---|---|---|
| **Beaucoup de petits objets** | C'est l'objet même de l'étude | Regardez quelques images : voit-on des véhicules au loin ? |
| **Trafic dense, véhicules qui se chevauchent** | Second axe du sujet | Cherchez des scènes d'embouteillage réel |
| **Prise de vue fixe, en hauteur** | Correspond au cas d'usage de surveillance | Vue CCTV plutôt que vue depuis un véhicule |
| **Classes proches de COCO** | Condition de validité de la comparaison | Voiture, moto, bus, camion doivent exister |

Téléchargez le jeu au **format YOLOv8** et décompressez-le dans `data/raw/`.
Si vous utilisez BMD-45 sur Hugging Face, ne récupérez pas tout le dépôt : le script [scripts/download_bmd45_subset.py](../scripts/download_bmd45_subset.py) extrait un sous-ensemble local de 3 000 images environ.

Pour les nouveaux membres du groupe, un guide très simple est aussi disponible dans [docs/COMMENCER.md](COMMENCER.md).

**Deux vérifications avant d'aller plus loin :**

- **La licence.** Notez-la, elle doit figurer dans le rapport. Les jeux Roboflow
  portent des licences hétérogènes.
- **La séparation entraînement / test.** Certains jeux publics contiennent les
  mêmes images sources dans plusieurs splits, ce qui gonfle artificiellement les
  métriques. Si le jeu que vous retenez est connu pour ce défaut, dites-le, ou
  refaites le découpage. Repérer ce genre de problème est exactement ce qu'un
  jury technique valorise.

**Durée** : une demi-journée, en regardant réellement les images.

---

## Étape 2 : importer le jeu

**Diapositive 6, étape 3.**

Un jeu téléchargé a ses propres classes, dans son propre ordre. Notre projet en
utilise quatre, alignées sur COCO, et c'est cet alignement qui rend la
comparaison possible. L'import fait la traduction.

Renseignez `configs/import.yaml` : le chemin du jeu, et la correspondance entre
ses classes et les nôtres.

```bash
python scripts/import_dataset.py
```

Au premier lancement, le script affiche **la liste réelle des classes du jeu**
et s'arrête si l'une d'elles n'est pas traitée. Recopiez-les dans
`configs/import.yaml`, en cible ou en `ignorer`, puis relancez.

Les classes sans équivalent COCO, comme les rickshaws ou les vélos, doivent être
mises à `ignorer` : les conserver fausserait la comparaison avec le modèle
pré-entraîné, qui ne sait pas les détecter.

Le script écrit dans `data/dataset/` et affiche le nombre d'instances par classe
et par split. **Lisez ce tableau.** Si une classe compte quelques unités en
test, sa métrique n'aura aucune valeur statistique : soit vous changez de jeu,
soit vous annoncez la limite. Publier un chiffre calculé sur trois objets serait
une faute.

Si le jeu source ne fournit pas de split de test, le script en prélève un sur la
validation et vous le signale. Sans jeu de test isolé, il n'y a pas de résultat.

**Durée** : 30 minutes, dont l'essentiel à remplir la correspondance.

---

## Étape 3 : mesurer le modèle pré-entraîné

**Diapositives 2, 3 et 5.** L'étape que la présentation actuelle suppose faite.

```bash
python scripts/evaluate.py --model models/yolov8n.pt --split test
```

Vous obtenez mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, FPS, **et le rappel
ventilé par taille d'objet**.

C'est cette dernière ligne qui transforme la diapositive 5 en résultat. Vous
passez de :

> « faible détection des véhicules éloignés »

à :

> « sur notre jeu de test, le modèle pré-entraîné retrouve 89 % des grands
> véhicules mais seulement 41 % des petits »

La seconde formulation est un résultat, la première une impression. Remplacez
les puces de la diapositive 5 par vos chiffres réels.

**Durée** : 10 à 30 minutes selon la taille du jeu et la machine.

---

## Étape 4 : fine-tuning

**Diapositive 6, étape 4.**

```bash
python scripts/train_yolo.py
```

Sur CPU, 50 epochs prennent plusieurs heures. Sur Google Colab avec un GPU T4,
comptez environ une heure. Lancez-le en fin de journée si vous travaillez sur
CPU.

Pour forcer le GPU sur Colab :

```bash
python scripts/train_yolo.py --device 0
```

Les réglages sont dans `configs/entrainement.yaml`. Deux paramètres comptent
particulièrement pour ce sujet, et sont à savoir expliquer :

- `mosaic: 1.0` assemble quatre images en une, ce qui réduit la taille apparente
  de chaque objet. C'est l'augmentation la plus directement liée à la
  problématique des petits véhicules.
- `scale: 0.7` autorise un zoom arrière important, qui génère lui aussi des
  objets plus petits que ceux réellement photographiés.

`degrees` et `flipud` restent à 0 : une scène routière a un haut et un bas.
Faire pivoter les images apprendrait au modèle des configurations qui n'existent
pas.

Si le jeu est volumineux, réduisez d'abord le nombre d'epochs plutôt que la
taille d'image : c'est la résolution qui conditionne la détection des petits
objets, et la baisser irait contre l'objectif du projet.

Le meilleur modèle est copié automatiquement dans `models/finetuned/`.

**Durée** : 1 heure sur Colab, une nuit sur CPU.

---

## Étape 5 : comparer

**Diapositive 6, étape 6, et diapositive 7.** C'est le résultat du projet.

```bash
python scripts/compare_models.py
```

Écrit dans `results/` : `comparaison.md`, `comparaison.csv`, `comparaison.json`.
Le tableau `comparaison.md` se reprend tel quel dans le rapport et les slides.

### Comment lire le tableau

La ligne décisive est **le rappel sur les objets petits**. C'est elle qui répond
à la question posée par le sujet. Trois lectures possibles, toutes présentables :

| Ce que vous observez | Ce que cela signifie | Ce qu'il faut dire |
|---|---|---|
| Les petits objets progressent nettement, les grands peu | Résultat attendu, le fine-tuning a corrigé le point visé | C'est votre résultat principal, mettez-le en avant |
| Toutes les tailles progressent pareillement | Le gain vient de l'adaptation au domaine, pas de la taille | Dites-le ainsi, c'est un résultat honnête et défendable |
| Rien ne progresse | Le jeu est déjà proche de COCO, ou trop petit | Rapportez le nombre d'instances par classe et la nature du jeu |

Le troisième cas est plus probable ici que si vous aviez collecté vos propres
images : un jeu public généraliste ressemble parfois beaucoup aux données
d'entraînement de COCO, et il reste alors peu de marge de progression. Si cela
se produit, le diagnostic est le résultat, et il se présente très bien.

Un résultat mesuré et expliqué vaut mieux qu'un résultat impressionnant sans
protocole.

**Durée** : 20 à 40 minutes.

---

## Étape 6 : produire les visuels

**Diapositives 5 et 7.** Un jury retient une image avant un tableau.

```bash
python scripts/detect_image.py data/raw/ --model models/yolov8n.pt
python scripts/detect_image.py data/raw/ --model models/finetuned/yolov8n_benin.pt
```

Les fichiers sont suffixés du nom du modèle, ce qui permet de les mettre côte à
côte sur une slide.

Choisissez **une seule image**, celle où la différence est la plus nette :
typiquement un embouteillage avec beaucoup de véhicules au fond. Un avant / après
sur une image bien choisie vaut mieux que six comparaisons moyennes.

Les trois photos d'embouteillage déjà présentes dans `data/raw/` conviennent
pour cet usage, même si elles ne servent pas à l'entraînement.

**Durée** : 30 minutes.

---

## Étape 7 : mettre la présentation à jour

Reprenez le rapport technique et remplacez chaque affirmation par le chiffre
correspondant.

| Diapositive | Ce qu'il y a actuellement | Ce qu'il faut y mettre |
|---|---|---|
| 3, objectif | « jeu de données représentatif du trafic béninois » | Le jeu réellement utilisé, avec sa source et sa licence |
| 5, limites | Puces qualitatives | Vos chiffres de l'étape 3, ventilés par taille |
| 6, évaluation | Liste de métriques | Ces métriques, renseignées |
| 7, résultats attendus | Attentes | Résultats obtenus, écarts en pourcentage |
| 7, conclusion | « nous espérons obtenir » | Ce que vous avez obtenu, limites comprises |

Trois ajouts qui pèsent lourd pour un coût faible :

- **La volumétrie.** Nombre d'images et d'instances par classe. Une étude sans
  volumétrie n'est pas évaluable. Le tableau est produit par l'étape 2.
- **La source et la licence du jeu.** Une ligne suffit, et son absence se
  remarque.
- **Une limite assumée.** Celle du contexte béninois, décrite plus haut, ou
  celle-ci : « au-delà d'une certaine distance le véhicule fait moins de N
  pixels et aucun modèle ne le détecte, c'est une limite physique et non un
  défaut de notre approche ». Annoncer une limite avant qu'on vous l'oppose vous
  positionne en ingénieur lucide.

---

## Répartition à trois

Découpage par livrable, pour que personne n'attende.

| Membre | Responsabilité | Livrable |
|---|---|---|
| A | Choix du jeu, import, licence, volumétrie | `data/dataset/` prêt, source documentée |
| B | Entraînement, réglages, relances | `models/finetuned/` |
| C | Évaluation, comparaison, visuels, slides | `results/comparaison.md` et la présentation |

Le travail étant beaucoup plus court sans collecte ni annotation, la marge
dégagée doit aller à **l'étape 1**, le choix du jeu, et à **l'étape 7**, la mise
à jour de la présentation. Ce sont les deux étapes où la qualité se voit.

Prévoyez aussi plusieurs entraînements plutôt qu'un seul : faire varier le
nombre d'epochs ou la taille d'image et rapporter l'effet est un résultat
supplémentaire, obtenu pour le seul coût du temps machine.

---

## Si le temps manque

Par ordre de ce qu'il faut sacrifier en premier :

1. **Réduisez le nombre d'epochs**, de 50 à 25. L'écart de performance sera
   moindre, mais la comparaison restera valide.
2. **Réduisez la taille du jeu d'entraînement.** Un sous-ensemble de 1 000 à
   2 000 images suffit à démontrer la méthode.
3. **Abandonnez les entraînements multiples**, gardez-en un seul.

Ce qu'il ne faut **jamais** sacrifier : le jeu de test et son isolation. Sans
jeu de test séparé, il n'y a pas de résultat, seulement une démonstration.

---

## Récapitulatif des commandes

```bash
python scripts/detect_image.py data/raw/                          # vérifier l'installation
python scripts/import_dataset.py                                  # importer le jeu téléchargé
python scripts/evaluate.py --model models/yolov8n.pt              # mesurer la référence
python scripts/train_yolo.py                                      # entraîner
python scripts/compare_models.py                                  # comparer
```
