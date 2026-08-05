# Guide Streamlit : construire l'application de démonstration du projet

Ce guide couvre Streamlit de zéro jusqu'au déploiement, avec en fil rouge
l'application du projet : permettre à une personne non technique d'envoyer une
photo de trafic et de voir, côte à côte, ce que détecte YOLOv8 pré-entraîné et
ce que détecte notre modèle fine-tuné.

Sommaire :

1. [Qu'est-ce que Streamlit](#1-quest-ce-que-streamlit)
2. [Installation et premier lancement](#2-installation-et-premier-lancement)
3. [Le modèle d'exécution : tout le script est rejoué](#3-le-modèle-dexécution)
4. [Afficher du contenu](#4-afficher-du-contenu)
5. [Les widgets d'entrée](#5-les-widgets-dentrée)
6. [Mise en page : colonnes, onglets, barre latérale](#6-mise-en-page)
7. [Le cache : indispensable pour un modèle ML](#7-le-cache)
8. [session_state : mémoriser entre deux interactions](#8-session_state)
9. [Formulaires et boutons](#9-formulaires-et-boutons)
10. [Application complète du projet (code commenté)](#10-application-complète-du-projet)
11. [Multipage, thème, configuration](#11-multipage-thème-configuration)
12. [Déploiement pour les coéquipiers et le jury](#12-déploiement)
13. [Pièges classiques et bonnes pratiques](#13-pièges-classiques)
14. [Streamlit en général : au-delà du projet](#14-streamlit-en-général)

---

## 1. Qu'est-ce que Streamlit

Streamlit transforme un script Python en application web interactive, sans
écrire de HTML, CSS ou JavaScript. On écrit un script qui s'exécute de haut en
bas ; chaque appel `st.quelque_chose(...)` ajoute un élément à la page.

C'est l'outil standard pour les démonstrations de modèles ML : l'interface
s'écrit en quelques dizaines de lignes et le public n'a besoin que d'un
navigateur.

## 2. Installation et premier lancement

```bash
pip install streamlit
```

Créer un fichier `app/app.py` :

```python
import streamlit as st

st.title("Détection de véhicules (Projet 1)")
st.write("Bonjour !")
```

Lancer :

```bash
streamlit run app/app.py
```

Le navigateur s'ouvre sur `http://localhost:8501`. À chaque sauvegarde du
fichier, la page propose « Rerun » (ou se recharge automatiquement si on
active « Always rerun » dans le menu).

> Pensez à ajouter `streamlit` dans `requirements.txt`.

## 3. Le modèle d'exécution

**Point le plus important de tout le guide.** À chaque interaction de
l'utilisateur (clic, upload, curseur déplacé), Streamlit **réexécute le script
entier de haut en bas**. Il n'y a pas de callbacks à câbler comme dans une GUI
classique : l'état de l'interface est simplement le résultat de la dernière
exécution du script.

Conséquences directes :

- une opération coûteuse (charger YOLOv8, lire un gros fichier) serait refaite
  à chaque clic → il faut la mettre en **cache** (section 7) ;
- une variable Python ordinaire est réinitialisée à chaque interaction → pour
  mémoriser quelque chose, il faut **`st.session_state`** (section 8).

Ces deux mécanismes existent précisément parce que tout est rejoué.

## 4. Afficher du contenu

```python
st.title("Titre principal")          # un seul par page en général
st.header("Section")
st.subheader("Sous-section")
st.markdown("Du **markdown**, des [liens](https://streamlit.io), du `code`.")
st.write(objet)                      # couteau suisse : texte, dict, DataFrame...
st.code("python scripts/train_yolo.py", language="bash")
st.latex(r"mAP = \frac{1}{N}\sum_{i=1}^{N} AP_i")   # oui, du LaTeX !

st.image("data/outputs/embouteillages1_yolov8n.jpg",
         caption="YOLOv8n pré-entraîné", use_container_width=True)

st.dataframe(df)                     # tableau interactif (tri, défilement)
st.table(df)                         # tableau statique, bien pour un petit comparatif
st.metric("mAP@0.5", "0.612", delta="+8.3 %")   # chiffre clé avec écart, parfait
                                                 # pour le tableau comparatif

st.info("Message d'information")
st.success("Réussite")
st.warning("Avertissement")
st.error("Erreur")
st.divider()                         # trait horizontal
```

`st.metric` mérite une mention spéciale pour ce projet : c'est l'affichage
idéal des écarts pré-entraîné / fine-tuné produits par `compare_models.py`.

## 5. Les widgets d'entrée

Chaque widget **renvoie la valeur courante** ; le script étant rejoué à chaque
interaction, la ligne suivante voit toujours la valeur à jour.

```python
# Celui dont on a le plus besoin : l'upload de fichier
fichier = st.file_uploader("Déposez une photo de trafic",
                           type=["jpg", "jpeg", "png"])
if fichier is not None:
    from PIL import Image
    image = Image.open(fichier)

# Curseur, parfait pour le seuil de confiance
conf = st.slider("Seuil de confiance", min_value=0.05, max_value=0.9,
                 value=0.25, step=0.05)

# Autres widgets courants
choix = st.selectbox("Modèle", ["Pré-entraîné", "Fine-tuné", "Les deux"])
cocher = st.checkbox("Afficher les scores", value=True)
mode = st.radio("Mode", ["Image", "Dossier"], horizontal=True)
n = st.number_input("Nombre d'images", min_value=1, value=10)
texte = st.text_input("Titre de la capture")
bouton = st.button("Lancer la détection")   # True uniquement lors du clic
photo = st.camera_input("Ou prenez une photo")  # webcam, très démonstratif
```

`st.camera_input` est un excellent effet de démonstration devant un jury : la
personne prend une photo et voit les détections immédiatement.

## 6. Mise en page

```python
# Barre latérale : y regrouper les réglages pour garder la page pour les résultats
with st.sidebar:
    st.header("Réglages")
    conf = st.slider("Seuil de confiance", 0.05, 0.9, 0.25)

# Colonnes : l'outil du "côte à côte" pré-entraîné / fine-tuné
col_gauche, col_droite = st.columns(2)
with col_gauche:
    st.subheader("YOLOv8 pré-entraîné")
    st.image(image_base)
with col_droite:
    st.subheader("YOLOv8 fine-tuné")
    st.image(image_finetune)

# Onglets : séparer démo / métriques / explications
onglet_demo, onglet_metriques, onglet_apropos = st.tabs(
    ["Démonstration", "Résultats chiffrés", "À propos"])
with onglet_metriques:
    st.table(df_comparaison)

# Bloc repliable : cacher les détails techniques aux non-techniciens
with st.expander("Détails techniques"):
    st.json(rapport)

# Conteneur vide à remplir plus tard (utile avec une barre de progression)
zone = st.empty()
zone.image(resultat)
```

Par défaut la page est étroite et centrée. Pour une mise en page large
(deux images côte à côte, c'est mieux) :

```python
st.set_page_config(page_title="Détection véhicules Bénin", layout="wide")
```

`st.set_page_config` doit être **le premier appel Streamlit** du script.

## 7. Le cache

Sans cache, YOLOv8 serait rechargé du disque à chaque clic. Deux décorateurs :

```python
# Pour les RESSOURCES (modèles, connexions) : un seul objet partagé,
# jamais recopié. C'est celui qu'il faut pour YOLO.
@st.cache_resource
def charger_modele(chemin: str):
    from ultralytics import YOLO
    return YOLO(chemin)

# Pour les DONNÉES (DataFrame, calculs) : le résultat est sérialisé et
# recopié à chaque appel, donc protégé des mutations.
@st.cache_data
def charger_comparaison(chemin: str):
    import polars as pl
    return pl.read_csv(chemin)
```

Fonctionnement : à l'appel, Streamlit calcule une clé à partir des arguments ;
si la fonction a déjà été appelée avec ces arguments, le résultat mémorisé est
renvoyé sans exécuter la fonction. Le premier chargement prend quelques
secondes, tous les suivants sont instantanés.

Règles :

- `cache_resource` pour tout objet lourd et non sérialisable : modèle,
  connexion, GPU ;
- `cache_data` pour tout ce qui ressemble à des données ;
- les arguments doivent être hachables : passer le **chemin** du modèle
  (chaîne), pas l'objet modèle ;
- `st.cache_data.clear()` et le menu « Clear cache » (touche `C`) vident le
  cache pendant le développement.

## 8. session_state

`st.session_state` est un dictionnaire qui **survit aux réexécutions** du
script (mais pas au rechargement de l'onglet du navigateur). C'est la mémoire
de la session utilisateur.

```python
# Initialisation défensive : ne créer la clé que si elle n'existe pas
if "historique" not in st.session_state:
    st.session_state.historique = []

if st.button("Lancer la détection"):
    resultat = detecter(image)
    st.session_state.historique.append(resultat)

st.write(f"{len(st.session_state.historique)} détection(s) cette session")
```

Chaque widget peut aussi écrire directement dedans via `key=` :

```python
st.slider("Seuil", 0.05, 0.9, 0.25, key="conf")
# ... la valeur est disponible partout via st.session_state.conf
```

Cas d'usage typique ici : conserver la dernière image analysée et ses
résultats pour que le déplacement du curseur de confiance ne perde pas tout.

## 9. Formulaires et boutons

Un `st.button` vaut `True` uniquement pendant l'exécution qui suit son clic ;
au clic suivant ailleurs, il redevient `False`. Pour un enchaînement « je règle
tout puis je valide », le formulaire évite de relancer la détection à chaque
réglage :

```python
with st.form("parametres"):
    conf = st.slider("Seuil de confiance", 0.05, 0.9, 0.25)
    modele_choisi = st.selectbox("Modèle", ["Les deux", "Pré-entraîné", "Fine-tuné"])
    lancer = st.form_submit_button("Détecter")

if lancer:
    ...  # exécuté seulement à la validation du formulaire
```

Pour les opérations longues, informer l'utilisateur :

```python
with st.spinner("Détection en cours..."):
    resultat = modele.predict(...)

barre = st.progress(0, text="Traitement du dossier")
for i, image in enumerate(images):
    ...
    barre.progress((i + 1) / len(images))

st.toast("Détection terminée !")   # notification discrète
```

Et pour laisser repartir avec le résultat :

```python
st.download_button("Télécharger l'image annotée",
                   data=octets_jpg, file_name="detection.jpg", mime="image/jpeg")
```

## 10. Application complète du projet

Fichier proposé : `app/app.py`. Ce code est fonctionnel tel quel dès que
`models/finetuned/yolov8n_benin.pt` existe ; en attendant, il fonctionne avec
le seul modèle pré-entraîné.

```python
"""Application de démonstration : YOLOv8 pré-entraîné vs fine-tuné.

Lancement :  streamlit run app/app.py
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

RACINE = Path(__file__).resolve().parent.parent
MODELE_BASE = RACINE / "models" / "yolov8n.pt"
MODELE_FINETUNE = RACINE / "models" / "finetuned" / "yolov8n_benin.pt"
COCO_VEHICULES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

st.set_page_config(page_title="Détection de véhicules Bénin", layout="wide")


@st.cache_resource
def charger_modele(chemin: str):
    from ultralytics import YOLO
    return YOLO(chemin)


def detecter(chemin_modele: Path, image: Image.Image, conf: float):
    """Renvoie (image annotée RGB, nombre de véhicules détectés)."""
    modele = charger_modele(str(chemin_modele))
    est_coco = len(modele.names) >= 80
    resultat = modele.predict(
        np.array(image),
        conf=conf,
        classes=COCO_VEHICULES if est_coco else None,
        verbose=False,
    )[0]
    annotee = resultat.plot()[:, :, ::-1]  # BGR -> RGB
    return annotee, len(resultat.boxes)


st.title("Détection de véhicules dans le trafic béninois")
st.markdown(
    "Envoyez une photo d'embouteillage : l'application montre côte à côte les "
    "détections du modèle **YOLOv8 générique** et celles de notre modèle "
    "**adapté au trafic urbain dense**."
)

with st.sidebar:
    st.header("Réglages")
    conf = st.slider(
        "Seuil de confiance", 0.05, 0.90, 0.25, 0.05,
        help="Plus le seuil est bas, plus le modèle affiche de détections, "
             "y compris incertaines.",
    )
    st.caption(
        "AMA Cohorte 2, Projet 1, Groupe 5 : "
        "Aïchatou TRAORE, Benoît DJOSSOU, Andréa AFOUDA"
    )

onglet_demo, onglet_chiffres, onglet_apropos = st.tabs(
    ["Démonstration", "Résultats chiffrés", "À propos"])

with onglet_demo:
    fichier = st.file_uploader("Photo de trafic", type=["jpg", "jpeg", "png"])
    photo = st.camera_input("… ou prenez une photo")
    source = fichier or photo

    if source is None:
        st.info("Déposez une image pour lancer la détection.")
    else:
        image = Image.open(source).convert("RGB")
        finetune_disponible = MODELE_FINETUNE.exists()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("YOLOv8 pré-entraîné (COCO)")
            with st.spinner("Détection..."):
                annotee, n = detecter(MODELE_BASE, image, conf)
            st.image(annotee, use_container_width=True)
            st.metric("Véhicules détectés", n)
        with col2:
            st.subheader("YOLOv8 fine-tuné (trafic dense)")
            if finetune_disponible:
                with st.spinner("Détection..."):
                    annotee_ft, n_ft = detecter(MODELE_FINETUNE, image, conf)
                st.image(annotee_ft, use_container_width=True)
                st.metric("Véhicules détectés", n_ft, delta=n_ft - n)
            else:
                st.warning("Modèle fine-tuné introuvable : lancez d'abord "
                           "`python scripts/train_yolo.py`.")

with onglet_chiffres:
    comparaison = RACINE / "results" / "comparaison.md"
    if comparaison.exists():
        st.markdown(comparaison.read_text(encoding="utf-8"))
    else:
        st.info("Le tableau comparatif n'existe pas encore : lancez "
                "`python scripts/compare_models.py`.")

with onglet_apropos:
    st.markdown(
        """
        **Problématique.** Les modèles YOLO perdent en précision sur les
        véhicules éloignés, petits ou masqués dans les embouteillages :
        exactement les conditions du trafic urbain béninois.

        **Approche.** Fine-tuning de YOLOv8 sur un jeu de trafic urbain dense,
        puis mesure de l'écart avec le modèle d'origine sur le même jeu de
        test, ventilée par taille apparente des véhicules.
        """
    )
```

Points de conception à retenir :

- le modèle est chargé **une seule fois** grâce à `cache_resource` ;
- l'application **dégrade proprement** si le modèle fine-tuné ou le tableau
  comparatif n'existent pas encore : chacun peut la lancer dès maintenant ;
- l'onglet « Résultats chiffrés » réutilise directement
  `results/comparaison.md` : aucun chiffre recopié à la main, l'application
  reste juste quand les résultats changent.

## 11. Multipage, thème, configuration

**Plusieurs pages.** Créer un dossier `pages/` à côté de `app.py` : chaque
fichier `pages/1_Nom.py` devient une page dans la barre latérale, préfixée par
son numéro pour l'ordre. Utile si vous voulez une page « Démo », une page
« Méthodologie », une page « Équipe ».

**Thème.** Fichier `.streamlit/config.toml` à la racine du projet :

```toml
[theme]
primaryColor = "#E63946"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F1FAEE"
textColor = "#1D3557"

[server]
maxUploadSize = 20   # Mo, limite du file_uploader
```

**Secrets.** Jamais de mot de passe ou de jeton dans le code : les mettre dans
`.streamlit/secrets.toml` (à ajouter au `.gitignore`), lus via
`st.secrets["cle"]`. Sur Streamlit Community Cloud, ils se saisissent dans
l'interface web.

## 12. Déploiement

Trois options, de la plus simple à la plus robuste :

**a) Réseau local (démo en salle).** `streamlit run app/app.py` affiche aussi
une « Network URL » : toute personne sur le même Wi-Fi peut ouvrir cette
adresse. Zéro configuration, suffisant pour une soutenance.

**b) Streamlit Community Cloud (recommandé).** Gratuit, se branche sur le
dépôt GitHub (déjà en place pour ce projet) :

1. le dépôt doit contenir `app/app.py` et un `requirements.txt` incluant
   `streamlit` et `ultralytics` ;
2. sur [share.streamlit.io](https://share.streamlit.io), se connecter avec
   GitHub, « New app », choisir dépôt / branche / fichier ;
3. chaque `git push` redéploie automatiquement.

Contraintes à anticiper : machine modeste (~1 Go de RAM, pas de GPU ; YOLOv8n
en CPU y répond en 1 à 2 s par image, acceptable pour une démo) et surtout
**les poids fine-tunés doivent être accessibles**. Or `models/*.pt` est dans
le `.gitignore`. Solutions : héberger les poids sur Hugging Face Hub ou une
release GitHub et les télécharger au démarrage dans une fonction
`@st.cache_resource`, ou lever l'exclusion pour ce seul fichier
(`!models/finetuned/yolov8n_benin.pt` dans `.gitignore`, quelques Mo pour
yolov8n, acceptable).

**c) Hugging Face Spaces.** Créer un Space de type Streamlit et y pousser le
code ; même logique, bien intégré si les poids sont déjà sur le Hub. Offre
aussi du matériel plus puissant (payant).

## 13. Pièges classiques

| Piège | Symptôme | Remède |
|---|---|---|
| Modèle chargé hors cache | Chaque clic prend plusieurs secondes | `@st.cache_resource` |
| Variable « perdue » | Une liste se vide à chaque interaction | `st.session_state` |
| `set_page_config` tardif | `StreamlitAPIException` | Le mettre en tout premier appel |
| Chemins relatifs au terminal | Fichiers introuvables selon le dossier de lancement | Construire les chemins depuis `Path(__file__)` comme dans les scripts du projet |
| Image OpenCV bleutée | Couleurs fausses | OpenCV est en BGR : inverser les canaux (`[:, :, ::-1]`) avant `st.image` |
| Gros upload refusé | Erreur à 200 Mo+ | `maxUploadSize` dans `config.toml` |
| Détection relancée au moindre réglage | Lenteur, GPU sollicité en boucle | `st.form`, ou mémoriser le résultat dans `session_state` |
| Deux widgets identiques | `DuplicateWidgetID` | Donner un `key=` unique à chacun |

Bonnes pratiques finales :

- garder l'application **mince** : elle appelle le code des `scripts/`, elle ne
  réimplémente rien (une seule vérité pour la logique de détection) ;
- penser au public non technique : textes en français, `help=` sur les
  réglages, aucune trace de code ou de chemin dans l'interface ;
- prévoir 2 ou 3 **images d'exemple embarquées** (boutons « Essayer avec cette
  image ») pour que la démo fonctionne même sans photo sous la main ;
- flouter plaques et visages sur toute capture d'écran mise dans le rapport ou
  les slides (voir README, section 11).

## 14. Streamlit en général

Tout ce qui précède se transpose à n'importe quel autre sujet. Cette section
est un cours d'usage général : les grandes familles d'applications que l'on
construit avec Streamlit, pour vous resservir de l'outil bien après ce projet.

### 14.1 Applications de données : tableaux et graphiques

Streamlit est d'abord utilisé pour explorer et présenter des données. Les
graphiques natifs se contentent d'un DataFrame (pandas ou polars) :

```python
import pandas as pd
import streamlit as st

df = pd.read_csv("ventes.csv")

st.line_chart(df, x="mois", y="chiffre_affaires")   # courbe
st.bar_chart(df, x="produit", y="quantite")         # barres
st.area_chart(df, x="mois", y=["nord", "sud"])      # aires empilées
st.scatter_chart(df, x="prix", y="ventes")          # nuage de points
st.map(df)                                          # points sur carte
                                                    # (colonnes lat / lon)
```

Pour un contrôle fin (titres, échelles, annotations), Streamlit affiche
directement les figures des bibliothèques classiques :

```python
# matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(df["prix"], bins=30)
ax.set_title("Distribution des prix")
st.pyplot(fig)

# plotly : graphiques interactifs (zoom, survol) sans effort
import plotly.express as px
st.plotly_chart(px.scatter(df, x="prix", y="ventes", color="region"))

# altair fonctionne de la même façon avec st.altair_chart
```

Le tableau éditable transforme une application en petit outil de saisie :

```python
df_modifie = st.data_editor(df, num_rows="dynamic")  # l'utilisateur peut
                                                     # éditer et ajouter des lignes
if st.button("Enregistrer"):
    df_modifie.to_csv("ventes.csv", index=False)
```

Enchaînement typique d'un tableau de bord : charger les données dans une
fonction `@st.cache_data`, proposer des filtres dans la barre latérale
(`st.multiselect`, `st.date_input`), afficher trois ou quatre `st.metric`
en haut via `st.columns`, puis les graphiques.

### 14.2 Se connecter à des données externes

`st.connection` gère la connexion, sa réutilisation et le cache des requêtes :

```python
# .streamlit/secrets.toml :
# [connections.ma_base]
# url = "postgresql://utilisateur:motdepasse@hote:5432/base"

conn = st.connection("ma_base", type="sql")
df = conn.query("SELECT * FROM commandes WHERE annee = 2026", ttl=600)
st.dataframe(df)
```

`ttl=600` signifie que le résultat est mémorisé dix minutes : la base n'est
pas interrogée à chaque interaction. Pour une API web, le duo
`requests` + `@st.cache_data(ttl=...)` suit la même logique :

```python
@st.cache_data(ttl=3600)
def meteo(ville: str) -> dict:
    import requests
    return requests.get(f"https://api.exemple.com/meteo/{ville}", timeout=10).json()
```

Rappel : les identifiants vont dans `secrets.toml` (exclu du git), jamais
dans le code.

### 14.3 Chatbots et applications d'IA conversationnelle

Streamlit fournit des composants de discussion prêts à l'emploi, devenus le
standard des démonstrations de modèles de langage :

```python
import streamlit as st

if "messages" not in st.session_state:
    st.session_state.messages = []

# Rejouer l'historique (le script est réexécuté à chaque message)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Posez votre question"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        reponse = generer_reponse(question)      # votre modèle ou une API
        st.markdown(reponse)
    st.session_state.messages.append({"role": "assistant", "content": reponse})
```

On retrouve les trois piliers du guide : `session_state` porte l'historique,
le script rejoué réaffiche la conversation, et le modèle serait chargé via
`@st.cache_resource`. Si la source produit la réponse au fil de l'eau
(streaming), `st.write_stream(generateur)` l'affiche mot à mot.

### 14.4 Contrôle d'exécution avancé

Trois outils à connaître quand une application grossit :

```python
# st.fragment : cette fonction se réexécute SEULE quand on interagit avec
# ses widgets, sans rejouer tout le script. Précieux quand le reste de la
# page est coûteux.
@st.fragment
def zone_reglages():
    st.slider("Seuil", 0.0, 1.0, key="seuil")
    st.selectbox("Vue", ["carte", "tableau"], key="vue")

# st.fragment(run_every="10s") : la fonction se relance toute seule,
# pour un tableau de bord qui se rafraîchit en continu.

# st.rerun() : relancer immédiatement le script (après une action qui
# change l'état, par exemple une connexion réussie).

# Callbacks : exécuter une fonction au moment précis du changement,
# avant la réexécution du script.
def au_changement():
    st.session_state.page = 1   # revenir en page 1 quand le filtre change

st.selectbox("Filtre", options, key="filtre", on_change=au_changement)
```

Et pour structurer une application qui devient grosse : découper en modules
Python ordinaires (la logique métier dans des fonctions importées, testables
sans Streamlit), garder les fichiers de pages minces, et centraliser les clés
de `session_state` dans un seul endroit documenté.

### 14.5 Parcours d'apprentissage

Pour progresser après ce projet, dans l'ordre :

1. [Documentation officielle](https://docs.streamlit.io) : les pages
   *Get started* puis *Concepts* reprennent ce guide en plus détaillé.
2. [30 Days of Streamlit](https://30days.streamlit.app) : un exercice guidé
   par jour, du premier `st.write` à l'application déployée.
3. [La galerie](https://streamlit.io/gallery) : des centaines d'applications
   avec leur code source, la meilleure façon de voler de bonnes idées.
4. [Les composants communautaires](https://streamlit.io/components) :
   AgGrid (tableaux avancés), Folium (cartes), drawable-canvas (dessin sur
   image, utile en vision par ordinateur), etc.

Idées d'applications pour s'exercer : un explorateur de fichiers CSV
(upload, filtres, statistiques, export), un tableau de bord de suivi de
dépenses personnelles, une interface pour n'importe quel modèle entraîné
pendant la formation, un formulaire d'annotation d'images qui écrit des
labels YOLO.

---

Documentation officielle : <https://docs.streamlit.io>. La référence des API
(`st.image`, `st.file_uploader`, etc.) y est claire et pleine d'exemples.
