# app.py : application de démonstration
# YOLOv8 pré-entraîné (COCO) contre YOLOv8 fine-tuné (Benin).
#
# Compare les deux modèles du projet sur les 4 classes de véhicules
# (voiture, moto, bus, camion), sur image ou sur vidéo, avec une
# ventilation des détections par taille apparente d'objet, qui est la
# question centrale du sujet (véhicules éloignés et de petite taille).
#
# Design « chaleur et encre » : fond crème, bleu nuit en couleur d'encre,
# terre cuite en accent. Le récit visuel est un avant / après fine-tuning.

import io
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import imageio.v2 as imageio
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

# ============================================
# CONSTANTES
# ============================================
RACINE = Path(__file__).resolve().parent
POIDS_STANDARD = RACINE / "models" / "yolov8n.pt"
POIDS_BENIN = RACINE / "models" / "finetuned" / "yolov8n_benin.pt"
DOSSIER_EXEMPLES = RACINE / "assets" / "exemples"

# Le modèle pré-entraîné raisonne sur les 80 classes COCO : on ne retient
# que les 4 classes de véhicules du projet (voir README, section 8).
CLASSES_COCO_VEHICULES = [2, 3, 5, 7]
NOMS_CLASSES = {
    "standard": {2: "voiture", 3: "moto", 5: "bus", 7: "camion"},
    "benin": {0: "voiture", 1: "moto", 2: "bus", 3: "camion"},
}
ORDRE_CLASSES = ["voiture", "moto", "bus", "camion"]
ORDRE_TAILLES = ["petit", "moyen", "grand"]

# Les images téléversées peuvent avoir n'importe quelle dimension : le
# réseau travaille de toute façon en 640 px. Au-delà de cette borne, on
# réduit l'image avant traitement pour épargner la mémoire du serveur
# (1 Go sur Streamlit Community Cloud) sans rien changer aux détections.
COTE_MAX_IMAGE = 2560

# Seuils COCO de taille d'objet, appliqués après remise à l'échelle de
# l'image vers la résolution d'entrée du réseau (voir README, section 9 :
# sans cette normalisation, tout objet d'une photo haute résolution est
# classé « grand » et l'analyse ne mesure plus rien).
RESOLUTION_RESEAU = 640
SEUIL_PETIT = 32 * 32
SEUIL_MOYEN = 96 * 96

# Récit avant / après : le modèle généraliste est le point de départ, le
# modèle fine-tuné est le résultat.
ETIQUETTES = {"standard": "Avant", "benin": "Après"}
DESCRIPTIONS = {
    "standard": "YOLOv8 généraliste, entraîné sur COCO",
    "benin": "le même YOLOv8, fine-tuné sur le trafic urbain dense",
}

# Palette partagée avec .streamlit/config.toml
ENCRE = "#1A1A2E"
ENCRE_DOUCE = "#4A4A63"
CREME = "#FBF7F0"
SABLE = "#F3ECDF"
LIGNE = "#E3D8C3"
TERRE = "#C1502E"
OCRE = "#8A7355"


# ============================================
# CHARGEMENT DES MODÈLES
# ============================================
@st.cache_resource(show_spinner="Chargement des modèles…")
def charger_modeles():
    """Charge les deux modèles une seule fois pour toute la session."""
    modeles, erreurs = {}, {}

    try:
        # Si le fichier local manque, Ultralytics télécharge yolov8n.pt tout seul.
        source = str(POIDS_STANDARD) if POIDS_STANDARD.exists() else "yolov8n.pt"
        modeles["standard"] = YOLO(source)
    except Exception as e:
        modeles["standard"] = None
        erreurs["standard"] = str(e)

    try:
        if POIDS_BENIN.exists():
            modeles["benin"] = YOLO(str(POIDS_BENIN))
        else:
            modeles["benin"] = None
            erreurs["benin"] = f"Poids fine-tunés introuvables : {POIDS_BENIN}"
    except Exception as e:
        modeles["benin"] = None
        erreurs["benin"] = str(e)

    return modeles, erreurs


# ============================================
# DÉTECTION
# ============================================
def _extraire_detections(resultat, cle_modele):
    """Convertit un résultat Ultralytics en liste de détections annotées
    (classe en français, confiance, boîte, catégorie de taille COCO)."""
    detections = []
    if resultat.boxes is None or len(resultat.boxes) == 0:
        return detections

    h, w = resultat.orig_shape
    echelle = RESOLUTION_RESEAU / max(h, w)
    for boite in resultat.boxes:
        x1, y1, x2, y2 = boite.xyxy[0].tolist()
        aire_reseau = (x2 - x1) * (y2 - y1) * echelle * echelle
        if aire_reseau < SEUIL_PETIT:
            taille = "petit"
        elif aire_reseau < SEUIL_MOYEN:
            taille = "moyen"
        else:
            taille = "grand"
        detections.append({
            "classe": NOMS_CLASSES[cle_modele].get(int(boite.cls), f"classe {int(boite.cls)}"),
            "confiance": float(boite.conf),
            "boite": (int(x1), int(y1), int(x2), int(y2)),
            "taille": taille,
        })
    return detections


@st.cache_data(max_entries=24, show_spinner=False)
def detecter_image(octets_image, conf, iou, cle_modele):
    """Détection sur une image. Mise en cache : re-cliquer un bouton de
    téléchargement ou changer d'onglet ne relance pas l'inférence."""
    modeles, _ = charger_modeles()
    modele = modeles[cle_modele]
    # exif_transpose : les photos de téléphone portent leur rotation en
    # métadonnée, sans quoi elles arrivent couchées.
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(octets_image))).convert("RGB")
    if max(image.size) > COTE_MAX_IMAGE:
        image.thumbnail((COTE_MAX_IMAGE, COTE_MAX_IMAGE))
    classes = CLASSES_COCO_VEHICULES if cle_modele == "standard" else None
    resultat = modele(image, conf=conf, iou=iou, classes=classes, verbose=False)[0]
    annotee = cv2.cvtColor(resultat.plot(), cv2.COLOR_BGR2RGB)
    return annotee, _extraire_detections(resultat, cle_modele)


def traiter_video(chemin_video, conf, iou, pas, rappel_progression=None):
    """Une seule passe sur la vidéo : chaque frame retenue est analysée par
    les deux modèles. Les vidéos annotées sont encodées en H.264 (lisible
    dans un navigateur, contrairement au codec mp4v d'OpenCV).

    Retourne (sorties, frames_traitees) où sorties[cle] contient les octets
    du mp4 annoté et les statistiques de détection du modèle."""
    modeles, _ = charger_modeles()
    actifs = {cle: m for cle, m in modeles.items() if m is not None}

    cap = cv2.VideoCapture(str(chemin_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # On saute (pas - 1) frames sur pas : la vidéo de sortie garde la durée
    # d'origine en abaissant son débit d'images d'autant.
    fps_sortie = max(1.0, fps / pas)

    stats = {cle: {"detections": 0, "par_classe": Counter(), "par_taille": Counter()}
             for cle in actifs}
    sorties = {}

    with tempfile.TemporaryDirectory() as tmp:
        redacteurs = {
            cle: imageio.get_writer(
                str(Path(tmp) / f"{cle}.mp4"),
                fps=fps_sortie, codec="libx264",
                pixelformat="yuv420p", macro_block_size=1,
            )
            for cle in actifs
        }
        indice, traitees = 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if indice % pas == 0:
                    # yuv420p exige des dimensions paires
                    frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
                    for cle, modele in actifs.items():
                        classes = CLASSES_COCO_VEHICULES if cle == "standard" else None
                        resultat = modele(frame, conf=conf, iou=iou,
                                          classes=classes, verbose=False)[0]
                        for det in _extraire_detections(resultat, cle):
                            stats[cle]["detections"] += 1
                            stats[cle]["par_classe"][det["classe"]] += 1
                            stats[cle]["par_taille"][det["taille"]] += 1
                        annotee = cv2.cvtColor(resultat.plot(), cv2.COLOR_BGR2RGB)
                        redacteurs[cle].append_data(annotee)
                    traitees += 1
                    if rappel_progression:
                        rappel_progression(indice + 1, total)
                indice += 1
        finally:
            cap.release()
            for redacteur in redacteurs.values():
                redacteur.close()

        for cle in actifs:
            sorties[cle] = {"octets": (Path(tmp) / f"{cle}.mp4").read_bytes(), **stats[cle]}

    return sorties, traitees


# ============================================
# AFFICHAGE
# ============================================
def compter(detections):
    par_classe = Counter(d["classe"] for d in detections)
    par_taille = Counter(d["taille"] for d in detections)
    return par_classe, par_taille


def entete_modele(cle):
    """Chip AVANT / APRÈS suivi de la description du modèle."""
    chip_css = "chip-avant" if cle == "standard" else "chip-apres"
    return (
        f'<div class="entete-modele">'
        f'<span class="chip {chip_css}">{ETIQUETTES[cle].upper()}</span>'
        f'<span class="entete-desc">{DESCRIPTIONS[cle]}</span>'
        f'</div>'
    )


def afficher_resultat_image(cle, annotee, detections):
    st.markdown(entete_modele(cle), unsafe_allow_html=True)
    st.image(annotee, width="stretch")

    par_classe, _ = compter(detections)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Véhicules détectés", len(detections))
    with col_b:
        if detections:
            conf_moyenne = sum(d["confiance"] for d in detections) / len(detections)
            st.metric("Confiance moyenne", f"{conf_moyenne:.1%}")
        else:
            st.metric("Confiance moyenne", "n/a")

    if detections:
        st.markdown(
            '<p class="ligne-classes">'
            + " · ".join(f"{par_classe[c]} {c}(s)" for c in ORDRE_CLASSES if par_classe[c])
            + "</p>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Détail des {len(detections)} détections"):
            st.dataframe(
                [{
                    "N°": i,
                    "Classe": d["classe"],
                    "Confiance": f"{d['confiance']:.1%}",
                    "Taille": d["taille"],
                    "Position": "({}, {}) → ({}, {})".format(*d["boite"]),
                } for i, d in enumerate(detections, 1)],
                width="stretch", hide_index=True,
            )


def afficher_comparaison(stats_par_modele):
    """Le verdict du récit avant / après, puis deux tableaux discrets.

    stats_par_modele[cle] = (par_classe, par_taille), Counters."""
    cles = list(stats_par_modele)
    st.markdown('<h3 class="titre-section">Ce que change le fine-tuning</h3>',
                unsafe_allow_html=True)

    if len(cles) == 2:
        ecart = (sum(stats_par_modele["benin"][0].values())
                 - sum(stats_par_modele["standard"][0].values()))
        petits = (stats_par_modele["benin"][1].get("petit", 0)
                  - stats_par_modele["standard"][1].get("petit", 0))
        signe = "+" if ecart >= 0 else ""
        signe_p = "+" if petits >= 0 else ""
        st.markdown(
            f'<div class="verdict">'
            f'<div class="verdict-bloc"><div class="verdict-nombre">{signe}{ecart}</div>'
            f'<div class="verdict-legende">détections après fine-tuning</div></div>'
            f'<div class="verdict-sep"></div>'
            f'<div class="verdict-bloc"><div class="verdict-nombre">{signe_p}{petits}</div>'
            f'<div class="verdict-legende">sur les objets petits, cœur du sujet</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="titre-tableau">Par classe</p>', unsafe_allow_html=True)
        st.dataframe(
            [{"Classe": c, **{ETIQUETTES[k]: stats_par_modele[k][0].get(c, 0) for k in cles}}
             for c in ORDRE_CLASSES]
            + [{"Classe": "Total",
                **{ETIQUETTES[k]: sum(stats_par_modele[k][0].values()) for k in cles}}],
            width="stretch", hide_index=True,
        )
    with col2:
        st.markdown('<p class="titre-tableau">Par taille apparente '
                    '<span class="note-tableau">(seuils COCO, normalisés à la '
                    'résolution du réseau)</span></p>', unsafe_allow_html=True)
        st.dataframe(
            [{"Taille": t, **{ETIQUETTES[k]: stats_par_modele[k][1].get(t, 0) for k in cles}}
             for t in ORDRE_TAILLES],
            width="stretch", hide_index=True,
        )

    st.markdown(
        '<p class="avertissement">Un nombre de détections plus élevé n\'est pas en soi '
        'une preuve d\'amélioration : il peut contenir des faux positifs. La mesure de '
        'référence reste le tableau <code>results/comparaison.md</code>, calculé sur le '
        'jeu de test annoté. La ligne à regarder ici : les objets <strong>petits</strong>.</p>',
        unsafe_allow_html=True,
    )


def bouton_telechargement_image(cle, annotee):
    image_pil = Image.fromarray(annotee)
    tampon = io.BytesIO()
    image_pil.save(tampon, format="JPEG", quality=92)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label=f"Télécharger la version {ETIQUETTES[cle]}",
        data=tampon.getvalue(),
        file_name=f"{cle}_vehicules_{horodatage}.jpg",
        mime="image/jpeg",
        width="stretch",
        key=f"dl_image_{cle}",
    )


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,600&family=Inter:wght@400;500;600&display=swap');

/* ---- Base typographique ---- */
html, body, [data-testid="stAppViewContainer"] * {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
h1, h2, h3, .titre-section, .verdict-nombre, [data-testid="stMetricValue"] {
    font-family: 'Fraunces', Georgia, serif !important;
}
/* Les icônes de Streamlit sont des ligatures Material Symbols : sans cette
   exception, la règle globale ci-dessus les affiche en toutes lettres
   (« keyboard_arrow_right » au lieu d'une flèche). */
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* ---- Barre latérale : identité de l'application ---- */
[data-testid="stSidebar"] {
    border-right: 1px solid __LIGNE__;
}
.marque { padding: 0.4rem 0 0.9rem 0; }
.marque-nom {
    font-family: 'Fraunces', Georgia, serif; font-weight: 700;
    font-size: 1.3rem; color: __ENCRE__; line-height: 1.15;
    margin-bottom: 0.25rem;
}
.marque-sous {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: __OCRE__;
}
.section-laterale {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: __OCRE__;
    margin: 1.3rem 0 0.3rem 0;
}
.etat-modele { font-size: 0.88rem; color: __ENCRE_DOUCE__; margin: 0.15rem 0; }
.point-ok { color: #2E7D4F; }
.point-ko { color: #B3261E; }
.credit-lateral {
    font-size: 0.75rem; color: __OCRE__; line-height: 1.6;
    border-top: 1px solid __LIGNE__; padding-top: 0.9rem; margin-top: 1.6rem;
}

/* ---- Barre d'application ---- */
.barre-app {
    display: flex; justify-content: space-between; align-items: baseline;
    flex-wrap: wrap; gap: 0.4rem;
    padding: 0.9rem 0 0.9rem 0; border-bottom: 1px solid __LIGNE__;
    margin-bottom: 0.6rem;
}
.barre-titre {
    font-family: 'Fraunces', Georgia, serif; font-weight: 600;
    font-size: 1.75rem; color: __ENCRE__; margin: 0; line-height: 1.2;
}
.barre-meta { font-size: 0.82rem; color: __OCRE__; }
.barre-intro {
    font-size: 0.95rem; line-height: 1.6; color: __ENCRE_DOUCE__;
    max-width: 52rem; margin: 0.4rem 0 0.8rem 0;
}

/* ---- Chips avant / après ---- */
.entete-modele { display: flex; align-items: baseline; gap: 0.6rem; margin: 0.4rem 0 0.7rem 0; }
.chip {
    display: inline-block; padding: 0.28rem 0.95rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
}
.chip-avant { background: __ENCRE__; color: __CREME__; }
.chip-apres { background: __TERRE__; color: __CREME__; }
.entete-desc { font-size: 0.9rem; color: __ENCRE_DOUCE__; }

/* ---- Images annotées ---- */
[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(26, 26, 46, 0.10);
    animation: apparition 0.5s ease both;
}
@keyframes apparition {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: none; }
}

/* ---- Métriques ---- */
[data-testid="stMetric"] {
    background: __SABLE__; border: 1px solid __LIGNE__;
    border-radius: 12px; padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] { color: __ENCRE__; }
[data-testid="stMetricLabel"] { color: __ENCRE_DOUCE__; }

/* ---- Verdict avant / après ---- */
.verdict {
    display: flex; align-items: center; gap: 2rem;
    background: __ENCRE__; color: __CREME__;
    border-radius: 16px; padding: 1.5rem 2rem; margin: 0.6rem 0 1.4rem 0;
}
.verdict-nombre { font-size: 2.6rem; font-weight: 700; line-height: 1; color: __CREME__; }
.verdict-legende { font-size: 0.85rem; color: #B9B4C7; margin-top: 0.4rem; }
.verdict-sep { width: 1px; height: 3.2rem; background: rgba(251, 247, 240, 0.25); }

/* ---- Sections et tableaux ---- */
.titre-section {
    font-size: 1.6rem; font-weight: 600; color: __ENCRE__;
    margin: 0.4rem 0 0.9rem 0;
}
.titre-tableau { font-weight: 600; font-size: 0.95rem; color: __ENCRE__; margin-bottom: 0.3rem; }
.note-tableau { font-weight: 400; font-size: 0.8rem; color: __OCRE__; }
.ligne-classes { font-size: 0.88rem; color: __OCRE__; margin: 0.1rem 0 0.5rem 0; }
.avertissement {
    font-size: 0.84rem; line-height: 1.55; color: __ENCRE_DOUCE__;
    border-left: 3px solid __TERRE__; padding-left: 0.9rem; margin-top: 0.9rem;
}

/* ---- Onglets ---- */
[data-testid="stTabs"] button {
    font-weight: 600; font-size: 1rem; color: __ENCRE_DOUCE__;
    transition: color 0.2s ease;
}
[data-testid="stTabs"] button[aria-selected="true"] { color: __TERRE__; }

/* ---- Zone de dépôt et boutons ---- */
[data-testid="stFileUploaderDropzone"] {
    background: __SABLE__; border: 1px dashed #C9B892; border-radius: 14px;
}
.stButton button, .stDownloadButton button {
    border-radius: 10px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(193, 80, 46, 0.22);
}

</style>
"""


def injecter_style():
    css = STYLE
    for cle, valeur in {
        "__ENCRE__": ENCRE, "__ENCRE_DOUCE__": ENCRE_DOUCE, "__CREME__": CREME,
        "__SABLE__": SABLE, "__LIGNE__": LIGNE, "__TERRE__": TERRE, "__OCRE__": OCRE,
    }.items():
        css = css.replace(cle, valeur)
    st.markdown(css, unsafe_allow_html=True)


# ============================================
# APPLICATION
# ============================================
def main():
    st.set_page_config(
        page_title="Détection de véhicules · Bénin",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    injecter_style()

    modeles, erreurs = charger_modeles()
    cles_actives = [cle for cle in ("standard", "benin") if modeles[cle] is not None]

    # ----- Barre latérale : identité, réglages, état -----
    with st.sidebar:
        st.markdown("""
            <div class="marque">
                <div>
                    <div class="marque-nom">Détection de véhicules</div>
                    <div class="marque-sous">Trafic urbain dense · Bénin</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-laterale">Réglages</p>', unsafe_allow_html=True)
        seuil_confiance = st.slider(
            "Seuil de confiance", 0.0, 1.0, 0.25, 0.05,
            help="En dessous de ce score, une détection est ignorée. "
                 "Baisser le seuil révèle plus de véhicules… et plus de faux positifs.",
        )
        seuil_iou = st.slider(
            "Seuil IoU (suppression des doublons)", 0.0, 1.0, 0.45, 0.05,
            help="Deux boîtes qui se recouvrent au-delà de ce seuil sont "
                 "considérées comme le même véhicule.",
        )

        st.markdown('<p class="section-laterale">Modèles</p>', unsafe_allow_html=True)
        for cle in ("standard", "benin"):
            if modeles[cle] is not None:
                st.markdown(
                    f'<p class="etat-modele"><span class="point-ok">●</span> '
                    f'{ETIQUETTES[cle]} : {DESCRIPTIONS[cle]}</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<p class="etat-modele"><span class="point-ko">●</span> '
                    f'{ETIQUETTES[cle]} : indisponible</p>',
                    unsafe_allow_html=True,
                )

        with st.expander("À propos"):
            st.markdown("""
            Un détecteur généraliste manque les motos et les véhicules
            éloignés du trafic urbain dense. Cette application compare
            YOLOv8 avant et après son fine-tuning sur des scènes de
            trafic comparables au contexte béninois, sur les 4 classes
            du projet : voiture, moto, bus, camion.
            """)

        st.markdown("""
            <div class="credit-lateral">
                Aïchatou Traore · Benoît Djossou · Andréa Afouda<br>
                AMA Cohorte 2, Projet Intégrateur 1, Groupe 5<br>
                Images d'exemple : jeu BMD-45 (CC BY 4.0)
            </div>
        """, unsafe_allow_html=True)

    # ----- Barre d'application -----
    st.markdown(f"""
        <div class="barre-app">
            <h1 class="barre-titre">Analyse comparative</h1>
            <span class="barre-meta">YOLOv8n · avant / après fine-tuning ·
            4 classes de véhicules</span>
        </div>
        <p class="barre-intro">Déposez une image ou une vidéo de circulation :
        les deux modèles l'analysent côte à côte et l'application mesure ce que
        le fine-tuning change, en particulier sur les véhicules petits ou éloignés.</p>
    """, unsafe_allow_html=True)

    for cle in ("standard", "benin"):
        if modeles[cle] is None:
            st.error(f"Modèle {ETIQUETTES[cle]} indisponible : {erreurs.get(cle, 'non chargé')}")
    if not cles_actives:
        st.stop()

    onglet_image, onglet_video = st.tabs(["Sur une image", "Sur une vidéo"])

    # ----- Onglet image -----
    with onglet_image:
        televerse = st.file_uploader(
            "Déposez une photo de circulation",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Formats supportés : JPG, PNG, BMP, TIFF. Toutes les dimensions "
                 "conviennent : les très grandes images sont réduites à "
                 f"{COTE_MAX_IMAGE} px avant analyse.",
        )

        # Images d'exemple embarquées : la démo fonctionne sans photo sous la main.
        exemples = sorted(DOSSIER_EXEMPLES.glob("*.jpg")) if DOSSIER_EXEMPLES.exists() else []
        if exemples and televerse is None:
            st.markdown("…ou choisissez une scène du jeu de test :")
            colonnes = st.columns(len(exemples))
            for colonne, chemin in zip(colonnes, exemples):
                with colonne:
                    st.image(str(chemin), width="stretch")
                    if st.button("Tester cette scène", key=f"exemple_{chemin.stem}",
                                 type="primary", width="stretch"):
                        st.session_state["exemple_choisi"] = chemin.name

        octets, nom = None, None
        if televerse is not None:
            octets, nom = televerse.getvalue(), televerse.name
            st.session_state.pop("exemple_choisi", None)
        elif st.session_state.get("exemple_choisi"):
            chemin = DOSSIER_EXEMPLES / st.session_state["exemple_choisi"]
            if chemin.exists():
                octets, nom = chemin.read_bytes(), chemin.name

        if octets is not None:
            resultats = {}
            with st.spinner(f"Analyse de {nom}…"):
                for cle in cles_actives:
                    resultats[cle] = detecter_image(octets, seuil_confiance, seuil_iou, cle)

            colonnes = st.columns(len(cles_actives), gap="large")
            for colonne, cle in zip(colonnes, cles_actives):
                with colonne:
                    afficher_resultat_image(cle, *resultats[cle])

            st.divider()
            afficher_comparaison({cle: compter(resultats[cle][1]) for cle in cles_actives})

            colonnes = st.columns(len(cles_actives), gap="large")
            for colonne, cle in zip(colonnes, cles_actives):
                with colonne:
                    bouton_telechargement_image(cle, resultats[cle][0])

    # ----- Onglet vidéo -----
    with onglet_video:
        televerse = st.file_uploader(
            "Déposez une vidéo de circulation",
            type=["mp4", "avi", "mov", "mkv"],
            help="Formats supportés : MP4, AVI, MOV, MKV (100 Mo max). "
                 "Privilégiez un extrait court : l'analyse se fait sur CPU.",
        )
        pas = st.select_slider(
            "Analyser 1 image sur…", options=[1, 2, 3, 5, 10], value=2,
            help="Sauter des images accélère d'autant le traitement. "
                 "La vidéo de sortie garde la durée d'origine.",
        )

        if televerse is not None:
            octets_video = televerse.getvalue()
            cle_cache = (televerse.name, len(octets_video), seuil_confiance, seuil_iou, pas)

            # Résultats mémorisés dans la session : cliquer un bouton de
            # téléchargement (qui relance le script) ne retraite pas la vidéo.
            if st.session_state.get("video_cle") != cle_cache:
                barre = st.progress(0)
                statut_texte = st.empty()

                def progression(courante, total):
                    if total > 0:
                        barre.progress(min(courante / total, 1.0))
                    statut_texte.text(f"Analyse : image {courante}/{total or '?'}")

                with tempfile.NamedTemporaryFile(
                        suffix=Path(televerse.name).suffix, delete=False) as tmp:
                    tmp.write(octets_video)
                    chemin_tmp = Path(tmp.name)
                try:
                    sorties, traitees = traiter_video(
                        chemin_tmp, seuil_confiance, seuil_iou, pas,
                        rappel_progression=progression,
                    )
                except Exception as e:
                    st.error(f"Erreur pendant le traitement : {e}")
                    st.stop()
                finally:
                    chemin_tmp.unlink(missing_ok=True)

                barre.progress(1.0)
                statut_texte.text(f"Traitement terminé : {traitees} images analysées")
                st.session_state["video_cle"] = cle_cache
                st.session_state["video_sorties"] = sorties

            sorties = st.session_state["video_sorties"]

            colonnes = st.columns(len(sorties), gap="large")
            for colonne, (cle, sortie) in zip(colonnes, sorties.items()):
                with colonne:
                    st.markdown(entete_modele(cle), unsafe_allow_html=True)
                    st.video(sortie["octets"])
                    st.metric("Détections cumulées", sortie["detections"])

            st.divider()
            afficher_comparaison(
                {cle: (sortie["par_classe"], sortie["par_taille"])
                 for cle, sortie in sorties.items()}
            )

            colonnes = st.columns(len(sorties), gap="large")
            horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
            for colonne, (cle, sortie) in zip(colonnes, sorties.items()):
                with colonne:
                    st.download_button(
                        label=f"Télécharger la version {ETIQUETTES[cle]}",
                        data=sortie["octets"],
                        file_name=f"{cle}_vehicules_{horodatage}.mp4",
                        mime="video/mp4",
                        width="stretch",
                        key=f"dl_video_{cle}",
                    )

if __name__ == "__main__":
    main()
