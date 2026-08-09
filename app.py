# app.py : application de démonstration
# YOLOv8 pré-entraîné (COCO) contre YOLOv8 fine-tuné (Benin).
#
# Compare les deux modèles du projet sur les 4 classes de véhicules
# (voiture, moto, bus, camion), sur image ou sur vidéo, avec une
# ventilation des détections par taille apparente d'objet, qui est la
# question centrale du sujet (véhicules éloignés et de petite taille).

import io
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import imageio.v2 as imageio
import streamlit as st
from PIL import Image
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

# Seuils COCO de taille d'objet, appliqués après remise à l'échelle de
# l'image vers la résolution d'entrée du réseau (voir README, section 9 :
# sans cette normalisation, tout objet d'une photo haute résolution est
# classé « grand » et l'analyse ne mesure plus rien).
RESOLUTION_RESEAU = 640
SEUIL_PETIT = 32 * 32
SEUIL_MOYEN = 96 * 96

ETIQUETTES = {"standard": "📦 Modèle Standard", "benin": "🇧🇯 Modèle Benin"}


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
    téléchargement ou changer de mode ne relance pas l'inférence."""
    modeles, _ = charger_modeles()
    modele = modeles[cle_modele]
    image = Image.open(io.BytesIO(octets_image)).convert("RGB")
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


def afficher_resultat_image(cle, annotee, detections):
    couleur = "#eff6ff" if cle == "standard" else "#ecfdf5"
    badge = "badge-standard" if cle == "standard" else "badge-benin"
    st.markdown(
        f'<div style="text-align: center; padding: 0.5rem; background: {couleur}; '
        f'border-radius: 10px;"><span class="{badge}">{ETIQUETTES[cle]}</span></div>',
        unsafe_allow_html=True,
    )
    st.image(annotee, width="stretch")

    par_classe, _ = compter(detections)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🚗 Véhicules détectés", len(detections))
    with col_b:
        if detections:
            conf_moyenne = sum(d["confiance"] for d in detections) / len(detections)
            st.metric("📊 Confiance moyenne", f"{conf_moyenne:.1%}")
        else:
            st.metric("📊 Confiance moyenne", "n/a")

    if detections:
        st.caption(" · ".join(f"{par_classe[c]} {c}(s)" for c in ORDRE_CLASSES if par_classe[c]))
        with st.expander(f"📋 Détail des {len(detections)} détections"):
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
    """Tableaux comparatifs par classe et par taille d'objet.

    stats_par_modele[cle] = (par_classe, par_taille), Counters."""
    cles = list(stats_par_modele)
    st.markdown("### 📊 Comparaison des détections")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Par classe**")
        st.dataframe(
            [{"Classe": c, **{ETIQUETTES[k]: stats_par_modele[k][0].get(c, 0) for k in cles}}
             for c in ORDRE_CLASSES]
            + [{"Classe": "Total",
                **{ETIQUETTES[k]: sum(stats_par_modele[k][0].values()) for k in cles}}],
            width="stretch", hide_index=True,
        )
    with col2:
        st.markdown("**Par taille apparente** (seuils COCO, normalisés à la résolution du réseau)")
        st.dataframe(
            [{"Taille": t, **{ETIQUETTES[k]: stats_par_modele[k][1].get(t, 0) for k in cles}}
             for t in ORDRE_TAILLES],
            width="stretch", hide_index=True,
        )

    if len(cles) == 2:
        ecart = (sum(stats_par_modele["benin"][0].values())
                 - sum(stats_par_modele["standard"][0].values()))
        petits = (stats_par_modele["benin"][1].get("petit", 0)
                  - stats_par_modele["standard"][1].get("petit", 0))
        signe = "+" if ecart >= 0 else ""
        signe_p = "+" if petits >= 0 else ""
        st.metric("Écart total (Benin moins Standard)", f"{signe}{ecart} détections",
                  f"{signe_p}{petits} sur les objets petits", delta_color="off")
    st.caption(
        "⚠️ Un nombre de détections plus élevé n'est pas en soi une preuve "
        "d'amélioration (il peut contenir des faux positifs). La mesure de "
        "référence reste le tableau `results/comparaison.md`, calculé sur le "
        "jeu de test annoté. La ligne à regarder ici : les objets **petits**, "
        "cœur de la problématique."
    )


def bouton_telechargement_image(cle, annotee):
    image_pil = Image.fromarray(annotee)
    tampon = io.BytesIO()
    image_pil.save(tampon, format="JPEG", quality=92)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label=f"💾 Télécharger : {ETIQUETTES[cle]}",
        data=tampon.getvalue(),
        file_name=f"{cle}_vehicules_{horodatage}.jpg",
        mime="image/jpeg",
        width="stretch",
        key=f"dl_image_{cle}",
    )


# ============================================
# APPLICATION
# ============================================
def main():
    st.set_page_config(
        page_title="Comparaison YOLOv8 : véhicules",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .main-header h1 {
            font-size: 2.4rem;
            margin: 0;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.95;
            margin: 0.5rem 0 0 0;
        }
        .main-header .sub {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 0.5rem;
        }
        .badge-standard {
            display: inline-block;
            padding: 0.3rem 1.2rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            background: #3b82f6;
            color: white;
        }
        .badge-benin {
            display: inline-block;
            padding: 0.3rem 1.2rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            background: #10b981;
            color: white;
        }
        .footer {
            text-align: center;
            padding: 2rem;
            color: #999;
            font-size: 0.9rem;
            border-top: 2px solid #f0f0f0;
            margin-top: 3rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-header">
            <h1>🚗 Détection de véhicules en trafic urbain dense</h1>
            <p>YOLOv8 pré-entraîné (COCO) contre YOLOv8 fine-tuné sur trafic dense</p>
            <div class="sub">voiture · moto · bus · camion (AMA Cohorte 2, Groupe 5)</div>
        </div>
    """, unsafe_allow_html=True)

    # ----- Barre latérale -----
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

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

        st.markdown("---")
        st.markdown("#### 📌 Mode d'analyse")
        mode = st.radio("Mode d'analyse", ["📷 Image", "🎥 Vidéo"],
                        label_visibility="collapsed")

        st.markdown("---")
        st.markdown("#### 📊 Modèles")
        modeles, erreurs = charger_modeles()
        for cle in ("standard", "benin"):
            if modeles[cle] is not None:
                st.success(f"✅ {ETIQUETTES[cle]}")
            else:
                st.error(f"❌ {ETIQUETTES[cle]} : {erreurs.get(cle, 'non chargé')}")

        with st.expander("ℹ️ À propos"):
            st.markdown("""
            - **Standard** : YOLOv8n pré-entraîné sur COCO (80 classes),
              restreint ici aux 4 classes de véhicules.
            - **Benin** : le même YOLOv8n, fine-tuné sur un jeu de trafic
              urbain dense (BMD-45).
            - La ventilation **par taille d'objet** répond à la question du
              sujet : les véhicules éloignés / petits sont-ils mieux détectés ?
            """)

    cles_actives = [cle for cle in ("standard", "benin") if modeles[cle] is not None]
    if not cles_actives:
        st.error("❌ Aucun modèle chargé, l'application ne peut pas fonctionner.")
        st.stop()

    # ----- Mode image -----
    if mode == "📷 Image":
        st.markdown("### 📷 Comparaison sur une image")

        televerse = st.file_uploader(
            "📤 Choisissez une image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Formats supportés : JPG, PNG, BMP, TIFF",
        )

        # Images d'exemple embarquées : la démo fonctionne sans photo sous la main.
        exemples = sorted(DOSSIER_EXEMPLES.glob("*.jpg")) if DOSSIER_EXEMPLES.exists() else []
        if exemples and televerse is None:
            st.markdown("… ou essayez avec une scène du jeu de test :")
            colonnes = st.columns(len(exemples))
            for colonne, chemin in zip(colonnes, exemples):
                with colonne:
                    st.image(str(chemin), width="stretch")
                    if st.button("Essayer cette image", key=f"exemple_{chemin.stem}",
                                 width="stretch"):
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
            with st.spinner(f"🔍 Analyse de {nom}…"):
                for cle in cles_actives:
                    resultats[cle] = detecter_image(octets, seuil_confiance, seuil_iou, cle)

            colonnes = st.columns(len(cles_actives))
            for colonne, cle in zip(colonnes, cles_actives):
                with colonne:
                    afficher_resultat_image(cle, *resultats[cle])

            st.divider()
            afficher_comparaison({cle: compter(resultats[cle][1]) for cle in cles_actives})

            st.divider()
            st.markdown("### 💾 Télécharger les images annotées")
            colonnes = st.columns(len(cles_actives))
            for colonne, cle in zip(colonnes, cles_actives):
                with colonne:
                    bouton_telechargement_image(cle, resultats[cle][0])

    # ----- Mode vidéo -----
    else:
        st.markdown("### 🎥 Comparaison sur une vidéo")
        st.info("Les deux modèles analysent la vidéo en une seule passe, "
                "image par image. Les 4 classes de véhicules sont détectées.")

        televerse = st.file_uploader(
            "📤 Choisissez une vidéo",
            type=["mp4", "avi", "mov", "mkv"],
            help="Formats supportés : MP4, AVI, MOV, MKV (100 Mo max)",
        )
        pas = st.select_slider(
            "Analyser 1 image sur…", options=[1, 2, 3, 5, 10], value=2,
            help="Sauter des images accélère d'autant le traitement (utile en "
                 "déploiement, où l'inférence se fait sur CPU). La vidéo de "
                 "sortie garde la durée d'origine.",
        )

        if televerse is not None:
            octets_video = televerse.getvalue()
            cle_cache = (televerse.name, len(octets_video), seuil_confiance, seuil_iou, pas)

            # Résultats mémorisés dans la session : cliquer un bouton de
            # téléchargement (qui relance le script) ne retraite pas la vidéo.
            if st.session_state.get("video_cle") != cle_cache:
                barre = st.progress(0)
                statut = st.empty()

                def progression(courante, total):
                    if total > 0:
                        barre.progress(min(courante / total, 1.0))
                    statut.text(f"🎬 Analyse : image {courante}/{total or '?'}")

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
                    st.error(f"❌ Erreur pendant le traitement : {e}")
                    st.stop()
                finally:
                    chemin_tmp.unlink(missing_ok=True)

                barre.progress(1.0)
                statut.text(f"✅ Traitement terminé : {traitees} images analysées")
                st.session_state["video_cle"] = cle_cache
                st.session_state["video_sorties"] = sorties

            sorties = st.session_state["video_sorties"]

            st.markdown("### 📹 Vidéos annotées")
            colonnes = st.columns(len(sorties))
            for colonne, (cle, sortie) in zip(colonnes, sorties.items()):
                with colonne:
                    badge = "badge-standard" if cle == "standard" else "badge-benin"
                    st.markdown(f'<span class="{badge}">{ETIQUETTES[cle]}</span>',
                                unsafe_allow_html=True)
                    st.video(sortie["octets"])
                    st.metric("🚗 Détections cumulées", sortie["detections"])

            st.divider()
            afficher_comparaison(
                {cle: (sortie["par_classe"], sortie["par_taille"])
                 for cle, sortie in sorties.items()}
            )

            st.divider()
            st.markdown("### 💾 Télécharger les vidéos annotées")
            colonnes = st.columns(len(sorties))
            horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
            for colonne, (cle, sortie) in zip(colonnes, sorties.items()):
                with colonne:
                    st.download_button(
                        label=f"💾 Télécharger : {ETIQUETTES[cle]}",
                        data=sortie["octets"],
                        file_name=f"{cle}_vehicules_{horodatage}.mp4",
                        mime="video/mp4",
                        width="stretch",
                        key=f"dl_video_{cle}",
                    )

    # ----- Pied de page -----
    st.divider()
    st.markdown("""
    <div class="footer">
        <p>🚗 Détection de véhicules en trafic urbain dense : YOLOv8 Standard vs fine-tuné</p>
        <p style='font-size: 0.8rem;'>AMA Cohorte 2, Projet Intégrateur 1, Groupe 5. Images d'exemple : jeu BMD-45 (CC BY 4.0)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
