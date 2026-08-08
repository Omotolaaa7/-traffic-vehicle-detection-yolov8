# app.py - Comparaison YOLOv8 : Standard vs Benin (détection des voitures uniquement)

import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os
import io
from datetime import datetime

# ============================================
# 1. CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="Comparaison YOLOv8 - Voitures",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 2. CSS PERSONNALISÉ
# ============================================
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
        font-size: 2.8rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.95;
        margin: 0.5rem 0 0 0;
    }
    .main-header .sub {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    .stat-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.12);
    }
    .stat-card .number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f3460;
    }
    .stat-card .label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .stat-card .model-name {
        font-size: 0.8rem;
        color: #999;
        margin-top: 0.3rem;
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

# ============================================
# 3. EN-TÊTE
# ============================================
st.markdown("""
    <div class="main-header">
        <h1>🚗 Comparaison YOLOv8 - Détection de Voitures</h1>
        <p>Comparez les performances des modèles Standard et Benin pour la détection de voitures</p>
        <div class="sub">⚡ Modèle Standard (COCO) vs Modèle Benin (entraîné sur vos données)</div>
    </div>
""", unsafe_allow_html=True)

# ============================================
# 4. BARRE LATÉRALE
# ============================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    st.markdown("#### 📊 Modèles chargés")
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 10px;">
        <div style="background: #eff6ff; padding: 10px; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <span class="badge-standard">📦 Standard</span>
            <p style="margin: 5px 0 0 0; font-size: 0.8rem; color: #666;">Entraîné sur COCO</p>
        </div>
        <div style="background: #ecfdf5; padding: 10px; border-radius: 8px; border-left: 4px solid #10b981;">
            <span class="badge-benin">🇧🇯 Benin</span>
            <p style="margin: 5px 0 0 0; font-size: 0.8rem; color: #666;">Entraîné sur voitures</p>
        </div>
    </div>
    <div style="margin-top: 10px; background: #fef3c7; padding: 8px; border-radius: 8px;">
        <p style="margin: 0; font-size: 0.8rem; color: #92400e;">🚗 Détection des voitures uniquement</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("#### 🎚️ Paramètres")
    
    confidence_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    
    iou_threshold = st.slider(
        "Seuil IOU",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05
    )
    
    st.markdown("---")
    
    # Sélection du mode
    st.markdown("#### 📌 Mode d'analyse")
    mode = st.radio(
        "",
        ["📷 Image", "🎥 Vidéo"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("#### 📊 Informations")
    with st.expander("ℹ️ À propos"):
        st.markdown("""
        - **Modèle Standard** : Détecte 80 classes (COCO)
        - **Modèle Benin** : Spécialisé en détection de voitures
        - **Comparaison** : Seules les voitures sont affichées
        """)
    
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.rerun()

# ============================================
# 5. CHARGEMENT DES MODÈLES
# ============================================
@st.cache_resource
def load_models():
    """Charge les deux modèles YOLOv8"""
    models = {}
    
    # Modèle Standard
    try:
        if not os.path.exists('yolov8n.pt'):
            st.warning("⚠️ Téléchargement de yolov8n.pt...")
            model_standard = YOLO('yolov8n.pt')
        else:
            model_standard = YOLO('yolov8n.pt')
        models['standard'] = model_standard
        st.sidebar.success("✅ Modèle Standard chargé")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur Standard : {e}")
        models['standard'] = None
    
    # Modèle Benin
    try:
        if not os.path.exists('yolov8n_benin.pt'):
            st.sidebar.error("❌ 'yolov8n_benin.pt' introuvable !")
            models['benin'] = None
        else:
            model_benin = YOLO('yolov8n_benin.pt')
            models['benin'] = model_benin
            st.sidebar.success("✅ Modèle Benin chargé")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur Benin : {e}")
        models['benin'] = None
    
    return models

models = load_models()

if models['standard'] is None and models['benin'] is None:
    st.error("❌ Aucun modèle chargé.")
    st.stop()

# ============================================
# 6. FONCTION : GARDER UNIQUEMENT LES VOITURES
# ============================================
def keep_only_cars(results):
    """Ne garde que les voitures (classe 2 pour Standard, classe personnalisée pour Benin)"""
    filtered_results = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            # Pour le modèle Standard, la classe 2 = car
            # Pour le modèle Benin, on garde toutes les classes (car il n'a que des voitures)
            car_indices = []
            for i, cls in enumerate(boxes.cls):
                class_id = int(cls)
                # Le modèle Standard utilise la classe 2 pour les voitures
                # Le modèle Benin n'a que des voitures (classe 0 généralement)
                if class_id == 2:  # COCO class 2 = car
                    car_indices.append(i)
            
            if car_indices:
                r.boxes = boxes[car_indices]
            else:
                r.boxes = boxes[[]]
        filtered_results.append(r)
    
    return filtered_results

# ============================================
# 7. SECTION IMAGE
# ============================================
if mode == "📷 Image":
    st.markdown("### 📷 Comparaison sur une image")
    
    uploaded_file = st.file_uploader(
        "📤 Choisissez une image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Formats supportés : JPG, PNG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        # Modèle Standard
        if models['standard'] is not None:
            status_text.text("🔍 Modèle Standard...")
            results['standard'] = models['standard'](image_np, conf=confidence_threshold, iou=iou_threshold)
            results['standard'] = keep_only_cars(results['standard'])
            progress_bar.progress(33)
        else:
            results['standard'] = None
        
        # Modèle Benin
        if models['benin'] is not None:
            status_text.text("🔍 Modèle Benin...")
            results['benin'] = models['benin'](image_np, conf=confidence_threshold, iou=iou_threshold)
            # Le modèle Benin n'a que des voitures, pas besoin de filtrer
            progress_bar.progress(66)
        else:
            results['benin'] = None
        
        progress_bar.progress(100)
        status_text.text("✅ Comparaison terminée !")
        
        # Image originale
        st.markdown("#### 📷 Image originale")
        st.image(image, use_container_width=True)
        st.divider()
        
        # Résultats côte à côte
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; background: #eff6ff; border-radius: 10px;">
                <span class="badge-standard">📦 Modèle Standard</span>
            </div>
            """, unsafe_allow_html=True)
            
            if results['standard'] is not None:
                for r in results['standard']:
                    nb_voitures = len(r.boxes)
                    annotated_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🚗 Voitures", nb_voitures)
                    with col_b:
                        if nb_voitures > 0:
                            conf_moy = sum(float(box.conf) for box in r.boxes) / nb_voitures
                            st.metric("📊 Confiance moyenne", f"{conf_moy:.1%}")
                        else:
                            st.metric("📊 Confiance moyenne", "-")
                    
                    if nb_voitures > 0:
                        with st.expander(f"📋 Détails des {nb_voitures} voitures"):
                            data = []
                            for i, box in enumerate(r.boxes, 1):
                                confiance = float(box.conf)
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                data.append({
                                    "N°": i,
                                    "Confiance": f"{confiance:.1%}",
                                    "Position": f"({x1}, {y1}) → ({x2}, {y2})"
                                })
                            st.dataframe(data, use_container_width=True, hide_index=True)
            else:
                st.warning("❌ Modèle Standard non disponible")
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; background: #ecfdf5; border-radius: 10px;">
                <span class="badge-benin">🇧🇯 Modèle Benin</span>
            </div>
            """, unsafe_allow_html=True)
            
            if results['benin'] is not None:
                for r in results['benin']:
                    nb_voitures = len(r.boxes)
                    annotated_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🚗 Voitures", nb_voitures)
                    with col_b:
                        if nb_voitures > 0:
                            conf_moy = sum(float(box.conf) for box in r.boxes) / nb_voitures
                            st.metric("📊 Confiance moyenne", f"{conf_moy:.1%}")
                        else:
                            st.metric("📊 Confiance moyenne", "-")
                    
                    if nb_voitures > 0:
                        with st.expander(f"📋 Détails des {nb_voitures} voitures"):
                            data = []
                            for i, box in enumerate(r.boxes, 1):
                                confiance = float(box.conf)
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                data.append({
                                    "N°": i,
                                    "Confiance": f"{confiance:.1%}",
                                    "Position": f"({x1}, {y1}) → ({x2}, {y2})"
                                })
                            st.dataframe(data, use_container_width=True, hide_index=True)
            else:
                st.warning("❌ Modèle Benin non disponible")
        
        # Statistiques de comparaison
        st.divider()
        st.markdown("### 📊 Comparaison des performances")
        
        nb_standard = 0
        nb_benin = 0
        
        if results['standard'] is not None:
            for r in results['standard']:
                nb_standard = len(r.boxes)
        
        if results['benin'] is not None:
            for r in results['benin']:
                nb_benin = len(r.boxes)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            diff = nb_benin - nb_standard
            if diff > 0:
                st.metric("📈 Différence", f"+{diff} voitures", "Le modèle Benin détecte plus de voitures")
            elif diff < 0:
                st.metric("📉 Différence", f"{diff} voitures", "Le modèle Standard détecte plus de voitures")
            else:
                st.metric("📊 Différence", "Égal", "Les deux modèles détectent le même nombre")
        
        with col2:
            if nb_standard > 0:
                ratio = nb_benin / nb_standard
                st.metric("📊 Ratio Benin/Standard", f"{ratio:.2f}x", f"Benin détecte {ratio:.1f}x plus de voitures")
            else:
                st.metric("📊 Ratio Benin/Standard", "N/A", "Pas de détection Standard")
        
        with col3:
            total = nb_standard + nb_benin
            st.metric("🚗 Total voitures", total, "Détectées par les deux modèles")
        
        # Téléchargement
        st.divider()
        st.markdown("### 💾 Télécharger les résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if results['standard'] is not None:
                for r in results['standard']:
                    annotated_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
                    annotated_pil = Image.fromarray(annotated_rgb)
                    buf = io.BytesIO()
                    annotated_pil.save(buf, format="JPEG")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📦 Télécharger - Modèle Standard",
                        data=buf.getvalue(),
                        file_name=f"standard_voitures_{timestamp}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
        
        with col2:
            if results['benin'] is not None:
                for r in results['benin']:
                    annotated_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
                    annotated_pil = Image.fromarray(annotated_rgb)
                    buf = io.BytesIO()
                    annotated_pil.save(buf, format="JPEG")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="🇧🇯 Télécharger - Modèle Benin",
                        data=buf.getvalue(),
                        file_name=f"benin_voitures_{timestamp}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )

# ============================================
# 8. SECTION VIDÉO
# ============================================
else:
    st.markdown("### 🎥 Comparaison sur une vidéo")
    st.info("🚗 Les deux modèles analysent la vidéo frame par frame - Seules les voitures sont détectées")
    
    uploaded_video = st.file_uploader(
        "📤 Choisissez une vidéo",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Formats supportés : MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video is not None:
        # Sauvegarder la vidéo
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        st.info("⏳ Traitement de la vidéo...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Statistiques globales
        stats_standard = {"frames": 0, "voitures": 0}
        stats_benin = {"frames": 0, "voitures": 0}
        
        try:
            # Traitement Standard
            if models['standard'] is not None:
                status_text.text("🎬 Traitement avec le modèle Standard...")
                
                cap = cv2.VideoCapture("temp_video.mp4")
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_standard = cv2.VideoWriter('output_standard.mp4', fourcc, fps, (width, height))
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = models['standard'](frame, conf=confidence_threshold, iou=iou_threshold)
                    results = keep_only_cars(results)
                    
                    for r in results:
                        stats_standard["voitures"] += len(r.boxes)
                    
                    annotated = results[0].plot()
                    out_standard.write(annotated)
                    
                    frame_count += 1
                    if total_frames > 0:
                        progress = int((frame_count / total_frames) * 50)
                        progress_bar.progress(progress)
                        status_text.text(f"🎬 Standard : {frame_count}/{total_frames} frames")
                
                cap.release()
                out_standard.release()
                stats_standard["frames"] = frame_count
            
            # Traitement Benin
            if models['benin'] is not None:
                status_text.text("🎬 Traitement avec le modèle Benin...")
                
                cap = cv2.VideoCapture("temp_video.mp4")
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_benin = cv2.VideoWriter('output_benin.mp4', fourcc, fps, (width, height))
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = models['benin'](frame, conf=confidence_threshold, iou=iou_threshold)
                    
                    for r in results:
                        stats_benin["voitures"] += len(r.boxes)
                    
                    annotated = results[0].plot()
                    out_benin.write(annotated)
                    
                    frame_count += 1
                    if total_frames > 0:
                        progress = 50 + int((frame_count / total_frames) * 50)
                        progress_bar.progress(progress)
                        status_text.text(f"🎬 Benin : {frame_count}/{total_frames} frames")
                
                cap.release()
                out_benin.release()
                stats_benin["frames"] = frame_count
            
            cv2.destroyAllWindows()
            os.remove("temp_video.mp4")
            
            status_text.text("✅ Traitement terminé !")
            
            # Afficher les vidéos côte à côte
            st.markdown("### 📹 Résultats vidéo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<span class="badge-standard">📦 Modèle Standard</span>', unsafe_allow_html=True)
                if os.path.exists("output_standard.mp4"):
                    with open("output_standard.mp4", "rb") as f:
                        st.video(f.read())
                    st.metric("🚗 Voitures détectées", stats_standard["voitures"])
                else:
                    st.warning("Vidéo Standard non disponible")
            
            with col2:
                st.markdown('<span class="badge-benin">🇧🇯 Modèle Benin</span>', unsafe_allow_html=True)
                if os.path.exists("output_benin.mp4"):
                    with open("output_benin.mp4", "rb") as f:
                        st.video(f.read())
                    st.metric("🚗 Voitures détectées", stats_benin["voitures"])
                else:
                    st.warning("Vidéo Benin non disponible")
            
            # Statistiques de comparaison
            st.divider()
            st.markdown("### 📊 Comparaison vidéo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                diff = stats_benin["voitures"] - stats_standard["voitures"]
                if diff > 0:
                    st.metric("📈 Différence", f"+{diff} voitures", "Benin détecte plus de voitures")
                elif diff < 0:
                    st.metric("📉 Différence", f"{diff} voitures", "Standard détecte plus de voitures")
                else:
                    st.metric("📊 Différence", "Égal", "Même nombre de voitures")
            
            with col2:
                if stats_standard["voitures"] > 0:
                    ratio = stats_benin["voitures"] / stats_standard["voitures"]
                    st.metric("📊 Ratio Benin/Standard", f"{ratio:.2f}x", f"Benin détecte {ratio:.1f}x plus de voitures")
                else:
                    st.metric("📊 Ratio Benin/Standard", "N/A", "Pas de détection Standard")
            
            with col3:
                total = stats_standard["voitures"] + stats_benin["voitures"]
                st.metric("🚗 Total voitures", total, "Détectées par les deux modèles")
            
            # Téléchargement
            st.divider()
            st.markdown("### 💾 Télécharger les vidéos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists("output_standard.mp4"):
                    with open("output_standard.mp4", "rb") as f:
                        video_bytes = f.read()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📦 Télécharger - Standard",
                        data=video_bytes,
                        file_name=f"standard_voitures_{timestamp}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                    os.remove("output_standard.mp4")
            
            with col2:
                if os.path.exists("output_benin.mp4"):
                    with open("output_benin.mp4", "rb") as f:
                        video_bytes = f.read()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="🇧🇯 Télécharger - Benin",
                        data=video_bytes,
                        file_name=f"benin_voitures_{timestamp}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                    os.remove("output_benin.mp4")
        
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
            if os.path.exists("temp_video.mp4"):
                os.remove("temp_video.mp4")

# ============================================
# 9. PIED DE PAGE
# ============================================
st.divider()
st.markdown("""
<div class="footer">
    <p>🚗 Comparaison YOLOv8 - Détection de Voitures | Standard vs Benin</p>
    <p style='font-size: 0.8rem;'>© 2024 - Développé avec ❤️ en utilisant Streamlit</p>
</div>
""", unsafe_allow_html=True)
