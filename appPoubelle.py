import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

# ==============================
# Configuration UI (design)
# ==============================
st.set_page_config(
    page_title="Détection Poubelle Pleine/Vide",
    page_icon="🗑️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS pour embellir l'UI
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px !important;
            color: #4CAF50;
            font-weight: bold;
        }
        .subtitle {
            color: #555;
            font-size: 20px;
            margin-bottom: 15px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .box {
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Charger modèle
# ==============================
model_path = "best.pt"
model = YOLO(model_path)

# ==============================
# Interface Streamlit
# ==============================
st.markdown("<h1 class='title'>🗑️ Détection Poubelle Pleine / Vide</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyse intelligente d’images et de vidéos avec YOLOv8</p>", unsafe_allow_html=True)

mode = st.radio("🎛️ Choisir le mode :", ["Image", "Vidéo"])

# ==============================
# Mode IMAGE
# ==============================
if mode == "Image":
    uploaded_file = st.file_uploader("📥 Importer une image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:

        st.markdown("<div class='box'>📷 Image originale</div>", unsafe_allow_html=True)
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)

        # Prédiction
        with st.spinner("🔍 Analyse de l'image en cours..."):
            results = model.predict(img)

        # Labels détectés
        detected_labels = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = results[0].names[cls]
            detected_labels.append(label)

        st.subheader("📝 Résultats de la prédiction")

        if len(detected_labels) == 0:
            st.error("❌ Aucune poubelle détectée")
        else:
            for label in detected_labels:
                if "vide" in label.lower():
                    st.success("🟢 Poubelle vide détectée")
                elif "pleine" in label.lower():
                    st.warning("🟡 Poubelle pleine détectée")
                else:
                    st.info(f"Objet détecté : {label}")

        # Image annotée
        st.markdown("<div class='box'>🖼️ Image annotée</div>", unsafe_allow_html=True)
        annotated_img = results[0].plot()
        st.image(annotated_img, use_column_width=True)

# ==============================
# Mode VIDEO
# ==============================
elif mode == "Vidéo":
    uploaded_video = st.file_uploader("📥 Importer une vidéo", type=["mp4", "avi", "mov"])
    if uploaded_video:

        st.markdown("<div class='box'>🎬 Vidéo originale</div>", unsafe_allow_html=True)

        # Sauvegarde du fichier source en temporaire
        src_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        src_tmp.write(uploaded_video.read())
        src_tmp.flush()
        src_path = src_tmp.name

        st.video(src_path)

        if st.button("🔍 Lancer la détection"):
            with st.spinner("⏳ Analyse vidéo en cours... Cela peut prendre un moment..."):
                cap = cv2.VideoCapture(src_path)
                if not cap.isOpened():
                    st.error("❌ Impossible d'ouvrir la vidéo source.")
                    st.stop()

                # Récupération des paramètres vidéo
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0:
                    fps = 24  # FPS par défaut si invalides

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width <= 0 or height <= 0:
                    st.error("❌ Dimensions invalides pour la vidéo.")
                    st.stop()

                # Fichier de sortie temporaire
                out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                output_path = out_tmp.name

                # Essayer un codec compatible (mp4)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # Fallback si l'ouverture échoue (AVI MJPG)
                if not out.isOpened():
                    out.release()
                    out_tmp.close()
                    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
                    output_path = out_tmp.name
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    st.error("❌ Échec de l'ouverture du fichier de sortie vidéo.")
                    cap.release()
                    st.stop()

                # Traitement frame par frame
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)
                    annotated = results[0].plot()

                    if annotated.shape[1] != width or annotated.shape[0] != height:
                        annotated = cv2.resize(annotated, (width, height))

                    out.write(annotated)

                cap.release()
                out.release()

            st.success("🎉 Détection terminée !")

            st.markdown("<div class='box'>🟩 Vidéo annotée</div>", unsafe_allow_html=True)

            with open(output_path, "rb") as vf:
                video_bytes = vf.read()

            if len(video_bytes) == 0:
                st.error("❌ La vidéo générée est vide. Réessaie avec une autre vidéo.")
            else:
                st.video(video_bytes)
                st.download_button("📥 Télécharger la vidéo annotée", data=video_bytes,
                                   file_name=os.path.basename(output_path), mime="video/mp4")
