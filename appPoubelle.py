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
# model_path = r"C:\Users\hp\Desktop\master2\deep learning\projetIndividuel1\runs\detect\poubelle_yolov8\weights\best.pt"
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

        # Sauvegarde temporaire
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(tfile.name)

        if st.button("🔍 Lancer la détection"):
            with st.spinner("⏳ Analyse vidéo en cours... Cela peut prendre un moment..."):
                
                cap = cv2.VideoCapture(tfile.name)
                output_path = "output_detected.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(3))
                height = int(cap.get(4))

                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                cap.release()
                out.release()
                #cv2.destroyAllWindows()

            st.success("🎉 Détection terminée !")

            st.markdown("<div class='box'>🟩 Vidéo annotée</div>", unsafe_allow_html=True)

            with open(output_path, "rb") as video_file:
                st.video(video_file.read())

            # Bouton téléchargement
            with open(output_path, "rb") as f:
                st.download_button("📥 Télécharger la vidéo annotée", f, file_name="video_detected.mp4")
