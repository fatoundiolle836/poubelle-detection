import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import numpy as np

# ==============================
# Configuration UI
# ==============================
st.set_page_config(
    page_title="Détection Poubelle Pleine/Vide",
    page_icon="🗑️",
    layout="centered",
)

# ==============================
# Charger modèle
# ==============================
model = YOLO("best.pt")

# ==============================
# Interface
# ==============================
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>🗑️ Détection Poubelle Pleine / Vide</h1>", unsafe_allow_html=True)

mode = st.radio("🎛️ Choisir le mode :", ["Image", "Vidéo"])

# ===================================================================
# MODE IMAGE
# ===================================================================
if mode == "Image":
    uploaded_file = st.file_uploader("📥 Importer une image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Image originale", use_column_width=True)

        with st.spinner("🔍 Analyse de l'image en cours..."):
            results = model.predict(img)

        detected_labels = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = results[0].names[cls]
            detected_labels.append(label)

        st.subheader("📝 Résultats")

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

        st.subheader("🖼️ Image annotée")
        annotated = results[0].plot()
        st.image(annotated, use_column_width=True)

# ===================================================================
# MODE VIDEO
# ===================================================================
elif mode == "Vidéo":
    uploaded_video = st.file_uploader("📥 Importer une vidéo", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(tfile.name)

        if st.button("🔍 Lancer la détection"):

            st.warning("""
            ⏳ **La détection est en cours…**
            Cela peut durer **15 à 30 secondes** selon la vidéo.  
            👉 *Ne fermez surtout pas la page.*
            """)

            with st.spinner("Analyse vidéo…"):

                cap = cv2.VideoCapture(tfile.name)

                # ⚡ Optimisation : réduire la résolution
                target_width = 640
                target_height = 360

                # ⚡ FPS réduit pour accélérer
                fps = 15

                output_path = "output_detected.webm"
                fourcc = cv2.VideoWriter_fourcc(*"VP90")
                out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0)

                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Resize -> accélère tout
                    frame = cv2.resize(frame, (target_width, target_height))

                    # Prédiction YOLO
                    results = model(frame)
                    annotated_frame = results[0].plot()

                    out.write(annotated_frame)

                    frame_idx += 1
                    progress.progress(frame_idx / total_frames)

                cap.release()
                out.release()

            st.success("🎉 Détection terminée !")

            st.subheader("🟩 Vidéo annotée")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button("📥 Télécharger la vidéo annotée", f, file_name="video_detected.webm")
