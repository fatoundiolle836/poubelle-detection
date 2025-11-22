import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

st.title("🚮 Détection de Poubelles avec YOLOv8")

# Charger le modèle
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Choix du mode
mode = st.radio("🎛️ Choisir le mode :", ["Image", "Vidéo"])

# ==============================
# Mode IMAGE
# ==============================
if mode == "Image":
    uploaded_file = st.file_uploader("📥 Importer une image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image envoyée", use_column_width=True)

        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # Prédiction
        results = model.predict(temp_path)

        st.subheader("Résultats de la détection")
        result_image = results[0].plot()
        st.image(result_image, caption="Détection", use_column_width=True)

# ==============================
# Mode VIDEO
# ==============================
elif mode == "Vidéo":
    uploaded_video = st.file_uploader("📥 Importer une vidéo", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # Sauvegarde temporaire
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(tfile.name)

        if st.button("🔍 Lancer la détection"):
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
            cv2.destroyAllWindows()

            st.success("🎉 Détection terminée !")

            st.markdown("### 🖼️ Vidéo annotée")
            with open(output_path, "rb") as video_file:
                st.video(video_file.read())

            with open(output_path, "rb") as f:
                st.download_button("📥 Télécharger la vidéo annotée", f, file_name="video_detected.mp4")
