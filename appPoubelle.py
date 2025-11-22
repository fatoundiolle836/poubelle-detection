import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.title("🚮 Détection de Poubelles avec YOLOv8")

# Charger le modèle
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Lire l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image envoyée", use_column_width=True)

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Prédiction
    results = model.predict(temp_path)

    # Affichage résultat
    st.subheader("Résultats de la détection")
    result_image = results[0].plot()  # image annotée

    st.image(result_image, caption="Détection", use_column_width=True)
