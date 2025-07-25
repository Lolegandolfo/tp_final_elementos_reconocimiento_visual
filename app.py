import streamlit as st
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import joblib

# Parámetros HOG y dimensiones actualizados a 128×128
ORIENTATIONS = 9
PPC = (8, 8)
CPB = (1, 1)
BN = 'L2-Hys'
IMAGE_SHAPE = (128, 128)

@st.cache_resource
def load_resources():
    # rf_model.pkl y codebook.pkl deben existir junto a este script
    rf = joblib.load('rf_model.pkl')
    codebook = joblib.load('codebook.pkl')
    return rf, codebook

# Cargar modelo y diccionario visual
rf_model, codebook = load_resources()

st.title('Detector de Caras con HOG + BoVW + RandomForest')

# Panel lateral de subida o cámara
st.sidebar.header('Carga de Imagen')
img_file = st.sidebar.file_uploader('Sube una foto', type=['png','jpg','jpeg'])
img_cam  = st.sidebar.camera_input('O toma una foto')
input_img = img_file or img_cam

if input_img is not None:
    # Leer y mostrar imagen original
    img = imread(input_img)
    st.image(img, caption='Imagen de entrada', use_column_width=True)

    # Preprocesamiento: escala de grises y resize a 128×128
    if img.ndim == 3:
        img = rgb2gray(img)
    img_resized = resize(img, IMAGE_SHAPE, anti_aliasing=True)

    # Extraer HOG con feature_vector=False para obtener parches
    hog_desc = hog(
        img_resized,
        orientations=ORIENTATIONS,
        pixels_per_cell=PPC,
        cells_per_block=CPB,
        block_norm=BN,
        feature_vector=False
    )
    desc = hog_desc.reshape(-1, ORIENTATIONS)

    # Construir histograma BoVW usando codebook cargado
    dists = np.linalg.norm(desc[:, None, :] - codebook[None, :, :], axis=2)
    assigns = np.argmin(dists, axis=1)
    hist, _ = np.histogram(assigns, bins=np.arange(len(codebook) + 1))
    feat = hist.astype(float) / hist.sum()

    # Predicción y probabilidad
    pred = rf_model.predict(feat.reshape(1, -1))[0]
    prob = rf_model.predict_proba(feat.reshape(1, -1))[0, 1]

    st.write('**Predicción:**', 'Cara' if pred == 1 else 'No‑cara')
    st.write(f'**Probabilidad:** {prob:.2f}')

    # Mostrar histograma de BoVW
    st.subheader('Histograma de BoVW')
    st.bar_chart(feat)
