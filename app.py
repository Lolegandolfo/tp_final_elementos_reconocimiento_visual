import streamlit as st
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import joblib

# Parámetros HOG y dimensiones
ORIENTATIONS = 9
PPC = (8, 8)
CPB = (1, 1)
BN = 'L2-Hys'
IMAGE_SHAPE = (128, 128)

@st.cache_resource
def load_resources():
    """
    Carga el modelo RandomForest y el codebook preentrenados desde archivos pickle.
    Asegura que los archivos se carguen desde el mismo directorio que este script.
    """
    # Obtiene el directorio donde está este script
    dir_path = os.path.dirname(os.path.abspath(__file__))
    rf_path = os.path.join(dir_path, 'rf_model.pkl')
    cb_path = os.path.join(dir_path, 'codebook.pkl')
    # Verifica existencia de archivos
    if not os.path.isfile(rf_path) or not os.path.isfile(cb_path):
        raise FileNotFoundError(f"Archivos rf_model.pkl o codebook.pkl no encontrados en {dir_path}")
    # Carga con joblib
    rf = joblib.load(rf_path)
    codebook = joblib.load(cb_path)
    return rf, codebook

# Cargar recursos
rf_model, codebook = load_resources()

st.title('Detector de Caras con HOG + BoVW + RandomForest')

# Panel lateral: opción de cargar imagen o usar cámara
st.sidebar.header('Carga de Imagen')
img_file = st.sidebar.file_uploader('Sube una foto', type=['png','jpg','jpeg'])
img_cam  = st.sidebar.camera_input('O toma una foto')
input_img = img_file or img_cam

if input_img is not None:
    # Leer y mostrar imagen original
    img = imread(input_img)
    st.image(img, caption='Imagen de entrada', use_column_width=True)

    # Preprocesamiento: conversión a gris y redimensionado a 128×128
    if img.ndim == 3:
        img = rgb2gray(img)
    img_resized = resize(img, IMAGE_SHAPE, anti_aliasing=True)

    # Extracción de descriptor HOG
    hog_desc = hog(
        img_resized,
        orientations=ORIENTATIONS,
        pixels_per_cell=PPC,
        cells_per_block=CPB,
        block_norm=BN,
        feature_vector=False
    )
    desc = hog_desc.reshape(-1, ORIENTATIONS)

    # Construcción de histograma BoVW usando el codebook cargado
    dists = np.linalg.norm(desc[:, None, :] - codebook[None, :, :], axis=2)
    assigns = np.argmin(dists, axis=1)
    hist, _ = np.histogram(assigns, bins=np.arange(codebook.shape[0] + 1))
    feat = hist.astype(float) / hist.sum()

    # Predicción con RandomForest
    pred = rf_model.predict(feat.reshape(1, -1))[0]
    prob = rf_model.predict_proba(feat.reshape(1, -1))[0, 1]

    # Mostrar resultados
    st.write('**Predicción:**', 'Cara' if pred == 1 else 'No‑cara')
    st.write(f'**Probabilidad de cara:** {prob:.2f}')

    # Visualizar histograma de BoVW
    st.subheader('Histograma de BoVW')
    st.bar_chart(feat)
