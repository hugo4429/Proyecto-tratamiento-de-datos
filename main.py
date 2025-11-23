# --- IMPORTACIONES PRINCIPALES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
from transformers import (
    AutoTokenizer, AutoModel, 
    AutoModelForSequenceClassification, Trainer, TrainingArguments
)
from datasets import Dataset
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec

# (Asegúrate de tener pandas y openpyxl instalados: pip install pandas openpyxl)

# Configurar semilla para reproducibilidad
np.random.seed(42)
torch.manual_seed(42)

# --- PASO 0: Carga y Preparación de Datos (Versión Excel) ---
print("--- Iniciando Paso 0: Carga y Preparación de Datos ---")

# <<< ATENCIÓN: MODIFICA ESTOS NOMBRES >>>
excel_file_path = 'Datasheet.xlsx'
sheet_name_real = 'politifact_real' 
sheet_name_fake = 'politifact_real'  
# <<< >>>

try:
    # Cargar la hoja de noticias reales
    df_real = pd.read_excel(excel_file_path, sheet_name=sheet_name_real)
    
    # Cargar la hoja de noticias falsas
    df_fake = pd.read_excel(excel_file_path, sheet_name=sheet_name_fake)

    # Crear la etiqueta (label)
    df_real['label'] = 0  # 0 para noticias reales
    df_fake['label'] = 1  # 1 para desinformación

    # Combinar ambos dataframes en uno solo
    df = pd.concat([df_real, df_fake], ignore_index=True)

    # ¡Importante! Barajar los datos
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Datos cargados y combinados desde '{excel_file_path}'. Total de muestras: {len(df)}")
    print(df.head())

except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{excel_file_path}'. Asegúrate de que el nombre es correcto.")
    # exit()
except Exception as e:
    print(f"Error al leer el archivo Excel: {e}")
    print("Por favor, verifica que los nombres de las hojas ('sheet_name_real' y 'sheet_name_fake') son correctos.")
    # exit()