import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ==============================================================================
# 1. CARGA DE DATOS PREPROCESADOS
# ==============================================================================
print("Cargando dataset procesado...")
df = pd.read_csv("dataset_procesado_final.csv")

# Asegurarnos de que no hay nulos (por seguridad)
df = df.dropna(subset=['input_text'])

print(f"Total de noticias: {len(df)}")
print(df.head())

# ==============================================================================
# 2. DIVISIÓN TRAIN / TEST (Requisito obligatorio del proyecto)
# ==============================================================================
# Separamos el 20% para test y 80% para entrenamiento
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['input_text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label'] # Mantiene la proporción de clases
)

print(f"\nConjunto de Entrenamiento: {len(X_train_text)} noticias")
print(f"Conjunto de Test: {len(X_test_text)} noticias")

# ==============================================================================
# 3. REPRESENTACIÓN VECTORIAL: TF-IDF
# ==============================================================================
print("\nGenerando vectores TF-IDF...")

# Configuración: Usamos un máximo de 5000 palabras más importantes para no saturar la RAM
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Ajustamos (fit) solo con el train y transformamos ambos
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print(f"Dimensiones de la matriz TF-IDF (Train): {X_train_tfidf.shape}")

# ==============================================================================
# 4. MODELADO BASELINE (Scikit-Learn: Regresión Logística)
# ==============================================================================
print("\nEntrenando modelo base (Regresión Logística)...")

# La Regresión Logística es excelente para textos con TF-IDF
modelo_base = LogisticRegression(max_iter=1000)
modelo_base.fit(X_train_tfidf, y_train)

# ==============================================================================
# 5. EVALUACIÓN
# ==============================================================================
print("\n--- RESULTADOS DEL MODELO BASE (TF-IDF) ---")

y_pred = modelo_base.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print(f"ACCURACY (Precisión Global): {acc:.4f}")
print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=['Neutro (0)', 'Hiperpartidista (1)']))

# Guardar el vectorizador para usarlo después si hace falta
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Vectorizador guardado como 'tfidf_vectorizer.pkl'")