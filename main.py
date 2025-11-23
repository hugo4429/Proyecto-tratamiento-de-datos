from datasets import load_dataset
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# --- 1. CARGA DE DATOS ---
# ==============================================================================
print("Cargando dataset...")
# Usamos trust_remote_code=True para evitar errores de seguridad con versiones nuevas de 'datasets'
dataset = load_dataset("hyperpartisan_news_detection", "byarticle", trust_remote_code=True)


# ==============================================================================
# --- 2. DEFINICIÓN DE FUNCIONES DE LIMPIEZA ---
# ==============================================================================
def limpiar_fila(fila):
    """
    Elimina etiquetas HTML y limpia espacios en blanco.
    """
    texto_sucio = fila['text']
    if texto_sucio:
        soup = BeautifulSoup(texto_sucio, "html.parser")
        # separator=' ' evita que se peguen palabras de párrafos distintos
        texto_limpio = soup.get_text(separator=' ').strip()
    else:
        texto_limpio = ""
    
    return {"text_clean": texto_limpio}


# ==============================================================================
# --- 3. APLICACIÓN DE LIMPIEZA ---
# ==============================================================================
print("Limpiando todo el dataset (esto puede tardar unos segundos)...")
dataset_limpio = dataset.map(limpiar_fila)

# Convertimos a DataFrame de Pandas para el análisis
df = pd.DataFrame(dataset_limpio['train'])


# ==============================================================================
# --- 4. ANÁLISIS DE LONGITUD (EDA) Y GRÁFICO ---
# ==============================================================================
print("\n--- INICIANDO ANÁLISIS DE LONGITUD ---")

# Calculamos palabras basándonos en espacios
df['num_palabras'] = df['text_clean'].apply(lambda x: len(str(x).split()))

# Generamos el histograma
plt.figure(figsize=(10, 6))
plt.hist(df['num_palabras'], bins=50, color='skyblue', edgecolor='black')
# Línea roja marcando el límite de 1000 palabras
plt.axvline(1000, color='red', linestyle='dashed', linewidth=2, label='Límite 1000 palabras')
plt.title('Distribución de longitud de las noticias')
plt.xlabel('Número de palabras')
plt.ylabel('Cantidad de artículos')
plt.legend()

# Guardamos el gráfico en disco en lugar de mostrarlo (evita errores en terminales sin pantalla)
plt.savefig('grafico_longitud.png')
print("✅ Gráfico guardado correctamente como 'grafico_longitud.png'")


# ==============================================================================
# --- 5. RECORTE DE TEXTOS LARGOS (>1000 PALABRAS) ---
# ==============================================================================
print("\n--- APLICANDO RECORTE INTELIGENTE (Head + Tail) ---")

def recortar_texto(texto, limite=1000):
    """
    Si el texto supera el límite, conserva el principio y el final.
    Estrategia: 800 palabras iniciales + 200 finales.
    """
    palabras = texto.split()
    if len(palabras) <= limite:
        return texto
    
    # Definimos los puntos de corte
    n_inicio = 800
    n_final = 200
    
    parte_inicio = palabras[:n_inicio]
    parte_final = palabras[-n_final:]
    
    # Unimos con un marcador [...]
    return " ".join(parte_inicio) + " [...] " + " ".join(parte_final)

# Aplicamos la función a la columna limpia
df['text_clean'] = df['text_clean'].apply(lambda x: recortar_texto(str(x)))

# Comprobación de seguridad
max_len_nuevo = df['text_clean'].apply(lambda x: len(str(x).split())).max()
print(f"Nueva longitud máxima en el dataset: {max_len_nuevo} palabras (aprox)")


# ==============================================================================
# --- 6. PREPARACIÓN FINAL Y GUARDADO ---
# ==============================================================================
print("\n--- Preparando dataset final para IA... ---")

# 1. Creamos la entrada completa (Título + Cuerpo)
df['input_text'] = df['title'] + ". " + df['text_clean']

# 2. Convertimos la etiqueta booleana a numérica (1=Hiperpartidista, 0=Neutro)
df['label'] = df['hyperpartisan'].astype(int)

# 3. Seleccionamos solo las columnas necesarias para el modelo
df_final = df[['input_text', 'label']]

# 4. Guardamos en CSV
nombre_archivo = "dataset_procesado_final.csv"
df_final.to_csv(nombre_archivo, index=False)

print(f"¡Proceso completado! Archivo guardado en: '{nombre_archivo}'")
print(df_final.head())