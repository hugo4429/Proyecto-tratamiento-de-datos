# Proyecto-tratamiento-de-datos
# Análisis de la polarización ideológica mediante detección automática de contenido hiperpartidista
### Proyecto Final – Tratamiento y Análisis de Datos  
### Universidad Carlos III de Madrid


##   1. Descripción del Proyecto

El objetivo de este proyecto es el diseño e implementación de un pipeline completo de clasificación automática de noticias hiperpartidistas, comparando distintas estrategias de representación del texto y modelos de aprendizaje automático con diferentes niveles de complejidad.

El trabajo se enmarca dentro del estudio de la polarización ideológica y la desinformación en medios digitales, abordando el problema desde una perspectiva de Procesamiento del Lenguaje Natural (PLN). A lo largo del proyecto se analizan las diferencias de rendimiento entre enfoques clásicos, redes neuronales y modelos Transformer preentrenados.

Concretamente, el proyecto implementa y compara:

- Tres representaciones del texto:
  - TF-IDF (representación clásica basada en frecuencias)
  - Embeddings simples con PyTorch
  - Embeddings contextuales con Transformers (BERT, RoBERTa)

- Tres tipos de modelos de clasificación:
  - Regresión logística (scikit-learn)
  - Red neuronal feed-forward implementada en PyTorch
  - Fine-tuning de DistilBERT utilizando Hugging Face Transformers

El dataset empleado es Hyperpartisan News Detection, disponible en Hugging Face, ampliamente utilizado en tareas de detección de sesgo ideológico.


## 2. Estructura del repositorio

```text
main.py                           Fase 1: Preprocesado, limpieza y EDA
fase2_tfidf.py                    Fase 2: TF-IDF + Regresión Logística
fase3_pytorch.py                  Fase 3: Red neuronal con embeddings propios
fase4_bert.py                     Fase 4: Fine-tuning de DistilBERT
dataset_procesado_final.csv       Dataset final limpio
grafico_longitud.png              Histograma de longitudes
grafico_entrenamiento_pytorch.png
requirements.txt                  Dependencias
README.md                         Memoria del proyecto
```

##   3. Instalación del entorno

Para garantizar la correcta ejecución del proyecto, se recomienda crear un entorno virtual e instalar las dependencias necesarias.

- Creación del entorno virtual
    - python -m venv env
- Activación del entorno
    - Windows: env\Scripts\activate
    - Linux / macOS: source env/bin/activate
- Instalación de dependencias
    - pip install -r requirements.txt

Las bibliotecas empleadas incluyen, entre otras:

 - pandas  
 - numpy  
 - scikit-learn  
 - matplotlib  
 - datasets (HuggingFace)  
 - transformers (HuggingFace)  
 - beautifulsoup4  
 - torch (PyTorch)


##    4. Fase 1 – Preprocesado y creación del dataset (main.py)

En esta fase se lleva a cabo la construcción del dataset final utilizado en las etapas de modelado. Se parte del conjunto Hyperpartisan News Detection en su configuración byarticle y se aplican técnicas sistemáticas de limpieza, normalización y análisis exploratorio.

- Objetivos:
    - Descargar el dataset desde HuggingFace (hyperpartisan_news_detection, configuración byarticle).
    - Unificar el título y el cuerpo del artículo en un único campo de entrada.
    - Limpiar el contenido textual eliminando etiquetas HTML y ruido no informativo.
    - Analizar la longitud de los textos como parte del análisis exploratorio (EDA).
    - Aplicar un recorte controlado en artículos excesivamente largos.
    - Transformar la etiqueta original en una variable binaria adecuada para clasificación.

- Preprocesado aplicado
    - Limpieza HTML mediante BeautifulSoup para extraer texto plano.
    - Normalización de espacios, saltos de línea y formato general.
    - Análisis de longitud de textos (número de palabras) para detectar valores atípicos.
    - Recorte de textos largos:
        - Estrategia Head + Tail: 800 palabras iniciales + 200 palabras finales.
        - Permite reducir el coste computacional manteniendo el contexto ideológico más relevante.
    - Conversión de etiquetas:
        - hyperpartisan → label ∈ {0,1}.
    - Eliminación preventiva de registros con texto vacío o nulo tras la limpieza.
    - Uso de semillas fijas (random_state = 42) para garantizar reproducibilidad.
    - Reducción del dataset a las columnas estrictamente necesarias para las fases posteriores.
- Análisis exploratorio (EDA)
    - Cálculo de la longitud de cada documento en palabras.
    - Generación y guardado del histograma de longitudes (grafico_longitud.png).

- Salida de la fase
    - dataset_procesado_final.csv – dataset limpio, normalizado y preparado para las fases de vectorización y modelado.
    - grafico_longitud.png – distribución de longitudes de los textos tras el preprocesado.


##    5. Fase 2 – Representación TF-IDF + Regresión Logística (fase2_tfidf.py)

En esta fase se construye un modelo clásico de machine learning basado en representaciones TF-IDF y una Regresión Logística, que actúa como baseline para comparar con modelos neuronales y transformadores.

- Objetivos
    - Generar una representación vectorial TF-IDF del campo input_text del dataset procesado.
    - Entrenar un modelo supervisado de Regresión Logística utilizando los vectores TF-IDF.
    - Evaluar el rendimiento del modelo mediante métricas estándar de clasificación.
    - Almacenar el vectorizador para reutilización posterior sin necesidad de reentrenarlo.
    - Detalles técnicos implementados
    - Carga de dataset_procesado_final.csv y eliminación preventiva de posibles valores nulos.
    - División estratificada en train (80%) y test (20%) para mantener la proporción de clases.
    - Uso de TfidfVectorizer con max_features=5000 y stop_words='english' para controlar dimensionalidad y ruido.
    - Ajuste del vectorizador únicamente sobre el conjunto de entrenamiento para evitar data leakage.
    - Entrenamiento de una Regresión Logística (max_iter=1000) adecuada para texto vectorizado.
    - Evaluación mediante:
        - Accuracy
        - Classification report (precision, recall, f1-score)
    - Guardado del vectorizador TF-IDF como tfidf_vectorizer.pkl.


- Resultados obtenidos
    - Accuracy del modelo sobre el conjunto de test.
    - Informe de clasificación completo, mostrando desempeño por clase (Neutro y Hiperpartidista).
    - Vectorizador guardado:
    - tfidf_vectorizer.pkl, útil para futuras predicciones o comparación con otros modelos.
- Este modelo constituye el baseline clásico del proyecto, sirviendo como referencia frente a métodos neuronales (PyTorch) y modelos Transformer (BERT).

##    6. Fase 3 – Embeddings simples + Red neuronal PyTorch (fase3_pytorch.py)
En esta fase se implementa un modelo neuronal ligero utilizando PyTorch. Se parte del texto limpio del dataset procesado para generar embeddings simples basados en un vocabulario limitado y entrenar una red neuronal capaz de clasificar noticias hiperpartidistas.

Objetivos
Construir un vocabulario a partir del conjunto de entrenamiento, seleccionando las 5000 palabras más frecuentes.
Transformar cada texto en una secuencia de índices enteros según el vocabulario generado.
Utilizar una arquitectura basada en:

Capa Embedding entrenable.

Average pooling para agregar información temporal.

Capas densas para la clasificación binaria.

Entrenar una red neuronal simple pero efectiva para la tarea.

Evaluar el modelo con métricas de rendimiento estándar.

Detalles técnicos implementados

Tokenización por división en palabras en minúsculas (.lower().split()).

Construcción del vocabulario mediante Counter() sobre el conjunto de entrenamiento.

Conversión de cada texto a una secuencia de longitud fija (max_len = 500) con padding.

Implementación del Dataset y DataLoader personalizados para PyTorch.

Red neuronal definida como:

Embedding → AveragePooling → Linear → ReLU → Linear → Sigmoid


Entrenamiento durante 20 épocas con optimizador Adam (lr=0.001) y función de pérdida BCELoss.

Gestión del caso especial donde PyTorch devuelve un escalar cuando el batch tiene tamaño 1 (corrección implementada).

Registro del training loss por época y guardado del gráfico correspondiente.

Arquitectura de la red

Capa de embeddings (nn.Embedding, dimensión 64, padding_idx=0).

Average pooling temporal (mean(dim=1)).

Capa oculta totalmente conectada (Linear(embed_dim, 32)) + ReLU.

Capa final (Linear(32, 1)) + Sigmoid para clasificación binaria.

Resultados obtenidos

Accuracy sobre el conjunto de test.

Classification report detallado con precisión, recall y F1-score para ambas clases.

grafico_entrenamiento_pytorch.png con la curva de aprendizaje (evolución del loss).
- *Objetivos*:
    - Construir un vocabulario (top 5000 palabras más frecuentes).
    - Convertir cada texto a una secuencia de índices.
    - Utilizar una capa Embedding + media temporal (average pooling).
    - Entrenar una red neuronal de clasificación.
    - Evaluar rendimiento.

- *Arquitectura*:
    - Capa de embeddings (nn.Embedding)
    - Average pooling
    - Capa oculta (ReLU)
    - Capa final sigmoide

- *Resultados*:
    - Accuracy en test.
    - classification_report
    - grafico_entrenamiento_pytorch.png

##    7. Fase 4 – DistilBERT Fine-Tuning (fase4_bert.py)
- *Objetivos*:
    - Tokenizar el texto con el tokenizador oficial de DistilBERT.
    - Ajustar (fine-tune) sus pesos para la clasificación binaria.
    - Evaluar modelo final.
- *Resultados*:
Accuracy final.

classification_report.

Carpeta resultados_bert/ con:

Checkpoints

Logs

Métricas de entrenamiento

Este es el modelo más potente del proyecto.

##    8. Comparación final de modelos
Representación	Modelo	Complejidad	Esperado
TF-IDF	Regresión logística	Baja	Baseline
Embeddings propios	PyTorch NN	Media	Mejor que TF-IDF
BERT contextual	DistilBERT	Alta	Mejor rendimiento global

##    9. Extensiones del proyecto (opcionales)
Para añadir valor al proyecto, se proponen extensiones posibles:

1. Explicabilidad del modelo (SHAP o LIME) ← Recomendada
Permite identificar qué palabras contribuyen más a predecir hiperpartidismo.

2. Optimización de hiperparámetros
GridSearch / Optuna para mejorar:

Dimensiones de embeddings

LR

Max length de BERT

Regularización en logística

3. Curvas ROC y PR para los tres modelos
Comparación visual muy potente.

4. Añadir análisis de polaridad (sentiment analysis)
5. Entrenar Word2Vec / FastText propio (más complejo)
##    10. Conclusión
Este proyecto implementa un pipeline completo de procesamiento y clasificación de texto, cubriendo desde métodos clásicos (TF-IDF) hasta modelos modernos basados en Transformers como DistilBERT.

La estructura modular permite extender o sustituir fácilmente cada fase del pipeline para futuras mejoras o experimentos.

✨ Autores
Equipo de estudiantes – UC3M, 2025