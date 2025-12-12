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
Modelo pesado                     Fase 2: Bert
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


##  5. Fase 2 – Representación TF-IDF + Regresión Logística (fase2_tfidf.py)

En esta fase se utiliza TF-IDF (Term Frequency – Inverse Document Frequency) como primera estrategia de representación vectorial del texto. Este enfoque transforma cada noticia en un vector numérico de dimensión fija, donde cada componente refleja la importancia de un término en función de su frecuencia local y su capacidad discriminativa en el corpus completo.

TF-IDF se emplea como baseline para comparar posteriormente su rendimiento frente a representaciones neuronales y embeddings contextuales.

### 5.1 Configuración de la vectorización TF-IDF

La vectorización del texto se realiza con los siguientes parámetros:

- Número máximo de características (max_features): 3000
  Se limita el vocabulario a los 3000 términos más relevantes para reducir dimensionalidad y ruido.
- Eliminación de stopwords: inglés
  Se eliminan palabras funcionales sin carga semántica relevante.

- Frecuencia mínima de documento (min_df): 10
  Se descartan términos que aparecen en menos de 10 documentos, evitando términos demasiado raros.

El vectorizador se ajusta exclusivamente sobre el conjunto de entrenamiento y posteriormente se aplica al conjunto de validación y test, evitando filtrado de información entre conjuntos (data leakage).

### 5.2 Modelos evaluados con TF-IDF
Sobre la representación TF-IDF se evalúan dos enfoques de clasificación distintos:

A. Regresión Logística (Scikit-learn)
Se utiliza un modelo de Regresión Logística como clasificador lineal de referencia, con la siguiente configuración:

- Número máximo de iteraciones: 1000
- Ajuste de pesos de clase: balanceado
- Semilla aleatoria: 42

El modelo produce probabilidades asociadas a la clase positiva (hiperpartidista), lo que permite una evaluación más rica que una predicción binaria directa.

Este enfoque constituye el baseline clásico del proyecto.

B. Red neuronal feed-forward (PyTorch)

Como alternativa al clasificador lineal, se entrena una red neuronal feed-forward implementada en PyTorch, utilizando los vectores TF-IDF como entrada.

Características principales del modelo:

- Dimensión de entrada: 3000 (correspondiente a TF-IDF)

- Arquitectura:

    - Capa densa de 128 neuronas
    - Capa densa de 64 neuronas
    - Capa de salida con activación sigmoide

- Función de pérdida: Binary Cross-Entropy
- Optimizador: Adam
- Número de épocas: 15
- Dropout: 0.3 para reducir overfitting

Este modelo permite analizar si una arquitectura no lineal es capaz de extraer patrones adicionales a partir de una representación TF-IDF clásica.

### 5.3 Evaluación y métricas
Ambos modelos se evalúan sobre el conjunto de test utilizando exactamente las mismas métricas:

- **Accuracy**, como medida global de rendimiento.

- **ROC-AUC**, para evaluar la capacidad discriminativa del modelo de forma independiente del umbral de decisión.

Además, para cada modelo se generan y almacenan las siguientes visualizaciones:

- **Matriz de confusión** (Neutro vs. Hiperpartidista).

- **Curva ROC** con el valor del área bajo la curva (AUC).

Los resultados obtenidos se almacenan para su comparación directa con las fases posteriores del proyecto, donde se emplean representaciones neuronales y modelos Transformer.

#### 5.3.1 Resultados TF-IDF + Regresión Logística (Scikit-learn)

![Matriz de confusión con TF-IDF Scikit-learn](images/conf_matrix_TFIDF_Sklearn.png)
![Matriz de confusión con TF-IDF PyTorch](images/conf_matrix_TFIDF_PyTorch.png)

#### 5.3.2 Resultados TF-IDF + Red neuronal (PyTorch)
#### 5.3.3 Comparación entre ambos enfoques
,s

gráficas
curvas rock
matriz de confusión

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