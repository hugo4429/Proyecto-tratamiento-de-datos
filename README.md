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
  - Word2vec (Embeddings simples con PyTorch)
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

- Número máximo de características (max_features): **3000**

  Se limita el vocabulario a los términos más relevantes para reducir dimensionalidad y ruido.
- Eliminación de stopwords: **inglés**

  Se eliminan palabras funcionales sin carga semántica relevante.

- Frecuencia mínima de documento (min_df): **10**

  Se descartan términos que aparecen en menos de 10 documentos, evitando términos demasiado raros.

El vectorizador se ajusta exclusivamente sobre el conjunto de entrenamiento y posteriormente se aplica al conjunto de validación y test, evitando filtrado de información entre conjuntos (data leakage).

### 5.2 Modelos evaluados con TF-IDF
Sobre la representación TF-IDF se evalúan dos enfoques de clasificación distintos:

***A. Regresión Logística (Scikit-learn)***

Se utiliza un modelo de Regresión Logística como clasificador lineal de referencia, con la siguiente configuración:

- Número máximo de iteraciones: 1000
- Ajuste de pesos de clase: balanceado
- Semilla aleatoria: 42

El modelo produce probabilidades asociadas a la clase positiva (hiperpartidista), lo que permite una evaluación más rica que una predicción binaria directa.

Este enfoque constituye el baseline clásico del proyecto.

***B. Red neuronal feed-forward (PyTorch)***

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
<p align="center">
  <img src="images/conf_matrix_TFIDF_Sklearn.png" width="300" />
  <img src="images/roc_TFIDF_Sklearn.png" width="300" />
</p>

<!--![Matriz de confusión con TF-IDF Scikit-learn](images/conf_matrix_TFIDF_Sklearn.png)
![Curvas ROC](images/roc_TFIDF_Sklearn.png)-->


#### 5.3.2.1 Resultados TF-IDF + Red neuronal (PyTorch)
<p align="center">
  <img src="images/conf_matrix_TFIDF_PyTorch.png" width="300" />
  <img src="images/roc_TFIDF_PyTorch.png" width="300" />
</p>
<!--![Matriz de confusión con TF-IDF PyTorch](images/conf_matrix_TFIDF_PyTorch.png)
![alt text](images/roc_TFIDF_PyTorch.png)-->

#### 5.3.2.2 Resultados TF-IDF + Red neuronal (PyTorch) con Early Stopping
<p align="center">
  <img src="images/conf_matrix_TFIDF_PyTorch_E_S.png" width="300" />
  <img src="images/roc_TFIDF_PyTorch_E_S.png" width="300" />
</p>
<!--![Matriz de confusión con TF-IDF Early-stop](images/conf_matrix_TFIDF_Google_PyTorch_E_S.png)
![Curvas ROC ](images/roc_TFIDF_Sklearn_E_S.png)-->

Durante el entrenamiento del modelo TF-IDF + PyTorch con un máximo de 100 épocas, se observa la siguiente evolución:

- En las primeras épocas, tanto la pérdida de entrenamiento como la pérdida de validación disminuyen de forma progresiva, lo que indica que el modelo está aprendiendo patrones relevantes a partir de los datos.

- A partir de aproximadamente la época 40, la pérdida de entrenamiento continúa disminuyendo de forma significativa, mientras que la pérdida de validación empieza a estabilizarse, mostrando una mejora cada vez más marginal.

- En la época 48, el mecanismo de early stopping detecta que la pérdida de validación deja de mejorar de manera consistente y detiene el entrenamiento de forma automática.

- El modelo recupera los pesos correspondientes a la mejor época en validación, garantizando así el mejor compromiso entre ajuste y generalización.

Este comportamiento confirma la presencia de un incipiente sobreajuste a partir de las últimas épocas, donde el modelo sigue optimizando el conjunto de entrenamiento sin lograr mejoras equivalentes en validación.


## 6. Fase 3 – Word2Vec preentrenado (Google News) y clasificación

En esta fase se utiliza una segunda estrategia de representación vectorial basada en Word2Vec, empleando embeddings preentrenados sobre Google News. El objetivo es pasar de una representación dispersa basada en frecuencias (TF-IDF) a una representación densa y semántica, donde palabras con significados similares tienden a ocupar posiciones cercanas en el espacio vectorial.

### 6.1 Carga del modelo Word2Vec (Google News)

Se carga el modelo word2vec-google-news-300, un modelo preentrenado de gran tamaño (≈ 1.6 GB) que proporciona vectores de 300 dimensiones para palabras del vocabulario. Al tratarse de un modelo preentrenado, no se ajustan los embeddings durante el proyecto: se reutilizan directamente como fuente de información semántica.

### 6.2 Vectorización de documentos a partir de embeddings de palabras

Como Word2Vec produce vectores por palabra, se requiere convertir cada noticia completa en un único vector de tamaño fijo. Para ello, se implementa una representación a nivel de documento basada en el promedio (mean pooling) de los embeddings de sus palabras:

- El texto se tokeniza con nltk.word_tokenize tras pasarlo a minúsculas.

- Se filtran los tokens, quedándose solo con las palabras que están presentes en el vocabulario del modelo.

- Se calcula la media de los embeddings de las palabras válidas.

Caso especial implementado: si un documento no contiene ninguna palabra presente en el modelo, se asigna un vector de ceros de dimensión 300. Esto garantiza que todos los documentos tienen representación válida y comparable.

El resultado es una matriz de características densa para cada partición:

- X_train_w2v, X_val_w2v, X_test_w2v con forma (n_documentos, 300).

### 6.3 Modelos evaluados con Word2Vec

Para mantener la comparabilidad experimental con TF-IDF, se evalúan dos clasificadores diferentes sobre la misma representación Word2Vec.

**A. Word2Vec + Regresión Logística (Scikit-learn)**

Se entrena un clasificador lineal de Regresión Logística sobre los vectores Word2Vec de 300 dimensiones, con la siguiente configuración:

- max_iter = 1000
- class_weight = "balanced" (para compensar posibles desbalances de clase)
- random_state = 42 (reproducibilidad)

El modelo genera probabilidades para la clase hiperpartidista y se evalúa con las mismas métricas del pipeline: Accuracy y ROC-AUC, además de guardar matriz de confusión y curva ROC.

**B. Word2Vec + Red neuronal (PyTorch) con Early Stopping**

Como alternativa no lineal, se entrena una red neuronal feed-forward en PyTorch usando los embeddings Word2Vec como entrada:

- Dimensión de entrada (input_dim): 300
- Entrenamiento con un máximo de 200 épocas
- Early Stopping activado con patience = 5, utilizando explícitamente el conjunto de validación para detener el entrenamiento si la pérdida de validación deja de mejorar y recuperar automáticamente el mejor modelo.

Este enfoque permite comprobar si, sobre una representación semántica densa como Word2Vec, un clasificador no lineal mejora el rendimiento frente al clasificador lineal.

### 6.4 Visualización de la arquitectura (diagrama de red)

Adicionalmente, se genera un diagrama de la arquitectura de la red neuronal utilizada con Word2Vec mediante torchviz y graphviz. Para ello se crea una instancia de la red con entrada de 300 dimensiones y se propaga un input ficticio (dummy input).

![alt text](images/diagrama_arquitectura_w2v.png)

El diagrama generado muestra la arquitectura interna de la red neuronal utilizada con Word2Vec, así como el flujo de operaciones que PyTorch emplea durante el entrenamiento. La red recibe como entrada un vector de 300 dimensiones, correspondiente al embedding Word2Vec de cada documento, y lo procesa a través de dos capas ocultas. La primera capa transforma la entrada de 300 a 128 neuronas, y la segunda reduce la representación de 128 a 64 neuronas, aplicando en ambos casos una función de activación ReLU para introducir no linealidad. Finalmente, una capa de salida de 1 neurona con activación sigmoide produce un único valor entre 0 y 1, que se interpreta como la probabilidad de que la noticia sea hiperpartidista.

Además de las capas y activaciones, el diagrama refleja cómo PyTorch organiza internamente los cálculos necesarios para el aprendizaje. Los bloques asociados a los pesos y sesgos de cada capa indican los parámetros entrenables del modelo, mientras que los nodos intermedios representan las operaciones matemáticas que permiten calcular los gradientes durante la retropropagación del error. Aunque el grafo puede parecer complejo, su función principal es documentar que el modelo sigue una estructura 300 → 128 → 64 → 1, coherente con la arquitectura definida, y que el entrenamiento se realiza correctamente mediante backpropagation.

### 6.5 Evaluación, artefactos y almacenamiento de resultados

Para ambas variantes (Scikit-learn y PyTorch) se generan automáticamente:

- Matriz de confusión
- Curva ROC

Los resultados se almacenan en una estructura común para facilitar la comparación con TF-IDF y con las fases posteriores del proyecto.
#### 6.5.1 Resultados Word2Vec + Regresión Logística (Scikit-learn)
![alt text](images/conf_matrix_W2V_Google_Sklearn.png)
![alt text](images/roc_W2V_Google_Sklearn.png)
#### 6.5.2 Resultados Word2Vec + Red neuronal (PyTorch)
![alt text](images/conf_matrix_W2V_Google_PyTorch.png)
![alt text](images/roc_W2V_Google_PyTorch.png)
![alt text](images/conf_matrix_W2V_Google_PyTorch_E_S.png)
![alt text](images/roc_W2V_Google_PyTorch_E_S.png)

## 7. Fase 4 – Embeddings contextuales con BERT (DistilBERT)

En esta fase se emplea una representación del texto basada en embeddings contextuales obtenidos mediante un modelo Transformer preentrenado. Concretamente, se utiliza DistilBERT, una versión más ligera de BERT que mantiene gran parte de su capacidad representacional con un menor coste computacional.

El objetivo de esta fase es evaluar si una representación contextual, capaz de tener en cuenta el significado de las palabras en función de su contexto, mejora la detección de noticias hiperpartidistas frente a representaciones estáticas como TF-IDF o Word2Vec.

### 7.1 Modelo utilizado: DistilBERT

Se emplea el modelo distilbert-base-uncased, preentrenado sobre grandes corpus de texto en inglés mediante tareas de modelado del lenguaje. Sus principales características son:

- Tipo de modelo: Transformer encoder
- Uso de atención para capturar dependencias contextuales
- Dimensión del embedding: 768
- Texto en minúsculas (uncased)
- Modelo preentrenado, sin ajuste adicional de sus pesos (no fine-tuning)

En esta fase, DistilBERT se utiliza exclusivamente como extractor de características, no como clasificador end-to-end.

### 7.2 Extracción de embeddings a nivel de documento

Cada texto se procesa individualmente mediante el tokenizador de DistilBERT, que convierte el texto en tokens compatibles con el modelo. Para controlar el coste computacional y asegurar una longitud uniforme, se aplica:

- Truncado a un máximo de 256 tokens
- Padding automático
- Procesamiento en modo inferencia (no_grad), sin cálculo de gradientes

A partir de la salida del modelo, se extrae el embedding correspondiente al token [CLS], que actúa como una representación global del documento. Este vector tiene 768 dimensiones y se utiliza como representación final del texto.

El resultado es una matriz de embeddings densos para cada partición del dataset:

- X_train_bert, X_val_bert, X_test_bert con dimensión (n_documentos, 768).

### 7.3 Modelos evaluados con embeddings BERT

Al igual que en las fases anteriores, se evalúan dos enfoques de clasificación sobre los embeddings extraídos con BERT, manteniendo la coherencia experimental.

**A. BERT Embeddings + Regresión Logística (Scikit-learn)**

En primer lugar, se entrena un modelo de Regresión Logística utilizando como entrada los embeddings BERT de 768 dimensiones.

Configuración principal del clasificador:

- Número máximo de iteraciones: 1000
- Pesos de clase balanceados
- Semilla fija para reproducibilidad

Este enfoque permite evaluar hasta qué punto la información contextual capturada por BERT es separable mediante un clasificador lineal.

**B. BERT Embeddings + Red neuronal (PyTorch) con Early Stopping**

Como alternativa no lineal, se entrena una red neuronal feed-forward en PyTorch utilizando los embeddings BERT como entrada.

Características del entrenamiento:

- Dimensión de entrada: 768
- Número máximo de épocas: 100
- Early stopping activado con patience = 5

Uso explícito del conjunto de validación para detener el entrenamiento y recuperar el mejor modelo

Este enfoque permite analizar si una arquitectura neuronal sencilla es capaz de explotar mejor la riqueza semántica de los embeddings contextuales.

### 7.4 Evaluación y objetivo comparativo

Ambos modelos se evalúan sobre el conjunto de test utilizando las mismas métricas que en el resto del proyecto:

- Accuracy
- ROC-AUC

Además, se generan automáticamente:

- Matrices de confusión
- Curvas ROC

Los resultados obtenidos en esta fase permiten comparar de forma directa TF-IDF, Word2Vec y BERT embeddings, analizando el impacto de pasar de representaciones basadas en frecuencias, a embeddings estáticos, y finalmente a embeddings contextuales, manteniendo constante el esquema de evaluación y los clasificadores empleados.

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