import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# ==============================================================================
# 1. CARGA Y PREPARACIÓN (Hugging Face Format)
# ==============================================================================
print("--- INICIANDO FINE-TUNING CON BERT (Hugging Face) ---")

# Cargar datos procesados
df = pd.read_csv("dataset_procesado_final.csv").dropna()

# Dividir igual que siempre (80/20)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

# Convertir a formato Dataset de Hugging Face
hg_train = Dataset.from_pandas(train_df)
hg_test = Dataset.from_pandas(test_df)

# ==============================================================================
# 2. TOKENIZACIÓN (Usando el tokenizador oficial de BERT)
# ==============================================================================
# Usamos 'distilbert' que es más rápido y ligero que BERT normal, ideal para pruebas
model_checkpoint = "distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    # Truncamos a 512 tokens (límite de BERT)
    return tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=256)

print("Tokenizando datos con BERT...")
tokenized_train = hg_train.map(preprocess_function, batched=True)
tokenized_test = hg_test.map(preprocess_function, batched=True)

# ==============================================================================
# 3. CONFIGURACIÓN DEL MODELO
# ==============================================================================
# Cargamos el modelo pre-entrenado y le decimos que tenemos 2 etiquetas (0 y 1)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./resultados_bert",
    learning_rate=2e-5,           # Tasa de aprendizaje muy baja (fino ajuste)
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,           # BERT aprende muy rápido, 3 épocas suelen bastar
    weight_decay=0.01,
    eval_strategy="epoch",  # Evaluar al final de cada época
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# ==============================================================================
# 4. ENTRENAMIENTO (FINE-TUNING)
# ==============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nEntrenando BERT (Esto puede tardar varios minutos)...")
trainer.train()

# ==============================================================================
# 5. EVALUACIÓN FINAL
# ==============================================================================
print("\nEvaluando BERT en el conjunto de test...")
preds_output = trainer.predict(tokenized_test)
y_pred = np.argmax(preds_output.predictions, axis=-1)
y_true = tokenized_test["label"]

print(f"\n--- RESULTADOS FINALES (BERT) ---")
print(f"ACCURACY: {accuracy_score(y_true, y_pred):.4f}")
print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(y_true, y_pred, target_names=['Neutro (0)', 'Hiperpartidista (1)']))