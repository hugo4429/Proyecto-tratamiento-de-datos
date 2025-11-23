import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURACIÓN Y PREPARACIÓN DE DATOS
# ==============================================================================
print("--- INICIANDO RED NEURONAL PYTORCH ---")

# Cargar datos
df = pd.read_csv("dataset_procesado_final.csv").dropna()

# Dividir Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    df['input_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# --- TOKENIZACIÓN Y VOCABULARIO ---
print("Creando vocabulario...")
def tokenizar(texto):
    return str(texto).lower().split()

# Contar palabras
contador = Counter()
for texto in X_train:
    contador.update(tokenizar(texto))

# Crear vocabulario
vocab_size = 5000
palabras_comunes = contador.most_common(vocab_size)
vocab = {palabra: i+1 for i, (palabra, _) in enumerate(palabras_comunes)}

def texto_a_indices(texto, max_len=500):
    tokens = tokenizar(texto)
    indices = [vocab.get(t, 0) for t in tokens]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

# ==============================================================================
# 2. CLASE DATASET
# ==============================================================================
class NewsDataset(Dataset):
    def __init__(self, textos, etiquetas):
        self.textos = textos.reset_index(drop=True)
        self.etiquetas = etiquetas.reset_index(drop=True)
        
    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, idx):
        texto_indices = texto_a_indices(self.textos[idx])
        label = self.etiquetas[idx]
        return torch.tensor(texto_indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

# Batch size 32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==============================================================================
# 3. DEFINICIÓN DE LA RED NEURONAL
# ==============================================================================
class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=32):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        out = embedded.mean(dim=1) # Average pooling
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

modelo = SimpleNN(vocab_size=vocab_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

# ==============================================================================
# 4. ENTRENAMIENTO
# ==============================================================================
epochs = 20
historial_loss = []

print(f"\nEntrenando por {epochs} épocas...")

for epoch in range(epochs):
    modelo.train()
    total_loss = 0
    for textos, etiquetas in train_loader:
        optimizer.zero_grad()
        outputs = modelo(textos).squeeze()
        loss = criterion(outputs, etiquetas)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    historial_loss.append(avg_loss)
    if (epoch+1) % 5 == 0:
        print(f"Época {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Guardar gráfico
plt.plot(historial_loss)
plt.title('Curva de Aprendizaje (Loss)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.savefig('grafico_entrenamiento_pytorch.png')
print("✅ Gráfico actualizado.")

# ==============================================================================
# 5. EVALUACIÓN (CORREGIDA)
# ==============================================================================
print("\nEvaluando modelo...")
modelo.eval()
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for textos, etiquetas in test_loader:
        outputs = modelo(textos).squeeze()
        predicted = (outputs > 0.5).float()
        
        # --- CORRECCIÓN DE BUG ---
        # Si el batch tiene tamaño 1, 'predicted' es un escalar (0 dimensiones)
        if predicted.ndim == 0:
            y_pred_list.append(predicted.item())
            y_true_list.append(etiquetas.item())
        else:
            # Si es un batch normal, lo convertimos a lista
            y_pred_list.extend(predicted.tolist())
            y_true_list.extend(etiquetas.tolist())

acc = accuracy_score(y_true_list, y_pred_list)
print(f"\n--- RESULTADOS DE LA RED NEURONAL ---")
print(f"ACCURACY: {acc:.4f}")
print("\nREPORTE DE CLASIFICACIÓN:")
# output_dict=False asegura que imprima el texto formateado
print(classification_report(y_true_list, y_pred_list, target_names=['Neutro (0)', 'Hiperpartidista (1)']))