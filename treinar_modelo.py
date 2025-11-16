import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib # Para salvar o modelo

# --- 1. Carregar os Dados ---
DATA_FILE = 'hand_landmarks.csv'

try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_FILE}' não encontrado.")
    print("Certifique-se de que o script da Fase 1 foi executado e o arquivo CSV foi criado.")
    exit()

print(f"Dados carregados! Total de {len(data)} amostras.")

# Remove linhas com dados ausentes (se houver)
data = data.dropna()

# --- 2. Preparar os Dados ---

# Separar 'features' (X) e 'labels' (y)
X = data.drop('label', axis=1) # Todas as colunas, exceto 'label'
y = data['label']              # Apenas a coluna 'label'

# Converter labels (letras 'A', 'B', 'C'...) em números (0, 1, 2...)
# O modelo de IA só entende números
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Salvar o LabelEncoder para usarmos na Fase 3 (para converter números de volta em letras)
joblib.dump(le, 'label_encoder.pkl')
print(f"Classes encontradas: {le.classes_}")

# --- 3. Dividir em Treino e Teste ---
# Usamos 20% dos dados para teste e 80% para treino
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, # Para resultados reprodutíveis
    stratify=y_encoded # Garante que a proporção de letras seja a mesma no treino e teste
)

print(f"Usando {len(X_train)} amostras para treino e {len(X_test)} para teste.")

# --- 4. Treinar o Modelo ---
print("Treinando o modelo (Random Forest)...")

# Inicializa o classificador
# n_estimators = número de "árvores" na floresta
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treina o modelo
model.fit(X_train, y_train)

print("Treinamento concluído!")

# --- 5. Avaliar o Modelo ---
print("Avaliando o modelo...")

# Faz predições nos dados de teste
y_pred = model.predict(X_test)

# Calcula a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do Modelo: {accuracy * 100:.2f}%")

# Mostra um relatório detalhado (precisão, recall, f1-score por letra)
# Precisamos converter os números de volta para letras para o relatório ficar legível
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

print("\nRelatório de Classificação:")
print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

# --- 6. Salvar o Modelo Treinado ---
MODEL_FILE = 'libras_rf_model.pkl'
joblib.dump(model, MODEL_FILE)

print(f"\nModelo salvo com sucesso em '{MODEL_FILE}'")
print("Próximo passo: Fase 3 (Predição em Tempo Real)")