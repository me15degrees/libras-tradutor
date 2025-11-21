import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib 


DATA_FILE = 'hand_landmarks.csv'

try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Erro: Arquivo '{DATA_FILE}' não encontrado.")
    print("Certifique-se de que o script da Fase 1 foi executado e o arquivo CSV foi criado.")
    exit()

print(f"Dados carregados! Total de {len(data)} amostras.")

data = data.dropna()

#
X = data.drop('label', axis=1) 
y = data['label']             

le = LabelEncoder()
y_encoded = le.fit_transform(y)

joblib.dump(le, 'label_encoder.pkl')
print(f"Classes encontradas: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded 
)

print(f"Usando {len(X_train)} amostras para treino e {len(X_test)} para teste.")

print("Treinando o modelo (Random Forest)...")


model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(X_train, y_train)

print("Treinamento concluído!")

print("Avaliando o modelo...")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do Modelo: {accuracy * 100:.2f}%")


y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

print("\nRelatório de Classificação:")
print(classification_report(y_test_labels, y_pred_labels, zero_division=0))


MODEL_FILE = 'libras_rf_model.pkl'
joblib.dump(model, MODEL_FILE)

print(f"\nModelo salvo com sucesso em '{MODEL_FILE}'")
print("Próximo passo: Fase 3 (Predição em Tempo Real)")