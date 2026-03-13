import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib 
from sklearn.tree import export_graphviz
import pydotplus
import os

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


model = RandomForestClassifier(
    n_estimators=200,      # Mais árvores ajudam na estabilidade
    max_depth=20,          # Evita que a árvore cresça infinitamente (overfitting)
    min_samples_split=5,   # Exige mais amostras para criar um novo "galho"
    random_state=42
)

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

NOME_PASTA = 'portraits'
if not os.path.exists(NOME_PASTA):
    os.makedirs(NOME_PASTA)
    print(f"\nPasta '{NOME_PASTA}' criada com sucesso.")

# Gerar imagem da estrutura de árvore como exemplo
arvore_exemplo = model.estimators_[0]

print(f"Gerando imagem da árvore de decisão (isso pode demorar um pouco)...")


dot_data = export_graphviz(
    arvore_exemplo,
    out_file=None, 
    feature_names=X.columns, 
    class_names=le.classes_, 
    filled=True, 
    rounded=True, 
    special_characters=True,
    max_depth=7)

graph = pydotplus.graph_from_dot_data(dot_data)


contador = 0
while os.path.exists(os.path.join(NOME_PASTA, f'estrutura_arvore_libras_{contador}.png')):
    contador += 1

nome_arquivo = f'estrutura_arvore_libras_{contador}.png'
caminho_imagem = os.path.join(NOME_PASTA, nome_arquivo)

try:
    graph.write_png(caminho_imagem)
    print(f"Sucesso! Imagem numerada salva como: {nome_arquivo}")
    print(f"Caminho completo: {os.path.abspath(caminho_imagem)}")
except Exception as e:
    print(f"Erro ao salvar a imagem: {e}")
    print("Dica: Verifique se o Graphviz está no PATH do sistema.")