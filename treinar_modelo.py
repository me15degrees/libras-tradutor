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
