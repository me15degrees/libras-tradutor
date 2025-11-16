import cv2
import mediapipe as mp
import os
import csv

# --- Configurações Iniciais ---

# IMPORTANTE: Altere esta linha para o caminho da sua pasta principal
# (a pasta que contém as subpastas 'train' e 'test')
DATA_DIR = '/home/me15degrees/Programação/libras-tradutor/dataset'

OUTPUT_FILE = 'hand_landmarks.csv'

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Modo de imagem estática
    max_num_hands=1,         # Detectar apenas uma mão
    min_detection_confidence=0.5 # Confiança mínima de detecção
)

# --- Preparação do Arquivo CSV ---

# Criar o cabeçalho (header) para o CSV
# Teremos 'label' (A, B, C...) e 63 colunas de dados (x0, y0, z0, x1, y1, z1, ...)
header = ['label']
for i in range(21): # 21 landmarks
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

# Lista para armazenar todas as linhas de dados
data_list = [header]
print("Iniciando processamento...")

# --- Loop Principal para Ler as Pastas ---

# Itera sobre as pastas 'train' e 'test'
for data_type in ['train', 'test']:
    data_path = os.path.join(DATA_DIR, data_type)
    
    # Itera sobre cada pasta de letra (A, B, C...)
    labels = os.listdir(data_path)
    
    for label in labels:
        label_path = os.path.join(data_path, label)
        
        # Ignora arquivos que não são diretórios (ex: .DS_Store no Mac)
        if not os.path.isdir(label_path):
            continue
            
        print(f"Processando pasta: {label_path}")
        
        # Itera sobre cada imagem na pasta da letra
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            
            # Lê a imagem
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Aviso: Não foi possível ler a imagem {image_name}")
                continue
                
            # Converte a imagem de BGR (OpenCV) para RGB (MediaPipe)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Processa a imagem com o MediaPipe
            results = hands.process(image_rgb)
            
            # Verifica se uma mão foi detectada
            if results.multi_hand_landmarks:
                # Pega os landmarks da primeira mão detectada
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Prepara a linha de dados para o CSV
                row_data = [label] # A primeira coluna é a letra
                
                # Adiciona as coordenadas (x, y, z) de cada um dos 21 landmarks
                for landmark in hand_landmarks.landmark:
                    row_data.extend([landmark.x, landmark.y, landmark.z])
                    
                # Adiciona a linha completa à nossa lista de dados
                data_list.append(row_data)
            else:
                # Imprime um aviso se nenhuma mão for encontrada
                print(f"  Aviso: Nenhuma mão detectada em {image_name}")

# Libera o recurso do MediaPipe
hands.close()

# --- Salvar os Dados no Arquivo CSV ---

print("\nProcessamento concluído. Salvando dados em CSV...")

try:
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data_list)
    
    print(f"Sucesso! {len(data_list) - 1} amostras salvas em {OUTPUT_FILE}")
    print("Próximo passo: Fase 2 (Treinamento)")

except IOError as e:
    print(f"Erro ao salvar o arquivo CSV: {e}")