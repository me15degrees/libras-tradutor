import cv2
import mediapipe as mp
import os
import csv


DATA_DIR = '/home/me15degrees/Programação/libras-tradutor/dataset'

OUTPUT_FILE = 'hand_landmarks.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  
    max_num_hands=1,         
    min_detection_confidence=0.5 
)

header = ['label']
for i in range(21): # 21 landmarks
    header.extend([f'x{i}', f'y{i}', f'z{i}'])


data_list = [header]
print("Iniciando processamento...")


for data_type in ['train', 'test']:
    data_path = os.path.join(DATA_DIR, data_type)
    
    labels = os.listdir(data_path)
    
    for label in labels:
        label_path = os.path.join(data_path, label)
        
        if not os.path.isdir(label_path):
            continue
            
        print(f"Processando pasta: {label_path}")
        
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Aviso: Não foi possível ler a imagem {image_name}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]
                
                row_data = [label] 
                
                for landmark in hand_landmarks.landmark:
                    row_data.extend([landmark.x, landmark.y, landmark.z])

                data_list.append(row_data)
            else:

                print(f"  Aviso: Nenhuma mão detectada em {image_name}")

hands.close()


print("\nProcessamento concluído. Salvando dados em CSV...")

try:
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data_list)
    
    print(f"Sucesso! {len(data_list) - 1} amostras salvas em {OUTPUT_FILE}")
    print("Próximo passo: Fase 2 (Treinamento)")

except IOError as e:
    print(f"Erro ao salvar o arquivo CSV: {e}")