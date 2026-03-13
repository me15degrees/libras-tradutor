import cv2
import mediapipe as mp
import joblib
import numpy as np
import sys
import warnings

# Silenciar avisos de versão do scikit-learn para limpar o terminal
warnings.filterwarnings("ignore", category=UserWarning)

# Mude para False se quiser usar o IP do celular
USE_WEBCAM = False 

# Se USE_WEBCAM for False, mude o IP abaixo
URL_CELULAR = "http://192.168.100.182:8080/video" 

try:
    model = joblib.load('libras_rf_model.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    print("Erro: Arquivos '.pkl' não encontrados no diretório atual.")
    sys.exit()

print("--- Tradutor de Libras Ativo ---")
print(f"Letras mapeadas: {le.classes_}")

if USE_WEBCAM:
    cap = cv2.VideoCapture(0) 
    print("Usando Webcam USB...")
else:
    cap = cv2.VideoCapture(URL_CELULAR)
    print(f"Conectando ao celular: {URL_CELULAR}...")

if not cap.isOpened():
    print("Erro! Verifique a conexão com a câmera.")
    sys.exit()

# Configuração do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,                 
    min_detection_confidence=0.7,    
    min_tracking_confidence=0.5
)

print("Iniciando detecção... Pressione CTRL+C no terminal para sair.")

# Reduzir resolução para economizar CPU no Raspberry Pi
LARGURA_PROC = 320  
ALTURA_PROC = 240

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nFalha ao receber frame da câmera.")
            break

        # Redimensionamento e Conversão
        frame = cv2.resize(frame, (LARGURA_PROC, ALTURA_PROC))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)

        status_msg = "Aguardando mão..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_row = []
                for landmark in hand_landmarks.landmark:
                    data_row.extend([landmark.x, landmark.y, landmark.z])

                # Predição
                landmarks_array = np.array([data_row]) 
                prediction_numeric = model.predict(landmarks_array) 
                letra_predita = le.inverse_transform(prediction_numeric)[0]
                status_msg = f"Letra Detectada: {letra_predita}   "

        sys.stdout.write(f"\r{status_msg}")
        sys.stdout.flush()

except KeyboardInterrupt:
    print("\nInterrompido pelo usuário.")

finally:
    print("Encerrando recursos...")
    cap.release()
    hands.close()
    print("Pronto.")