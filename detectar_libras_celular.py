import cv2
import mediapipe as mp
import joblib
import numpy as np
import sys

# --- 1. Configurações da Câmera ---

# Mude para False se quiser usar o IP do celular
USE_WEBCAM = False 

# Se USE_WEBCAM for False, mude o IP abaixo
URL_CELULAR = "http://192.168.100.99:8080/video" 

# --- 2. Carregar o Modelo e o LabelEncoder ---
try:
    model = joblib.load('libras_rf_model.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    print("Erro: Arquivos 'libras_rf_model.pkl' ou 'label_encoder.pkl' não encontrados.")
    print("Certifique-se de que os arquivos da Fase 2 estão no mesmo diretório.")
    sys.exit()

print("Modelo e LabelEncoder carregados com sucesso!")
print(f"O modelo foi treinado para reconhecer estas letras: {le.classes_}")

# --- 3. Inicializar Câmera ---
if USE_WEBCAM:
    cap = cv2.VideoCapture(0) # 0 para webcam padrão
    print("Usando Webcam (0)...")
else:
    cap = cv2.VideoCapture(URL_CELULAR)
    print(f"Tentando conectar ao celular: {URL_CELULAR}...")

if not cap.isOpened():
    print("Erro! Não foi possível abrir a câmera.")
    sys.exit()

# --- 4. Inicializar o MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,                 # Detectar apenas 1 mão
    min_detection_confidence=0.7,    # Confiança maior para detecção
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

print("Câmera iniciada. Pressione 'q' para sair.")

# --- 5. Loop Principal de Detecção e Predição ---
while True:
    # Ler o frame
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame. Verifique a conexão da câmera.")
        break

    # Inverter o frame (efeito espelho)
    frame = cv2.flip(frame, 1)

    # Converter para RGB (o MediaPipe espera RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem e encontrar mãos
    results = hands.process(frame_rgb)

    letra_predita = "" # Inicializa com string vazia

    # --- A MÁGICA ACONTECE AQUI ---
    if results.multi_hand_landmarks:
        # Itera sobre cada mão detectada (no nosso caso, apenas 1)
        for hand_landmarks in results.multi_hand_landmarks:
            
            # 1. Desenha os pontos na tela (para visualização)
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

            # 2. Coletar os 63 pontos (x, y, z * 21)
            # Exatamente como fizemos na Fase 1
            data_row = []
            for landmark in hand_landmarks.landmark:
                data_row.extend([landmark.x, landmark.y, landmark.z])

            # 3. Converter para o formato que o modelo espera
            # O Scikit-learn espera um "array 2D" (uma lista de amostras)
            landmarks_array = np.array([data_row]) 

            # 4. Fazer a Predição!
            prediction_numeric = model.predict(landmarks_array) # Ex: [0]
            
            # 5. Converter a predição numérica de volta para Letra
            # (usando o LabelEncoder que salvamos)
            letra_predita = le.inverse_transform(prediction_numeric)[0] # Ex: "A"
    
    # --- 6. Mostrar o Resultado na Tela ---
    
    # Desenha um retângulo de fundo para o texto ficar legível
    cv2.rectangle(frame, (10, 10), (160, 110), (0, 0, 0), -1) # Fundo preto
    
    # Escreve a letra prevista
    cv2.putText(
        frame, 
        letra_predita, 
        (30, 90), # Posição
        cv2.FONT_HERSHEY_SIMPLEX, 
        3, # Tamanho da fonte
        (255, 255, 255), # Cor (Branco)
        4, # Espessura
        cv2.LINE_AA
    )

    # Mostrar a janela de vídeo
    cv2.imshow('Detector de LIBRAS (Fase 3)', frame)
    
    # Sair do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Limpeza ---
print("\nEncerrando...")
cap.release()
cv2.destroyAllWindows()
hands.close()