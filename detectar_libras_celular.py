import cv2
import mediapipe as mp
import joblib
import numpy as np
import sys

# Mude para False se quiser usar o IP do celular
USE_WEBCAM = False 

# Se USE_WEBCAM for False, mude o IP abaixo
URL_CELULAR = "http://192.168.100.155:8080/video" 

try:
    model = joblib.load('libras_rf_model.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    print("Erro: Arquivos 'libras_rf_model.pkl' ou 'label_encoder.pkl' não encontrados.")
    print("Certifique-se de que os arquivos da Fase 2 estão no mesmo diretório.")
    sys.exit()

print("Modelo e LabelEncoder carregados com sucesso!")
print(f"O modelo foi treinado para reconhecer estas letras: {le.classes_}")

if USE_WEBCAM:
    cap = cv2.VideoCapture(0) 
    print("Usando Webcam (0)...")
else:
    cap = cv2.VideoCapture(URL_CELULAR)
    print(f"Tentando conectar ao celular: {URL_CELULAR}...")

if not cap.isOpened():
    print("Erro! Não foi possível abrir a câmera.")
    sys.exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,                 
    min_detection_confidence=0.7,    
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

print("Câmera iniciada. Pressione 'q' para sair.")

LARGURA_PROC = 480  
ALTURA_PROC = 360

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (LARGURA_PROC, ALTURA_PROC))

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #rotação

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    letra_predita = "" 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenhar apenas se necessário (consome CPU)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data_row = []
            for landmark in hand_landmarks.landmark:
                data_row.extend([landmark.x, landmark.y, landmark.z])

            landmarks_array = np.array([data_row]) 
            prediction_numeric = model.predict(landmarks_array) 
            letra_predita = le.inverse_transform(prediction_numeric)[0] 
    
    # Interface gráfica simplificada
    cv2.putText(frame, f"Letra: {letra_predita}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Orange Pi LIBRAS', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\nEncerrando...")
cap.release()
cv2.destroyAllWindows()
hands.close()