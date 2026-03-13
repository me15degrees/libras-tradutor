import cv2
import mediapipe as mp
import os

# --- CONFIGURAÇÕES ---
VIDEO_DIR = "videos_dataset"
BASE_DATASET = "dataset"
FPS_TARGET = 5       
split_ratio = 0.8  # 80% para treino, 20% para teste

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Criar estrutura básica
for folder in ['train', 'test']:
    os.makedirs(os.path.join(BASE_DATASET, folder), exist_ok=True)

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

for video_name in video_files:
    letra = os.path.splitext(video_name)[0].upper()
    video_path = os.path.join(VIDEO_DIR, video_name)
    
    # Criar subpastas da letra em ambos
    train_path = os.path.join(BASE_DATASET, 'train', letra)
    test_path = os.path.join(BASE_DATASET, 'test', letra)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    intervalo = max(1, int(video_fps / FPS_TARGET))
    
    frames_validos = []
    frame_idx = 0

    print(f"Lendo vídeo da Letra {letra}...")

    # Primeiro passo: Coletar todos os frames onde a mão aparece
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % intervalo == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                frames_validos.append(frame.copy())
        
        frame_idx += 1
    cap.release()

    # Segundo passo: Dividir e Salvar
    num_train = int(len(frames_validos) * split_ratio)
    
    for i, f in enumerate(frames_validos):
        if i < num_train:
            cv2.imwrite(os.path.join(train_path, f"{letra}_tr_{i}.jpg"), f)
        else:
            cv2.imwrite(os.path.join(test_path, f"{letra}_ts_{i}.jpg"), f)

    print(f"   -> {letra}: {num_train} imagens p/ Treino | {len(frames_validos)-num_train} p/ Teste.")

hands.close()
print("\nDataset pronto! Agora seu 'coletar_dados.py' encontrará as pastas certas.")