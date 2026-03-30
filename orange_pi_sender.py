"""
orange_pi_sender.py
Roda na Orange Pi Zero 2.
- Captura via IP Cam
- Processa com MediaPipe + modelo treinado
- Envia APENAS a predição (letra + confiança) via ZeroMQ
- O PC recebe o vídeo diretamente da IP Cam
"""
 
import cv2
import mediapipe as mp
import joblib
import numpy as np
import zmq
import sys
import time
import json
 
# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
USE_WEBCAM       = False
URL_CELULAR      = "http://192.168.100.218:8080/video"
 
PC_IP            = "192.168.100.220"
ZMQ_PORT         = 5555
 
LARGURA_PROC     = 320
ALTURA_PROC      = 240
ROTACIONAR       = True
SEND_INTERVAL_MS = 0     # envia toda predição imediatamente, sem espera
# ─────────────────────────────────────────────
 
 
def carregar_modelos():
    try:
        model = joblib.load('libras_rf_model.pkl')
        le    = joblib.load('label_encoder.pkl')
        print(f"[OK] Modelo carregado. Letras: {list(le.classes_)}")
        return model, le
    except FileNotFoundError as e:
        print(f"[ERRO] {e}")
        sys.exit(1)
 
 
def abrir_camera():
    fonte = 0 if USE_WEBCAM else URL_CELULAR
    cap   = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera: {fonte}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  LARGURA_PROC)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_PROC)
    print(f"[OK] Câmera aberta: {fonte}")
    return cap
 
 
def criar_socket_zmq():
    ctx    = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)          # descarta imediatamente se não enviou
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
    socket.setsockopt(zmq.IMMEDIATE, 1)
    socket.connect(f"tcp://{PC_IP}:{ZMQ_PORT}")
    print(f"[OK] ZeroMQ PUSH → tcp://{PC_IP}:{ZMQ_PORT}")
    return socket
 
 
def extrair_landmarks(hand_landmarks):
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array([data])
 
 
def main():
    model, le  = carregar_modelos()
    cap         = abrir_camera()
    socket      = criar_socket_zmq()
 
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,              # modelo leve (0=lite, 1=full) — muito mais rápido
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
 
    frames_erro   = 0
 
    print("[RODANDO] Ctrl+C para parar.\n")
 
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                frames_erro += 1
                if frames_erro % 30 == 0:
                    print(f"[AVISO] {frames_erro} frames perdidos.")
                time.sleep(0.05)
                continue
 
            frame = cv2.resize(frame, (LARGURA_PROC, ALTURA_PROC))
 
            if ROTACIONAR:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(frame_rgb)
 
            letra_predita = ""
            confianca     = 0.0
            landmarks_list = []
 
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Coleta landmarks para enviar ao PC
                    landmarks_list = [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in hand_landmarks.landmark
                    ]
 
                    landmarks_array = extrair_landmarks(hand_landmarks)
                    prediction_num  = model.predict(landmarks_array)
                    letra_predita   = le.inverse_transform(prediction_num)[0]
 
                    try:
                        proba     = model.predict_proba(landmarks_array)[0]
                        confianca = round(float(np.max(proba)) * 100, 1)
                    except AttributeError:
                        confianca = -1.0
 
            # Envia imediatamente toda predição
            payload = json.dumps({
                "letra":     letra_predita,
                "confianca": confianca,
                "ts":        time.time(),
                "landmarks": landmarks_list,   # coordenadas normalizadas (0.0~1.0)
            })
            try:
                socket.send_string(payload, zmq.NOBLOCK)
            except zmq.Again:
                pass  # descarta se não conseguiu enviar — não acumula
 
            if letra_predita:
                print(f"  → {letra_predita} ({confianca}%)")
 
    except KeyboardInterrupt:
        print("\n[INFO] Encerrado.")
    finally:
        cap.release()
        hands.close()
        socket.close()
        print("[OK] Recursos liberados.")
 
 
if __name__ == "__main__":
    main()