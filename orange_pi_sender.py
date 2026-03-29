"""
orange_pi_sender.py
Roda na Orange Pi Zero 2.
- Captura via IP Cam (ou webcam local)
- Processa com MediaPipe + modelo treinado
- Envia a predição via ZeroMQ PUSH para o PC

Instalar dependências na Orange Pi:
  pip install opencv-python-headless mediapipe joblib numpy pyzmq
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
# CONFIGURAÇÕES — edite conforme sua rede
# ─────────────────────────────────────────────
USE_WEBCAM       = False
URL_CELULAR      = "http://10.10.125.46:8080/video"

PC_IP            = "10.10.125.10"   # << IP do seu PC na rede
ZMQ_PORT         = 5555
ZMQ_TOPIC        = "libras"

LARGURA_PROC     = 320   # menor = mais rápido na Orange Pi
ALTURA_PROC      = 240
ROTACIONAR       = True  # False se não precisar rotacionar
SEND_INTERVAL_MS = 80    # envia no máximo a cada 80ms (~12 predições/s)
# ─────────────────────────────────────────────


def carregar_modelos():
    try:
        model = joblib.load('libras_rf_model.pkl')
        le    = joblib.load('label_encoder.pkl')
        print(f"[OK] Modelo carregado. Letras: {list(le.classes_)}")
        return model, le
    except FileNotFoundError as e:
        print(f"[ERRO] {e}")
        print("Coloque 'libras_rf_model.pkl' e 'label_encoder.pkl' no mesmo diretório.")
        sys.exit(1)


def abrir_camera():
    fonte = 0 if USE_WEBCAM else URL_CELULAR
    cap   = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera: {fonte}")
        sys.exit(1)
    # Sugere resolução baixa para economizar CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  LARGURA_PROC)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_PROC)
    print(f"[OK] Câmera aberta: {fonte}")
    return cap


def criar_socket_zmq():
    ctx    = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 5)          # descarta frames antigos se fila encher
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{PC_IP}:{ZMQ_PORT}")
    print(f"[OK] ZeroMQ PUSH conectado em tcp://{PC_IP}:{ZMQ_PORT}")
    return socket


def extrair_landmarks(hand_landmarks):
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array([data])


def main():
    model, le = carregar_modelos()
    cap        = abrir_camera()
    socket     = criar_socket_zmq()

    mp_hands   = mp.solutions.hands
    hands      = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    print("[RODANDO] Ctrl+C para parar.\n")

    ultimo_envio   = 0
    letra_anterior = ""
    frames_ok      = 0
    frames_erro    = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                frames_erro += 1
                if frames_erro % 30 == 0:
                    print(f"[AVISO] {frames_erro} frames perdidos. Verifique a câmera.")
                time.sleep(0.05)
                continue

            frames_ok += 1
            frame = cv2.resize(frame, (LARGURA_PROC, ALTURA_PROC))

            if ROTACIONAR:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(frame_rgb)

            letra_predita  = ""
            confianca_info = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks_array   = extrair_landmarks(hand_landmarks)
                    prediction_num    = model.predict(landmarks_array)
                    letra_predita     = le.inverse_transform(prediction_num)[0]

                    # Confiança (se o modelo suportar predict_proba)
                    try:
                        proba     = model.predict_proba(landmarks_array)[0]
                        confianca = round(float(np.max(proba)) * 100, 1)
                    except AttributeError:
                        confianca = -1.0

            # Envia apenas se passou o intervalo mínimo
            agora = time.time() * 1000
            if agora - ultimo_envio >= SEND_INTERVAL_MS:
                payload = json.dumps({
                    "letra":     letra_predita,
                    "confianca": confianca if letra_predita else 0.0,
                    "ts":        time.time(),
                })
                socket.send_string(payload, zmq.NOBLOCK)
                ultimo_envio = agora

                if letra_predita and letra_predita != letra_anterior:
                    print(f"  → Letra: {letra_predita}  ({confianca}%)")
                    letra_anterior = letra_predita

    except KeyboardInterrupt:
        print(f"\n[INFO] Encerrado. Frames lidos: {frames_ok} | Perdidos: {frames_erro}")
    finally:
        cap.release()
        hands.close()
        socket.close()
        print("[OK] Recursos liberados.")


if __name__ == "__main__":
    main()