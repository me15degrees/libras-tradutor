"""
pc_receiver.py
Roda no seu PC.
- Captura o vídeo DIRETAMENTE da IP Cam (full quality)
- Recebe a predição da Orange Pi via ZeroMQ
- Exibe o vídeo com overlay da letra + confiança + histórico

Instalar:
  pip install opencv-python pyzmq numpy
"""

import cv2
import zmq
import numpy as np
import json
import time
import collections
import threading

URL_CELULAR       = "http://192.168.100.218:8080/video"
ROTACIONAR        = True

ZMQ_PORT          = 5555
JANELA_TITULO     = "LIBRAS - Tradutor em Tempo Real"

SMOOTHING_WINDOW  = 1     # sem acumulação — exibe imediatamente
SEM_SINAL_TIMEOUT = 3.0

# Resolução alvo da janela (ajuste para o seu monitor)
TELA_W            = 1920
TELA_H            = 1080
HEADER_H          = 60

# Largura do vídeo — o painel ocupa o restante
CAM_W             = 720   # vídeo ocupa 720px, painel ocupa 1200px
CAM_H             = TELA_H - HEADER_H
PAINEL_W          = TELA_W - CAM_W

# Cores BGR
C_BG      = (28,  28,  32)
C_PAINEL  = (38,  38,  48)
C_HEADER  = (22,  22,  28)
C_VERDE   = (80,  215, 110)
C_AMARELO = (50,  200, 255)
C_VERMELHO= (70,  70,  220)
C_BRANCO  = (240, 240, 240)
C_CINZA   = (130, 130, 145)
C_LINHA   = (60,  60,  75)

estado = {
    "letra":     "",
    "confianca": 0.0,
    "latencia":  0.0,
    "ultimo_rx": time.time(),
    "historico": [],
    "landmarks": [],
}
estado_lock = threading.Lock()


def thread_zmq():
    """Roda em background recebendo predições da Orange Pi."""
    historico_local = collections.deque(maxlen=50)

    ctx    = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 2)    # descarta acúmulo — só o mais recente
    socket.setsockopt(zmq.RCVTIMEO, 50)
    socket.bind(f"tcp://0.0.0.0:{ZMQ_PORT}")
    print(f"[OK] ZeroMQ PULL escutando na porta {ZMQ_PORT}")

    letra_anterior = ""

    while True:
        try:
            # Drena todas as mensagens enfileiradas e fica só com a mais recente
            msg = None
            while True:
                try:
                    msg = socket.recv_string(zmq.NOBLOCK)
                except zmq.Again:
                    break

            if msg is None:
                # Nenhuma mensagem nova — espera bloqueante curta
                try:
                    msg = socket.recv_string()
                except zmq.Again:
                    continue

            data = json.loads(msg)

            letra_raw   = data.get("letra", "")
            confianca   = data.get("confianca", 0.0)
            latencia_ms = (time.time() - data.get("ts", time.time())) * 1000

            # Exibe imediatamente sem votar
            if letra_raw and letra_raw != letra_anterior:
                historico_local.append(letra_raw)
                letra_anterior = letra_raw
                print(f"  {letra_raw}  conf:{confianca:.0f}%  lat:{latencia_ms:.1f}ms")

            with estado_lock:
                estado["letra"]     = letra_raw
                estado["confianca"] = confianca
                estado["latencia"]  = latencia_ms
                estado["ultimo_rx"] = time.time()
                estado["historico"] = list(historico_local)
                estado["landmarks"] = data.get("landmarks", [])

        except zmq.Again:
            pass
        except Exception as e:
            print(f"[ZMQ ERRO] {e}")


def cor_confianca(v):
    if v >= 70: return C_VERDE
    if v >= 40: return C_AMARELO
    return C_VERMELHO


def barra(img, x, y, w, h, valor, cor):
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 55, 68), -1)
    p = int(w * min(valor / 100.0, 1.0))
    if p > 0:
        cv2.rectangle(img, (x, y), (x + p, y + h), cor, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_LINHA, 1)


# Conexões do MediaPipe Hands (pares de índices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def montar_janela(frame_cam, letra, confianca, latencia, historico, sem_sinal, landmarks):
    canvas = np.zeros((TELA_H, TELA_W, 3), dtype=np.uint8)
    canvas[:] = C_BG

    cv2.rectangle(canvas, (0, 0), (TELA_W, HEADER_H), C_HEADER, -1)
    cv2.line(canvas, (0, HEADER_H), (TELA_W, HEADER_H), C_LINHA, 2)

    cv2.putText(canvas, "LIBRAS  Tradutor", (24, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, C_VERDE, 2, cv2.LINE_AA)

    status = "Aguardando Orange Pi..." if sem_sinal else f"Lat: {latencia:.0f} ms"
    s_cor  = C_AMARELO if sem_sinal else C_VERDE
    cv2.putText(canvas, status, (TELA_W - 360, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, s_cor, 1, cv2.LINE_AA)
    cam_resized = cv2.resize(frame_cam, (CAM_W, CAM_H))
    
    canvas[HEADER_H:HEADER_H + CAM_H, 0:CAM_W] = cam_resized
    px = CAM_W
    py = HEADER_H
    cv2.rectangle(canvas, (px, py), (TELA_W, TELA_H), C_PAINEL, -1)
    cv2.line(canvas, (px, py), (px, TELA_H), C_LINHA, 2)

    m = 50   # margem interna do painel
    y = py + 80

    cv2.putText(canvas, "CONFIANCA", (px + m, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_CINZA, 2, cv2.LINE_AA)
    y += 50
    barra_w = PAINEL_W - m * 2
    barra(canvas, px + m, y, barra_w, 30,
          confianca if (not sem_sinal and letra) else 0,
          cor_confianca(confianca))
    y += 55
    val = f"{confianca:.0f}%" if (not sem_sinal and letra) else "--"
    cv2.putText(canvas, val, (px + m, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                cor_confianca(confianca) if letra else C_CINZA, 2, cv2.LINE_AA)

    y += 55
    cv2.line(canvas, (px + m, y), (TELA_W - m, y), C_LINHA, 2)
    y += 60

    cv2.putText(canvas, "LETRA", (px + m, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, C_CINZA, 2, cv2.LINE_AA)
    y += 30

    if letra and not sem_sinal:
        lc = cor_confianca(confianca)
        # Sombra
        cv2.putText(canvas, letra, (px + m + 3, y + 103),
                    cv2.FONT_HERSHEY_DUPLEX, 6.0, (15, 15, 20), 12, cv2.LINE_AA)
        # Letra principal
        cv2.putText(canvas, letra, (px + m, y + 100),
                    cv2.FONT_HERSHEY_DUPLEX, 6.0, lc, 8, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "?", (px + m + 20, y + 100),
                    cv2.FONT_HERSHEY_DUPLEX, 6.0, C_CINZA, 6, cv2.LINE_AA)
    y += 130

 
    cv2.line(canvas, (px + m, y), (TELA_W - m, y), C_LINHA, 2)
    y += 55

    cv2.putText(canvas, "HISTORICO", (px + m, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_CINZA, 2, cv2.LINE_AA)
    y += 55

    # Divide em linhas de 10 letras
    chunks = [historico[i:i+10] for i in range(0, len(historico), 10)]
    if not chunks:
        chunks = [["-"]]
    for chunk in chunks[-4:]:   # exibe até 4 linhas
        linha = "  ".join(chunk)
        cv2.putText(canvas, linha, (px + m, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, C_BRANCO, 2, cv2.LINE_AA)
        y += 55


    cv2.putText(canvas, "[Q] Sair    [C] Limpar historico",
                (px + m, TELA_H - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_CINZA, 1, cv2.LINE_AA)

    return canvas

frame_lock   = threading.Lock()
frame_atual  = {"img": None}
camera_viva  = True

def thread_camera():
    global camera_viva
    cap = cv2.VideoCapture(URL_CELULAR)
    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir a câmera.")
        camera_viva = False
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[OK] Câmera conectada: {URL_CELULAR}")

    while camera_viva:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        if ROTACIONAR:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)   # espelha horizontalmente (efeito espelho)
        with frame_lock:
            frame_atual["img"] = frame

    cap.release()


def main():
    # Thread ZMQ (predições da Orange Pi)
    t_zmq = threading.Thread(target=thread_zmq, daemon=True)
    t_zmq.start()

    # Thread câmera (frame sempre atualizado, sem buffer acumulado)
    t_cam = threading.Thread(target=thread_camera, daemon=True)
    t_cam.start()

    estado["historico"] = []

    cv2.namedWindow(JANELA_TITULO, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JANELA_TITULO, TELA_W, TELA_H)
    cv2.setWindowProperty(JANELA_TITULO, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("[RODANDO] Q para sair, C para limpar histórico.\n")

    frame_espera = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cv2.putText(frame_espera, "Aguardando camera...", (30, CAM_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 90), 1, cv2.LINE_AA)

    while True:
        with frame_lock:
            frame = frame_atual["img"].copy() if frame_atual["img"] is not None else frame_espera.copy()

        with estado_lock:
            letra     = estado["letra"]
            confianca = estado["confianca"]
            latencia  = estado["latencia"]
            historico = estado.get("historico", [])
            landmarks = estado.get("landmarks", [])
            ultimo_rx = estado["ultimo_rx"]

        sem_sinal = (time.time() - ultimo_rx) > SEM_SINAL_TIMEOUT

        janela = montar_janela(frame, letra, confianca, latencia, historico, sem_sinal, landmarks)
        cv2.imshow(JANELA_TITULO, janela)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('c'):
            with estado_lock:
                estado["historico"] = []
                estado["letra"]     = ""
            print("[INFO] Histórico limpo.")

    global camera_viva
    camera_viva = False
    cv2.destroyAllWindows()
    print("[OK] Encerrado.")


if __name__ == "__main__":
    main()