<div align="center">
  
[![Em Progresso](https://img.shields.io/badge/Status-Em%20Progresso-yellow.svg)](https://github.com/me15degreesm/interface-calculadora-rendimento)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
  
</div>

# Projeto de Identificação do Alfabeto em LIBRAS
 
Este é um projeto desenvolvido inicialmente para a matéria de Extensão na Faculdade de Engenharia Elétrica (FEELT/UFU). Através de aprendizado de máquina e visão computacional, o sistema identifica letras estáticas do alfabeto em LIBRAS (Língua Brasileira de Sinais). O objetivo central foi construir um pipeline capaz de capturar imagens de gestos em tempo real, extrair características anatômicas das mãos e classificar corretamente as letras do alfabeto manual.
 
> ⚠️ **OBSERVAÇÃO:** Letras que exigem movimento (como **Ç, H, J, K**) possuem dinâmicas que não são captadas por este algoritmo, que foca no processamento de poses estáticas em frames de vídeo.
 
<div align="center">
    <img src="img/image.png"/>
</div>
 
 
## 📱 Funcionamento e Arquitetura (Distribuída)

O projeto utiliza uma topologia de rede onde o processamento pesado de IA é delegado à **Orange Pi Zero 2**, enquanto o PC foca na renderização de alta qualidade e interface de usuário (UI).

### Fluxo de Dados:

1.  **Smartphone (IP Cam):** Captura o vídeo e atua como servidor MJPEG via Wi-Fi (App recomendado: *IP Webcam*).
2.  **Orange Pi Zero 2 (Processador H616):**
    * Consome o stream da câmera em baixa resolução ($320 \times 240$) para preservar a CPU.
    * Extrai os 21 landmarks da mão via **MediaPipe** (modelo `lite`).
    * Classifica a letra usando um modelo **Random Forest** treinado.
    * Envia a predição via **ZeroMQ (Protocolo TCP - PUSH/PULL)** para o PC.
3.  **PC (Interface e Exibição):**
    * Consome o vídeo original da câmera em alta definição (full quality).
    * Recebe as predições (letra, confiança e latência) em uma thread dedicada.
    * Renderiza uma interface profissional com histórico de tradução e barra de confiança.

```text
[ Celular (IP Cam) ] 
       │
       ├─── Stream MJPEG (HTTP) ───► [ Orange Pi Zero 2 ] (Inferência de IA)
       │                                     │
       │                                     ▼
       │                          [ ZeroMQ: JSON via TCP ] (Porta 5555)
       │                                     │
       └─── Stream MJPEG (HTTP) ──────────► [ PC ] (Interface e UI Fullscreen)

```
---
 
 ## 🗂️ Estrutura do Repositório

- `orange_pi_sender.py`: Script de inferência que roda na Orange Pi. Otimizado para o processador H616, envia apenas dados estruturados (JSON) para evitar sobrecarga de rede.

- `pc_receiver.py`: Interface principal que roda no PC. Gerencia múltiplas threads para garantir que o vídeo rode de forma fluida (60 FPS) independente da velocidade de resposta da IA.

- `libras_rf_model.pkl`: Modelo Random Forest serializado.

- `coletar_dados.py` & `treinar_modelo.py`: Ferramentas para captura de novos sinais e retreinamento do classificador.
> OBS: O sistema requer uma estrutura de diretórios onde a pasta dataset contenha subpastas nomeadas conforme a letra correspondente (ex: dataset/A), contendo os frames para treinamento.

## 🛠️ Tecnologias e Otimizações

- **ZeroMQ (ZMQ)**: O padrão Push/Pull com HWM=1 (High Water Mark) garante que o PC sempre processe a predição mais recente.

- **Multithreading**: O uso de threading.Lock() no Python evita conflitos de memória entre a recepção de vídeo e a atualização dos dados da IA.

- **Performance**: O MediaPipe na Orange Pi está configurado com model_complexity=0 (lite).

## 🚀 Como Executar
1. Configuração de Rede

No topo dos arquivos .py, ajuste as variáveis de IP:

- URL_CELULAR: O link HTTP gerado pelo app no smartphone.

- PC_IP: O endereço IP local do seu computador.

2. Instalação de dependências
```bash

# No PC (Windows/Linux/Mac)
pip install opencv-python pyzmq numpy

# Na Orange Pi (Armbian/Debian)
pip install opencv-python-headless pyzmq mediapipe joblib numpy
```
3. Ordem de Inicialização

     1) Inicie o servidor de vídeo no Celular (IP CAM).

     2) No PC, execute: $python pc_receiver.py.

     3) Na Orange Pi, execute: $python orange_pi_sender.py.

Pronto, agora é só enquadrar o celular para transmitir os gestos e visualizá-los na janela que abrir no computador.


<div align="center">
  
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/maria-eduarda-nascimento-andrade-bb0b86213/)
  [![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=flat&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCh6sgz1ij_my64lX8rQnPXg)
  
</div>
