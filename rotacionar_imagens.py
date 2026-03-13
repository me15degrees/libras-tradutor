import cv2
import os

# Caminho raiz do seu dataset
BASE_DIR = 'dataset'

print("Iniciando a rotação das imagens...")

# Percorre train e test
for sub_pasta in ['train', 'test']:
    caminho_sub = os.path.join(BASE_DIR, sub_pasta)
    
    if not os.path.exists(caminho_sub):
        continue

    # Percorre cada pasta de letra (A, B, C...)
    for letra in os.listdir(caminho_sub):
        caminho_letra = os.path.join(caminho_sub, letra)
        
        if os.path.isdir(caminho_letra):
            print(f"Rotacionando imagens da letra: {letra} em {sub_pasta}")
            
            for arquivo in os.listdir(caminho_letra):
                if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                    caminho_img = os.path.join(caminho_letra, arquivo)
                    
                    # Carrega a imagem
                    img = cv2.imread(caminho_img)
                    
                    if img is not None:
                        # Rotaciona 90 graus para a esquerda (anti-horário)
                        # ROTATE_90_COUNTERCLOCKWISE = 90 graus esquerda
                        img_rotacionada = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        
                        # Salva por cima da original
                        cv2.imwrite(caminho_img, img_rotacionada)
                    else:
                        print(f"Erro ao ler: {arquivo}")

print("\nSucesso! Todas as imagens foram rotacionadas.")