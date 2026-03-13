import os
import random

# --- CONFIGURAÇÕES ---
BASE_DATASET = "dataset"
LETRAS = ['A', 'B', 'C', 'D', 'E', 'I', 'L', 'M', 'N', 'O', 'R', 'S', 'U', 'V', 'W']
PERCENTUAL_MIGRAR = 0.20  # 20% do que está no treino vai para o teste

print(f"📦 Iniciando migração de {PERCENTUAL_MIGRAR*100}% dos dados de TRAIN para TEST...")

for letra in LETRAS:
    caminho_treino = os.path.join(BASE_DATASET, 'train', letra)
    caminho_teste = os.path.join(BASE_DATASET, 'test', letra)

    # Verifica se a pasta de treino existe
    if not os.path.exists(caminho_treino):
        print(f"⚠️  Pasta de treino para '{letra}' não encontrada. Pulando...")
        continue

    # Garante que a pasta de teste existe
    os.makedirs(caminho_teste, exist_ok=True)

    # Lista arquivos de imagem no treino
    arquivos = [f for f in os.listdir(caminho_treino) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(arquivos) == 0:
        print(f"❓ Letra {letra}: Nenhuma imagem encontrada no treino.")
        continue

    # Sorteia quais arquivos serão movidos
    quantidade_para_mover = int(len(arquivos) * PERCENTUAL_MIGRAR)
    # Garante que mova ao menos 1 se houver arquivos
    quantidade_para_mover = max(1, quantidade_para_mover) if len(arquivos) > 0 else 0
    
    arquivos_para_mover = random.sample(arquivos, quantidade_para_mover)

    # Move os arquivos
    for img in arquivos_para_mover:
        origem = os.path.join(caminho_treino, img)
        destino = os.path.join(caminho_teste, img)
        os.replace(origem, destino)

    print(f"✅ Letra {letra}: {quantidade_para_mover} imagens migradas para TEST.")

print("\n--- Migração concluída! ---")