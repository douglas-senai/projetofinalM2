import cv2
import numpy as np
import io
from ultralytics import YOLO
from app.config import MODEL_NAME, CLASSES_DE_INTERESSE

# Carrega o modelo uma única vez na memória
model = YOLO(MODEL_NAME)

def obter_imagem_cv2(imagem_bytes: bytes):
    """Função auxiliar para converter bytes na imagem que o OpenCV entende."""
    nparr = np.frombuffer(imagem_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem inválida ou corrompida.")
    return img

def processar_e_desenhar_imagem(imagem_bytes: bytes) -> io.BytesIO:
    """Roda o YOLO, desenha na imagem e retorna os bytes do JPEG final."""
    img = obter_imagem_cv2(imagem_bytes)
    results = model(img)
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            if cls_id in CLASSES_DE_INTERESSE:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf)
                
                cor = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), cor, 3)
                
                texto = f"{CLASSES_DE_INTERESSE[cls_id]} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), cor, -1)
                cv2.putText(img, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    _, img_encoded = cv2.imencode('.jpg', img)
    return io.BytesIO(img_encoded.tobytes())

def extrair_estatisticas_deteccao(imagem_bytes: bytes) -> dict:
    """Roda o YOLO e retorna apenas um dicionário com contagem e coordenadas (x1, y1, x2, y2)."""
    img = obter_imagem_cv2(imagem_bytes)
    results = model(img)
    
    deteccoes = []
    contagem_por_classe = {nome: 0 for nome in CLASSES_DE_INTERESSE.values()}
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            if cls_id in CLASSES_DE_INTERESSE:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf)
                nome_classe = CLASSES_DE_INTERESSE[cls_id]
                
                # Incrementa o contador da classe específica detectada
                contagem_por_classe[nome_classe] += 1
                
                # Salva os dados analíticos
                deteccoes.append({
                    "objeto": nome_classe,
                    "confianca": round(conf, 2),
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })
                
    return {
        "total_detectado": len(deteccoes),
        "resumo_contagem": contagem_por_classe,
        "deteccoes": deteccoes
    }
