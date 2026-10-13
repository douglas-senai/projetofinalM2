from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from app.detector import processar_e_desenhar_imagem, extrair_estatisticas_deteccao

app = FastAPI(title="Smart Detection API - Estatísticas e Visão")

@app.get("/")
def home():
    return {"status": "API Online", "documentacao": "/docs"}

# Rota 1: Retorna a imagem modificada na tela
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        imagem_final_bytes = processar_e_desenhar_imagem(contents)
        return StreamingResponse(imagem_final_bytes, media_type="image/jpeg")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erro interno: {str(e)}"})

# Rota 2: Nova rota que retorna apenas dados numéricos e as Bounding Boxes (JSON)
@app.post("/predict/stats")
async def predict_stats(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        dados_json = extrair_estatisticas_deteccao(contents)
        return JSONResponse(content=dados_json)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erro interno: {str(e)}"})
