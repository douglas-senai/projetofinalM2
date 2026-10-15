# 🧠 Smart Detection API — Estatísticas e Visão

Este documento apresenta a especificação técnica oficial e a arquitetura de inteligência artificial. O sistema foi projetado para expor endpoints de alto desempenho capazes de realizar análises analíticas e renderização visual baseadas em visão computacional.

## 📋 1. Descrição do Dataset

O modelo foi treinado e validado a partir de um subconjunto customizado extraído de bases públicas globais de larga escala, como o COCO Dataset (Common Objects in Context). A amostragem foi filtrada e rotulada especificamente para solucionar o desafio de classificar e rastrear elementos biológicos, domésticos e de infraestrutura urbana em um único pipeline de dados.
O dataset mapeia 10 classes específicas, estruturadas sob a seguinte indexação:

• 0: pessoa

• 2: carro

• 14: passaro

• 15: gato

• 16: cahorro

• 17: cavalo

• 21: urso

• 46: banana

• 47: maça

• 49: laranja

**Desafio que resolve:** O ecossistema automatiza a triagem volumétrica de itens e seres vivos em fluxos de mídia, resolvendo problemas práticos como auditoria de estoques alimentares (maça, banana, laranja), controle de segurança e tráfego (pessoa, carro), classificação de animais domésticos (gato, cahorro) e sensoriamento/monitoramento ecológico de fauna (passaro, urso, cavalo).

## 🧠 2. Descrição da Inteligência Artificial (IA)

A arquitetura de processamento neural utiliza o framework de aprendizado profundo YOLOv8 (You Only Look Once), desenvolvido pela Ultralytics.

**• Versão e Arquitetura:** YOLOv8n (Variação Nano).

**• Justificativa Técnica:** O modelo do tipo Nano possui uma arquitetura otimizada composta por cerca de 3.2 milhões de parâmetros. Essa configuração foi selecionada de forma estratégica para viabilizar o deploy do ecossistema de visão computacional em servidores baseados puramente em CPU e com recursos computacionais restritos, tais como o plano gratuito da plataforma Render (limite de 512MB de memória RAM). Desse modo, garante-se uma latência de processamento de poucos milissegundos sem riscos de interrupções por estouro de memória (Out Of Memory).

**• Mecanismo de Otimização:** O arquivo de pesos (.pt) é instanciado estaticamente na inicialização do microsserviço. Com isso, os pesos sinápticos permanecem em cache no processo principal, mitigando o gargalo de leitura de disco a cada nova requisição HTTP.

## 💻 3. Instruções de Execução

**Preparação do Ambiente**

Clone este repositório
```bash
git clone https://github.com/douglas-senai/projetofinalM2.git
```

Certifique-se de possuir o Python instalado em seu ambiente (versão 3.8 ou superior). Acesse a pasta raiz de execução da API (detector_yolo_api) e instale as dependências listadas no gerenciador de pacotes:

Instalação dos pacotes nucleares (FastAPI, OpenCV-Headless, NumPy e Ultralytics) - certifique-se de estar na pasta detector_yolo_api

pip install -r requirements.txt

**Execução em Ambiente Local (Desenvolvimento)**

Para iniciar a API em sua máquina local habilitando o recarregamento em tempo real (hot-reload) a cada modificação nos arquivos de código, utilize o Uvicorn:

uvicorn app.main:app --reload

• Endereço Local da API: http://127.0.0.1:8000

• Documentação Swagger Interativa: http://127.0.0

**Execução em Ambiente de Produção (Deploy no Render)**

Link direto do serviço hospedado: https://projetofinalm2-2026.onrender.com/docs

Nota: Por ser um serviço free do render e ter recursos muito limitados o primeiro acesso e post pode demorar alguns minutos.

## 📦 4. Estrutura do Payload de Saída (JSON)

Ao realizar uma requisição para o endpoint analítico da API (POST /detectar-estatisticas), enviando um arquivo de imagem através de formulário multipart, a API processa as matrizes e devolve uma estrutura estrita e tipada.
Exemplo de Payload de Retorno (JSON):

```json
{
  "total_detectado": 3,
  "resumo_contagem": {
    "maça": 0,
    "banana": 2,
    "urso": 0,
    "passaro": 0,
    "carro": 0,
    "gato": 0,
    "cahorro": 1,
    "cavalo": 0,
    "pessoa": 0,
    "laranja": 0
  },
  "deteccoes": [
    {
      "objeto": "cahorro",
      "confianca": 0.95,
      "bounding_box": {
        "x1": 85,
        "y1": 140,
        "x2": 390,
        "y2": 520
      }
    },
    {
      "objeto": "banana",
      "confianca": 0.88,
      "bounding_box": {
        "x1": 420,
        "y1": 90,
        "x2": 510,
        "y2": 180
      }
    },
    {
      "objeto": "banana",
      "confianca": 0.84,
      "bounding_box": {
        "x1": 490,
        "y1": 110,
        "x2": 570,
        "y2": 195
      }
    }
  ]
}
```
