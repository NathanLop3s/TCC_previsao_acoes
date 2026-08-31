from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "results"
BANCO_RESULTADOS = RAIZ / "resultados.db"

for pasta in (DADOS_BRUTOS, DADOS_PROCESSADOS, RESULTADOS):
    pasta.mkdir(parents=True, exist_ok=True)

TICKER = "ITUB4.SA"
PERIODO_HISTORICO = "5y"

JANELA_SMA_CURTA = 5
JANELA_SMA_LONGA = 20
JANELA_RSI = 14

JANELAS_TEMPORAIS = [5, 10, 20]

ARQUITETURAS_MLP = {
    "1_camada_32": (32,),
    "1_camada_64": (64,),
    "2_camadas_64_32": (64, 32),
    "3_camadas_64_32_16": (64, 32, 16),
}

PROPORCAO_TREINO = 0.8
SEED = 42