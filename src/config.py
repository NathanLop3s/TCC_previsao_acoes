from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "results"
PREVISOES = RESULTADOS / "previsoes"
# Banco novo: o resultados.db antigo (rodada de 17/09/2026) fica preservado.
BANCO_RESULTADOS = RAIZ / "resultados_v2.db"

for pasta in (DADOS_BRUTOS, DADOS_PROCESSADOS, RESULTADOS, PREVISOES):
    pasta.mkdir(parents=True, exist_ok=True)

# Ativos estudados (conforme o texto do TCC: ITUB4 e PETR4)
TICKERS = ["ITUB4.SA", "PETR4.SA"]

# Período histórico FIXO de cinco anos (antes era period="5y", que muda a cada execução).
# Atenção: no yfinance a data final é exclusiva, por isso 2026-01-01 inclui 31/12/2025.
DATA_INICIO = "2021-01-01"
DATA_FIM = "2026-01-01"

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
SEEDS = [42, 7, 123, 2024, 99]

# Desvio padrão entre as execuções: 1 = amostral (n-1), 0 = populacional (n).
DDOF_DESVIO = 1
