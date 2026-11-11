import sys
from pathlib import Path

import pandas as pd

import config
from avaliacao import plotar_comparacao_configuracoes
from coleta import coletar_dados
from estatistica import analisar
from experimentos import rodar_grade_experimentos


def obter_dados_brutos(ticker: str, forcar_nova_coleta: bool = False) -> pd.DataFrame:
    caminho_arquivo = config.DADOS_BRUTOS / f"{ticker.replace('.', '_')}.parquet"

    if caminho_arquivo.exists() and not forcar_nova_coleta:
        print(f"Usando dados já coletados em {caminho_arquivo}")
        return pd.read_parquet(caminho_arquivo)

    return coletar_dados(ticker, config.DATA_INICIO, config.DATA_FIM, config.DADOS_BRUTOS)


def main() -> None:
    resumos, execucoes, baselines = [], [], []

    for ticker in config.TICKERS:
        print(f"\n=== Projeto MLP -- predição de preços ({ticker}) ===\n")
        df_bruto = obter_dados_brutos(ticker)
        print(f"{len(df_bruto)} registros carregados "
              f"({df_bruto.index.min().date()} a {df_bruto.index.max().date()}).\n")

        resumo, exec_ind, baseline = rodar_grade_experimentos(
            df_bruto=df_bruto,
            ticker=ticker,
            janelas_temporais=config.JANELAS_TEMPORAIS,
            arquiteturas=config.ARQUITETURAS_MLP,
            janela_sma_curta=config.JANELA_SMA_CURTA,
            janela_sma_longa=config.JANELA_SMA_LONGA,
            janela_rsi=config.JANELA_RSI,
            proporcao_treino=config.PROPORCAO_TREINO,
            seeds=config.SEEDS,
            ddof_desvio=config.DDOF_DESVIO,
            caminho_banco=config.BANCO_RESULTADOS,
            pasta_resultados=config.RESULTADOS,
            pasta_previsoes=config.PREVISOES,
        )
        plotar_comparacao_configuracoes(
            resumo, ticker,
            config.RESULTADOS / f"{ticker.replace('.', '_')}_comparacao_rmse.png",
        )
        resumos.append(resumo)
        execucoes.append(exec_ind)
        baselines.append(baseline)

    df_resumo = pd.concat(resumos, ignore_index=True)
    df_exec = pd.concat(execucoes, ignore_index=True)
    df_base = pd.concat(baselines, ignore_index=True)

    df_resumo.to_csv(config.RESULTADOS / "resultados_experimentos.csv", index=False)
    df_exec.to_csv(config.RESULTADOS / "execucoes_individuais.csv", index=False)
    df_base.to_csv(config.RESULTADOS / "baseline_persistencia.csv", index=False)

    colunas = [
        "janela_temporal", "arquitetura", "rmse_medio", "rmse_desvio", "rmse_mediana",
        "mae_medio", "mape_medio", "mase_medio", "r2_medio", "acuracia_direcional_medio",
    ]
    for ticker in config.TICKERS:
        r = df_resumo[df_resumo["ticker"] == ticker].sort_values("rmse_medio")
        print(f"\n=== {ticker}: resumo (ordenado por RMSE médio) ===")
        print(r[colunas].to_string(index=False))
        print(f"\n--- {ticker}: baseline de persistência ---")
        print(df_base[df_base["ticker"] == ticker].drop(columns=["ticker"]).to_string(index=False))
        print()
        print(analisar(df_exec, ticker, config.RESULTADOS))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
