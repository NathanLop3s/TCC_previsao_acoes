import sys
from pathlib import Path

import pandas as pd

import config
from avaliacao import plotar_comparacao_configuracoes
from coleta import coletar_dados
from experimentos import rodar_grade_experimentos


def obter_dados_brutos(forcar_nova_coleta: bool = False) -> pd.DataFrame:
    caminho_arquivo = config.DADOS_BRUTOS / f"{config.TICKER.replace('.', '_')}.parquet"

    if caminho_arquivo.exists() and not forcar_nova_coleta:
        print(f"Usando dados já coletados em {caminho_arquivo}")
        return pd.read_parquet(caminho_arquivo)

    return coletar_dados(config.TICKER, config.PERIODO_HISTORICO, config.DADOS_BRUTOS)


def main() -> None:
    print(f"=== Projeto MLP -- predição de preços ({config.TICKER}) ===\n")

    df_bruto = obter_dados_brutos()
    print(f"{len(df_bruto)} registros carregados.\n")

    df_resultados = rodar_grade_experimentos(
        df_bruto=df_bruto,
        ticker=config.TICKER,
        janelas_temporais=config.JANELAS_TEMPORAIS,
        arquiteturas=config.ARQUITETURAS_MLP,
        janela_sma_curta=config.JANELA_SMA_CURTA,
        janela_sma_longa=config.JANELA_SMA_LONGA,
        janela_rsi=config.JANELA_RSI,
        proporcao_treino=config.PROPORCAO_TREINO,
        seeds=config.SEEDS,
        caminho_banco=config.BANCO_RESULTADOS,
        pasta_resultados=config.RESULTADOS,
    )

    caminho_csv = config.RESULTADOS / "resultados_experimentos.csv"
    df_resultados.to_csv(caminho_csv, index=False)

    plotar_comparacao_configuracoes(
        df_resultados, config.RESULTADOS / "comparacao_rmse.png"
    )

    print("\n=== Resumo dos experimentos (ordenado por RMSE médio) ===")
    print(
        df_resultados.sort_values("rmse_medio")[
            ["janela_temporal", "arquitetura", "rmse_medio", "rmse_desvio", "r2_medio"]
        ].to_string(index=False)
    )

    melhor = df_resultados.sort_values("rmse_medio").iloc[0]
    print(
        f"\nMelhor configuração: janela={melhor['janela_temporal']} dias, "
        f"arquitetura={melhor['arquitetura']} "
        f"(RMSE médio={melhor['rmse_medio']:.4f} ± {melhor['rmse_desvio']:.4f}, "
        f"R² médio={melhor['r2_medio']:.4f})"
    )
    print(f"\nResultados completos salvos em: {config.BANCO_RESULTADOS}")
    print(f"CSV resumo salvo em: {caminho_csv}")
    print(f"Gráficos salvos em: {config.RESULTADOS}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()