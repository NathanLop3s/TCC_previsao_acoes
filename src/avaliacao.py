from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def calcular_metricas(y_real: np.ndarray, y_previsto: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
    r2 = r2_score(y_real, y_previsto)
    return {"rmse": rmse, "r2": r2}


def plotar_previsto_vs_real(
    datas: pd.DatetimeIndex,
    y_real: np.ndarray,
    y_previsto: np.ndarray,
    titulo: str,
    caminho_saida: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(datas, y_real, label="Preço real", linewidth=1.5)
    plt.plot(datas, y_previsto, label="Preço previsto", linewidth=1.5, linestyle="--")
    plt.title(titulo)
    plt.xlabel("Data")
    plt.ylabel("Preço de fechamento (R$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=120)
    plt.close()

def plotar_comparacao_configuracoes(
    df_resultados: pd.DataFrame, caminho_saida: Path
) -> None:
    pivot_media = df_resultados.pivot(
        index="arquitetura", columns="janela_temporal", values="rmse_medio"
    )
    pivot_desvio = df_resultados.pivot(
        index="arquitetura", columns="janela_temporal", values="rmse_desvio"
    )

    ax = pivot_media.plot(
        kind="bar", yerr=pivot_desvio, capsize=4, figsize=(10, 5)
    )
    ax.set_title(
        "RMSE médio por arquitetura e janela temporal (± desvio padrão entre seeds)"
    )
    ax.set_xlabel("Arquitetura da MLP")
    ax.set_ylabel("RMSE médio")
    ax.legend(title="Janela temporal (dias)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=120)
    plt.close()