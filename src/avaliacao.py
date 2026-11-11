from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def calcular_metricas(
    y_real: np.ndarray,
    y_previsto: np.ndarray,
    y_anterior: np.ndarray,
    escala_mase: float,
) -> dict:
    """
    y_real      -> fechamento real do dia t+1
    y_previsto  -> fechamento previsto para t+1
    y_anterior  -> fechamento real do dia t (último dia da janela)
    escala_mase -> MAE da previsão ingênua (persistência) no conjunto de TREINO
                   (Hyndman; Koehler, 2006)
    """
    rmse = float(np.sqrt(mean_squared_error(y_real, y_previsto)))
    mae = float(mean_absolute_error(y_real, y_previsto))
    mape = float(mean_absolute_percentage_error(y_real, y_previsto) * 100)  # em %
    mase = mae / escala_mase
    r2 = float(r2_score(y_real, y_previsto))

    # Acurácia direcional: % de dias em que o modelo acertou se o preço
    # subiu ou caiu em relação ao fechamento do dia anterior.
    direcao_real = np.sign(y_real - y_anterior)
    direcao_prevista = np.sign(y_previsto - y_anterior)
    acuracia_direcional = float(np.mean(direcao_real == direcao_prevista) * 100)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "mase": mase,
        "r2": r2,
        "acuracia_direcional": acuracia_direcional,
    }


def escala_mase_treino(y_treino: np.ndarray, y_anterior_treino: np.ndarray) -> float:
    return float(np.mean(np.abs(y_treino - y_anterior_treino)))


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
    plt.savefig(caminho_saida, dpi=150)
    plt.close()


def plotar_comparacao_configuracoes(
    df_resultados: pd.DataFrame, ticker: str, caminho_saida: Path
) -> None:
    pivot_media = df_resultados.pivot(
        index="arquitetura", columns="janela_temporal", values="rmse_medio"
    )
    pivot_desvio = df_resultados.pivot(
        index="arquitetura", columns="janela_temporal", values="rmse_desvio"
    )

    ax = pivot_media.plot(kind="bar", yerr=pivot_desvio, capsize=4, figsize=(10, 5))
    ax.set_title(
        f"{ticker} — RMSE médio por arquitetura e janela temporal "
        f"(± desvio padrão entre seeds)"
    )
    ax.set_xlabel("Arquitetura da MLP")
    ax.set_ylabel("RMSE médio (R$)")
    ax.legend(title="Janela temporal (dias)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close()
