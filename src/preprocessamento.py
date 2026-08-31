import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from indicadores import adicionar_indicadores


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df[df["Close"] > 0]
    return df


def construir_janelas(
    df: pd.DataFrame, tamanho_janela: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    fechamentos = df["Close"].values
    indicadores = df[["SMA_curta", "SMA_longa", "RSI"]].values
    datas = df.index

    X, y, y_datas = [], [], []

    for i in range(tamanho_janela - 1, len(df) - 1):
        janela_precos = fechamentos[i - tamanho_janela + 1 : i + 1]
        features_extras = indicadores[i]
        amostra = np.concatenate([janela_precos, features_extras])

        X.append(amostra)
        y.append(fechamentos[i + 1])
        y_datas.append(datas[i + 1])

    return np.array(X), np.array(y), pd.DatetimeIndex(y_datas)


def preparar_dataset(
    df_bruto: pd.DataFrame,
    tamanho_janela: int,
    janela_sma_curta: int,
    janela_sma_longa: int,
    janela_rsi: int,
    proporcao_treino: float,
) -> dict:
    df = limpar_dados(df_bruto)
    df = adicionar_indicadores(df, janela_sma_curta, janela_sma_longa, janela_rsi)
    df = df.dropna()

    X, y, datas_y = construir_janelas(df, tamanho_janela)

    corte = int(len(X) * proporcao_treino)
    X_treino, X_teste = X[:corte], X[corte:]
    y_treino, y_teste = y[:corte], y[corte:]
    datas_teste = datas_y[corte:]

    if len(X_teste) == 0:
        raise ValueError(
            "Conjunto de teste vazio -- aumente o histórico coletado ou "
            "reduza a proporção de treino / o tamanho da janela."
        )

    scaler_X = MinMaxScaler()
    X_treino_norm = scaler_X.fit_transform(X_treino)
    X_teste_norm = scaler_X.transform(X_teste)

    scaler_y = MinMaxScaler()
    y_treino_norm = scaler_y.fit_transform(y_treino.reshape(-1, 1)).ravel()
    y_teste_norm = scaler_y.transform(y_teste.reshape(-1, 1)).ravel()

    return {
        "X_treino": X_treino_norm,
        "X_teste": X_teste_norm,
        "y_treino": y_treino_norm,
        "y_teste": y_teste_norm,
        "scaler_y": scaler_y,
        "datas_teste": datas_teste,
        "y_teste_original": y_teste,
    }