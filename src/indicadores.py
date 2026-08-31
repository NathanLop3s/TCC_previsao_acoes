import pandas as pd


def media_movel_simples(serie: pd.Series, janela: int) -> pd.Series:
    return serie.rolling(window=janela).mean()


def indice_forca_relativa(serie: pd.Series, janela: int = 14) -> pd.Series:
    delta = serie.diff()

    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)

    media_ganho = ganho.ewm(alpha=1 / janela, adjust=False).mean()
    media_perda = perda.ewm(alpha=1 / janela, adjust=False).mean()

    rs = media_ganho / media_perda.replace(0, pd.NA)
    ifr = 100 - (100 / (1 + rs))

    ifr = ifr.fillna(100)

    return ifr


def adicionar_indicadores(
    df: pd.DataFrame,
    janela_sma_curta: int,
    janela_sma_longa: int,
    janela_rsi: int,
) -> pd.DataFrame:
    df = df.copy()
    df["SMA_curta"] = media_movel_simples(df["Close"], janela_sma_curta)
    df["SMA_longa"] = media_movel_simples(df["Close"], janela_sma_longa)
    df["RSI"] = indice_forca_relativa(df["Close"], janela_rsi)
    return df