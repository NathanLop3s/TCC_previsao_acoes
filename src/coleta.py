import json
import logging
import platform
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def versoes_bibliotecas() -> dict:
    """Versões exatas usadas na execução (para citar no TCC)."""
    libs = ["yfinance", "pandas", "numpy", "scikit-learn", "matplotlib", "statsmodels", "scipy"]
    versoes = {"python": platform.python_version()}
    for lib in libs:
        try:
            versoes[lib] = version(lib)
        except Exception:
            versoes[lib] = "não instalada"
    return versoes


def coletar_dados(ticker: str, data_inicio: str, data_fim: str, destino: Path) -> pd.DataFrame:
    logger.info(f"Baixando dados de {ticker} ({data_inicio} a {data_fim}, fim exclusivo)...")

    df = yf.download(
        ticker, start=data_inicio, end=data_fim, progress=False, auto_adjust=True
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"Nenhum dado retornado para {ticker}. Verifique o ticker "
            f"e a conexão com a internet."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Data"
    df = df.sort_index()

    nome = ticker.replace(".", "_")
    df.to_parquet(destino / f"{nome}.parquet")
    df.to_csv(destino / f"{nome}.csv")

    # Metadados da coleta: respondem "data dos dados, data da coleta e versões".
    metadados = {
        "ticker": ticker,
        "fonte": "Yahoo Finance via biblioteca yfinance",
        "auto_adjust": True,
        "data_inicio_solicitada": data_inicio,
        "data_fim_solicitada_exclusiva": data_fim,
        "primeiro_pregao": str(df.index.min().date()),
        "ultimo_pregao": str(df.index.max().date()),
        "n_registros_brutos": int(len(df)),
        "data_hora_coleta": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versoes": versoes_bibliotecas(),
    }
    with open(destino / f"{nome}_metadados.json", "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)

    return df
