import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def coletar_dados(ticker: str, periodo: str, destino: Path) -> pd.DataFrame:
    logger.info(f"Baixando dados de {ticker} (período: {periodo})...")

    df = yf.download(ticker, period=periodo, progress=False, auto_adjust=True)

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

    caminho_saida = destino / f"{ticker.replace('.', '_')}.parquet"
    df.to_parquet(caminho_saida)
    logger.info(f"{len(df)} registros salvos em {caminho_saida}")

    return df


if __name__ == "__main__":
    from config import TICKER, PERIODO_HISTORICO, DADOS_BRUTOS

    coletar_dados(TICKER, PERIODO_HISTORICO, DADOS_BRUTOS)