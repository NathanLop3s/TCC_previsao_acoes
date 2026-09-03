import sqlite3
import time
from pathlib import Path

import pandas as pd

from avaliacao import calcular_metricas, plotar_previsto_vs_real
from modelo import criar_mlp
from preprocessamento import preparar_dataset


def criar_tabela_resultados(caminho_banco: Path) -> None:
    conexao = sqlite3.connect(caminho_banco)
    conexao.execute(
        """
        CREATE TABLE IF NOT EXISTS experimentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            janela_temporal INTEGER,
            arquitetura TEXT,
            camadas TEXT,
            rmse REAL,
            r2 REAL,
            n_amostras_treino INTEGER,
            n_amostras_teste INTEGER,
            timestamp TEXT
        )
        """
    )
    conexao.commit()
    conexao.close()


def registrar_resultado(caminho_banco: Path, resultado: dict) -> None:
    conexao = sqlite3.connect(caminho_banco)
    conexao.execute(
        """
        INSERT INTO experimentos
            (ticker, janela_temporal, arquitetura, camadas, rmse, r2,
             n_amostras_treino, n_amostras_teste, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            resultado["ticker"],
            resultado["janela_temporal"],
            resultado["arquitetura"],
            str(resultado["camadas"]),
            resultado["rmse"],
            resultado["r2"],
            resultado["n_amostras_treino"],
            resultado["n_amostras_teste"],
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conexao.commit()
    conexao.close()


def rodar_grade_experimentos(
    df_bruto: pd.DataFrame,
    ticker: str,
    janelas_temporais: list,
    arquiteturas: dict,
    janela_sma_curta: int,
    janela_sma_longa: int,
    janela_rsi: int,
    proporcao_treino: float,
    seed: int,
    caminho_banco: Path,
    pasta_resultados: Path,
    salvar_grafico_melhor: bool = True,
) -> pd.DataFrame:
    criar_tabela_resultados(caminho_banco)

    linhas_resultado = []
    melhor_rmse = float("inf")
    melhor_config = None

    total = len(janelas_temporais) * len(arquiteturas)
    contador = 0

    for tamanho_janela in janelas_temporais:
        dataset = preparar_dataset(
            df_bruto=df_bruto,
            tamanho_janela=tamanho_janela,
            janela_sma_curta=janela_sma_curta,
            janela_sma_longa=janela_sma_longa,
            janela_rsi=janela_rsi,
            proporcao_treino=proporcao_treino,
        )

        for nome_arquitetura, camadas in arquiteturas.items():
            contador += 1
            print(
                f"[{contador}/{total}] janela={tamanho_janela} dias | "
                f"arquitetura={nome_arquitetura} {camadas}"
            )

            modelo = criar_mlp(hidden_layer_sizes=camadas, seed=seed)
            modelo.fit(dataset["X_treino"], dataset["y_treino"])

            previsao_norm = modelo.predict(dataset["X_teste"])
            previsao = dataset["scaler_y"].inverse_transform(
                previsao_norm.reshape(-1, 1)
            ).ravel()

            metricas = calcular_metricas(dataset["y_teste_original"], previsao)

            resultado = {
                "ticker": ticker,
                "janela_temporal": tamanho_janela,
                "arquitetura": nome_arquitetura,
                "camadas": camadas,
                "rmse": metricas["rmse"],
                "r2": metricas["r2"],
                "n_amostras_treino": len(dataset["X_treino"]),
                "n_amostras_teste": len(dataset["X_teste"]),
            }
            registrar_resultado(caminho_banco, resultado)
            linhas_resultado.append(resultado)

            if metricas["rmse"] < melhor_rmse:
                melhor_rmse = metricas["rmse"]
                melhor_config = {
                    **resultado,
                    "datas_teste": dataset["datas_teste"],
                    "y_real": dataset["y_teste_original"],
                    "y_previsto": previsao,
                }

    if salvar_grafico_melhor and melhor_config is not None:
        plotar_previsto_vs_real(
            datas=melhor_config["datas_teste"],
            y_real=melhor_config["y_real"],
            y_previsto=melhor_config["y_previsto"],
            titulo=(
                f"{ticker} -- melhor config: janela={melhor_config['janela_temporal']}d, "
                f"{melhor_config['arquitetura']} (RMSE={melhor_config['rmse']:.3f})"
            ),
            caminho_saida=pasta_resultados / "melhor_modelo_previsto_vs_real.png",
        )

    return pd.DataFrame(linhas_resultado)