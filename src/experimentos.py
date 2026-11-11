import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from avaliacao import (
    calcular_metricas,
    escala_mase_treino,
    plotar_previsto_vs_real,
)
from modelo import criar_mlp
from preprocessamento import preparar_dataset

METRICAS = ["rmse", "mae", "mape", "mase", "r2", "acuracia_direcional"]


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
            seed INTEGER,
            rmse REAL,
            mae REAL,
            mape REAL,
            mase REAL,
            r2 REAL,
            acuracia_direcional REAL,
            epocas INTEGER,
            n_amostras_treino INTEGER,
            n_amostras_teste INTEGER,
            inicio_teste TEXT,
            fim_teste TEXT,
            timestamp TEXT
        )
        """
    )
    conexao.commit()
    conexao.close()


def registrar_resultado(caminho_banco: Path, r: dict) -> None:
    colunas = [
        "ticker", "janela_temporal", "arquitetura", "camadas", "seed",
        *METRICAS, "epocas", "n_amostras_treino", "n_amostras_teste",
        "inicio_teste", "fim_teste", "timestamp",
    ]
    valores = [str(r[c]) if c == "camadas" else r[c] for c in colunas]
    conexao = sqlite3.connect(caminho_banco)
    conexao.execute(
        f"INSERT INTO experimentos ({', '.join(colunas)}) "
        f"VALUES ({', '.join('?' * len(colunas))})",
        valores,
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
    seeds: list,
    ddof_desvio: int,
    caminho_banco: Path,
    pasta_resultados: Path,
    pasta_previsoes: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna (resumo por configuração, execuções individuais, baseline de persistência)."""
    criar_tabela_resultados(caminho_banco)
    nome_ticker = ticker.replace(".", "_")

    linhas_resumo, linhas_execucoes, linhas_baseline = [], [], []
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
        escala = escala_mase_treino(
            dataset["y_treino_original"], dataset["y_anterior_treino"]
        )

        # Baseline de persistência: prevê que o fechamento de amanhã = o de hoje.
        base = calcular_metricas(
            dataset["y_teste_original"], dataset["y_anterior_teste"],
            dataset["y_anterior_teste"], escala,
        )
        # A persistência nunca prevê alta nem queda, então a acurácia direcional
        # não se aplica a ela.
        base["acuracia_direcional"] = float("nan")
        linhas_baseline.append({
            "ticker": ticker, "janela_temporal": tamanho_janela,
            "modelo": "persistencia (y_t+1 = y_t)", **base,
            "inicio_treino": dataset["periodo_treino"][0],
            "fim_treino": dataset["periodo_treino"][1],
            "inicio_teste": dataset["periodo_teste"][0],
            "fim_teste": dataset["periodo_teste"][1],
            "n_amostras_treino": len(dataset["X_treino"]),
            "n_amostras_teste": len(dataset["X_teste"]),
        })

        for nome_arquitetura, camadas in arquiteturas.items():
            contador += 1
            print(
                f"[{ticker}] [{contador}/{total}] janela={tamanho_janela} dias | "
                f"arquitetura={nome_arquitetura} {camadas} ({len(seeds)} seeds)"
            )

            execucoes = []
            for seed in seeds:
                modelo = criar_mlp(hidden_layer_sizes=camadas, seed=seed)
                modelo.fit(dataset["X_treino"], dataset["y_treino"])

                previsao_norm = modelo.predict(dataset["X_teste"])
                previsao = dataset["scaler_y"].inverse_transform(
                    previsao_norm.reshape(-1, 1)
                ).ravel()

                metricas = calcular_metricas(
                    dataset["y_teste_original"], previsao,
                    dataset["y_anterior_teste"], escala,
                )

                registro = {
                    "ticker": ticker,
                    "janela_temporal": tamanho_janela,
                    "arquitetura": nome_arquitetura,
                    "camadas": camadas,
                    "seed": seed,
                    **metricas,
                    "epocas": int(modelo.n_iter_),
                    "n_amostras_treino": len(dataset["X_treino"]),
                    "n_amostras_teste": len(dataset["X_teste"]),
                    "inicio_teste": dataset["periodo_teste"][0],
                    "fim_teste": dataset["periodo_teste"][1],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                registrar_resultado(caminho_banco, registro)
                linhas_execucoes.append({**registro, "camadas": str(camadas)})

                # Previsões salvas (permite recalcular qualquer métrica depois)
                pd.DataFrame({
                    "data": dataset["datas_teste"],
                    "fechamento_anterior": dataset["y_anterior_teste"],
                    "fechamento_real": dataset["y_teste_original"],
                    "fechamento_previsto": previsao,
                }).to_csv(
                    pasta_previsoes
                    / f"{nome_ticker}_j{tamanho_janela}_{nome_arquitetura}_seed{seed}.csv",
                    index=False,
                )

                execucoes.append({**metricas, "seed": seed, "y_previsto": previsao})

            resumo = {
                "ticker": ticker,
                "janela_temporal": tamanho_janela,
                "arquitetura": nome_arquitetura,
                "camadas": str(camadas),
                "n_seeds": len(seeds),
            }
            for m in METRICAS:
                valores = np.array([e[m] for e in execucoes])
                resumo[f"{m}_medio"] = float(np.mean(valores))
                resumo[f"{m}_desvio"] = float(np.std(valores, ddof=ddof_desvio))
                resumo[f"{m}_mediana"] = float(np.median(valores))
            linhas_resumo.append(resumo)

            if resumo["rmse_medio"] < melhor_rmse:
                melhor_rmse = resumo["rmse_medio"]
                execucao_tipica = min(
                    execucoes, key=lambda e: abs(e["rmse"] - resumo["rmse_medio"])
                )
                melhor_config = {
                    **resumo,
                    "seed_tipica": execucao_tipica["seed"],
                    "datas_teste": dataset["datas_teste"],
                    "y_real": dataset["y_teste_original"],
                    "y_previsto": execucao_tipica["y_previsto"],
                }

    if melhor_config is not None:
        plotar_previsto_vs_real(
            datas=melhor_config["datas_teste"],
            y_real=melhor_config["y_real"],
            y_previsto=melhor_config["y_previsto"],
            titulo=(
                f"{ticker} — melhor configuração (média): "
                f"janela={melhor_config['janela_temporal']}d, "
                f"{melhor_config['arquitetura']} | seed {melhor_config['seed_tipica']} "
                f"(RMSE médio={melhor_config['rmse_medio']:.3f} ± "
                f"{melhor_config['rmse_desvio']:.3f})"
            ),
            caminho_saida=pasta_resultados / f"{nome_ticker}_previsto_vs_real.png",
        )

    return (
        pd.DataFrame(linhas_resumo),
        pd.DataFrame(linhas_execucoes),
        pd.DataFrame(linhas_baseline),
    )
