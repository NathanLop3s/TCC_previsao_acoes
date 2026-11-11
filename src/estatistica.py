"""
Testes estatísticos sobre o RMSE das execuções individuais (5 seeds por configuração).

- ANOVA de dois fatores com interação: RMSE ~ janela + arquitetura + janela:arquitetura
  (fator janela -> H1, fator arquitetura -> H2, interação -> H3)
- Pressupostos: Shapiro-Wilk nos resíduos (normalidade) e Levene entre os grupos
  (homogeneidade de variâncias)
- Pós-teste de Tukey HSD entre as 12 combinações (quais pares diferem de fato)
"""
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def analisar(df_execucoes: pd.DataFrame, ticker: str, pasta_saida: Path, alfa: float = 0.05) -> str:
    df = df_execucoes[df_execucoes["ticker"] == ticker].copy()
    df["janela"] = df["janela_temporal"].astype(str)
    df["combinacao"] = df["janela"] + "d_" + df["arquitetura"]
    nome = ticker.replace(".", "_")

    modelo = smf.ols("rmse ~ C(janela) * C(arquitetura)", data=df).fit()
    tabela_anova = sm.stats.anova_lm(modelo, typ=2)
    tabela_anova.to_csv(pasta_saida / f"{nome}_anova.csv")

    shapiro = stats.shapiro(modelo.resid)
    grupos = [g["rmse"].values for _, g in df.groupby("combinacao")]
    levene = stats.levene(*grupos)

    tukey = pairwise_tukeyhsd(df["rmse"], df["combinacao"], alpha=alfa)
    df_tukey = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    df_tukey.to_csv(pasta_saida / f"{nome}_tukey.csv", index=False)

    # Kruskal-Wallis (não paramétrico): confirmação caso os resíduos não sejam normais
    linhas_kw, linhas_tukey_fator = [], []
    for fator in ["janela", "arquitetura"]:
        grupos_f = [g["rmse"].values for _, g in df.groupby(fator)]
        kw = stats.kruskal(*grupos_f)
        linhas_kw.append(f"Kruskal-Wallis ({fator}): H={kw.statistic:.4f}, p={kw.pvalue:.4f}")
        # Tukey por fator (comparações entre os níveis de cada fator)
        tk = pairwise_tukeyhsd(df["rmse"], df[fator], alpha=alfa)
        linhas_tukey_fator.append(f"Tukey HSD por fator ({fator}):\n{tk.summary()}")

    relatorio = [
        f"===== {ticker} — ANOVA de dois fatores (variável: RMSE por execução) =====",
        tabela_anova.to_string(),
        "",
        f"Shapiro-Wilk (normalidade dos resíduos): W={shapiro.statistic:.4f}, p={shapiro.pvalue:.4f}",
        f"Levene (homogeneidade das variâncias):   W={levene.statistic:.4f}, p={levene.pvalue:.4f}",
        "",
        *linhas_kw,
        "",
        *linhas_tukey_fator,
        "",
        f"Tukey HSD (alfa={alfa}) — pares com diferença significativa:",
        df_tukey[df_tukey["reject"] == True].to_string(index=False)
        if (df_tukey["reject"] == True).any() else "  nenhum par significativo",
        "",
    ]
    texto = "\n".join(relatorio)
    (pasta_saida / f"{nome}_estatistica.txt").write_text(texto, encoding="utf-8")
    return texto
