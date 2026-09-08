# Predição de Preços de Ações da B3 com Rede Neural MLP

Projeto de TCC que investiga o uso de uma rede neural Multilayer Perceptron (MLP)
para prever o preço de fechamento de ativos da B3, testando o impacto de diferentes
janelas temporais de entrada (H1) e diferentes arquiteturas de rede (H2).

## Hipóteses

- **H1:** janelas temporais mais longas (mais dias de histórico como entrada) melhoram
  a capacidade do modelo de capturar padrões de curto prazo.
- **H2:** arquiteturas com duas ou mais camadas ocultas reduzem o RMSE em comparação
  com uma rede de camada única.

## Tecnologias

- **Python 3.11**
- **yfinance** — coleta de dados históricos (B3, via Yahoo Finance)
- **pandas** — limpeza, indicadores técnicos e construção das janelas temporais
- **scikit-learn** — modelo MLP (`MLPRegressor`), normalização e métricas
- **matplotlib** — geração dos gráficos de resultado
- **SQLite** — registro dos resultados de cada experimento

## Estrutura do projeto

├── data/
│ ├── raw/ dados brutos baixados da API
│ └── processed/ reservado para dados intermediários
├── results/ gráficos e CSV gerados
├── src/
│ ├── config.py configurações centrais (ticker, grade de hiperparâmetros)
│ ├── coleta.py coleta de dados via yfinance
│ ├── indicadores.py SMA e RSI (Wilder, 1978)
│ ├── preprocessamento.py limpeza, janelas temporais, normalização
│ ├── modelo.py definição da arquitetura MLP
│ ├── experimentos.py grade de experimentos H1 x H2
│ ├── avaliacao.py métricas (RMSE, R²) e gráficos
│ └── main.py orquestra o pipeline completo
├── requirements.txt
└── resultados.db banco SQLite com histórico de experimentos 

## Como rodar

```bash
git clone <url-do-repo>
cd TCC_previsao_acoes

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

cd src
python main.py
```

O ticker, o período histórico e a grade de hiperparâmetros testada podem ser
alterados diretamente em `src/config.py`.

## Resultados


## Autor

Nathan Da Cruz Lopes — TCC em Ciência da Computação.
