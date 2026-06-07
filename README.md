# 🤖 IA — Modelos de Regressão e Classificação Linear

---

## 📋 Descrição

Este repositório implementa e compara modelos preditivos supervisionados em duas tarefas distintas de Machine Learning, desenvolvidos **com NumPy e Matplotlib**, sem uso de bibliotecas de ML prontas (scikit-learn, etc.).

| Etapa | Tarefa | Dataset |
| ----- | ------ | ------- |
| 1 | **Regressão** — Previsão de potência de aerogerador a partir da velocidade do vento | `data/aerogerador.dat` |
| 2 | **Classificação** — Reconhecimento de expressões faciais via sinais EMG | `data/EMGsDataset.csv` |

---

## 📁 Estrutura do Repositório

```

├── README.md
├── src/
│   ├── 1_REGRESSAO.py        # Implementação completa da tarefa de regressão
│   └── 2_CLASSIFICACAO.py    # Implementação completa da tarefa de classificação
├── data/
│   ├── aerogerador.dat       # Dataset de velocidade do vento × potência gerada
│   └── EMGsDataset.csv       # Dataset de sinais EMG faciais (50.000 amostras)
└── report/
    └── relatorio_IA.pdf      # Relatório completo com resultados e discussões
```

---

## ⚙️ Requisitos

```bash
pip install numpy matplotlib
```

Versões utilizadas: Python 3.x · NumPy · Matplotlib

---

## 🌬️ Tópico 1 — Regressão (`src/1_REGRESSAO.py`)

### Objetivo

Prever a **potência gerada (kW)** por um aerogerador a partir da **velocidade do vento (m/s)**.

### Dataset

* **Arquivo:** `data/aerogerador.dat`
* Variável independente: velocidade do vento (m/s)
* Variável dependente: potência gerada (kW)

### Modelos Implementados

| Modelo | Descrição |
| ------ | --------- |
| **Média da variável dependente** | Baseline — prediz sempre a média de y do treino |
| **MQO Tradicional** | Mínimos Quadrados Ordinários: β = (XᵀX)⁻¹Xᵀy |
| **MQO Regularizado (Tikhonov)** | Ridge Regression com λ ∈ {0, 0.25, 0.5, 0.75, 1} |

### Validação

* **Método:** Random Subsampling (Monte Carlo)
* **Rodadas:** R = 500
* **Split:** 80% treino / 20% teste
* **Métricas:** MSE e R²

### Resultados (Tabela MSE — R=500 rodadas)

| Modelo | Média | Desvio-Padrão | Máx. | Mín. |
| ------ | ----- | -------------- | ---- | ---- |
| Média da var. dep. | 11.140,32 | 619,13 | 13.237,72 | 9.346,40 |
| MQO Tradicional | 778,64 | 166,08 | 1.298,09 | 459,11 |
| Tikhonov λ=0 | 778,64 | 166,08 | 1.298,09 | 459,11 |
| Tikhonov λ=0.25 | 778,64 | 166,05 | 1.297,96 | 459,17 |
| Tikhonov λ=0.5 | 778,64 | 166,02 | 1.297,83 | 459,24 |
| Tikhonov λ=0.75 | 778,64 | 165,98 | 1.297,70 | 459,30 |
| Tikhonov λ=1 | 778,64 | 165,95 | 1.297,57 | 459,37 |

### Resultados (Tabela R² — R=500 rodadas)

| Modelo | Média | Desvio-Padrão | Máx. | Mín. |
| ------ | ----- | -------------- | ---- | ---- |
| Média da var. dep. | −0,0026 | 0,0037 | 0 | −0,0363 |
| MQO Tradicional | 0,9301 | 0,0136 | 0,9566 | 0,8866 |
| Tikhonov (todos λ) | 0,9301 | 0,0136 | 0,9566 | 0,8866 |

> **Conclusão:** O MQO e os modelos de Tikhonov tiveram desempenho praticamente idêntico (R² ≈ 0,93), demonstrando que a regularização não agrega ganho expressivo neste cenário de baixa dimensionalidade (p=1) com dados abundantes. O modelo de média obteve R² negativo, confirmando total incapacidade preditiva.

---

## 🧠 Tópico 2 — Classificação (`src/2_CLASSIFICACAO.py`)

### Objetivo

Reconhecer **5 expressões faciais** a partir de sinais de eletromiografia (EMG) capturados em dois músculos faciais.

### Dataset

* **Arquivo:** `data/EMGsDataset.csv`
* **Amostras:** N = 50.000
* **Features:** p = 2 (Sensor 1: Corrugador do Supercílio; Sensor 2: Zigomático Maior)
* **Classes:** C = 5

| Rótulo | Expressão |
| ------- | --------- |
| 1 | Neutro |
| 2 | Sorriso |
| 3 | Sobrancelhas Levantadas |
| 4 | Surpreso |
| 5 | Rabugento |

### Modelos Implementados

| Modelo | Tipo de Fronteira |
| ------ | ----------------- |
| **MQO Tradicional** | Linear |
| **Classificador Gaussiano Tradicional (QDA)** | Quadrática |
| **Classificador Gaussiano — Covariâncias Iguais (LDA)** | Linear |
| **Classificador Gaussiano — Covariância Agregada** | Linear |
| **Naive Bayes** | Quadrática (diagonal) |
| **Classificador de Friedman (Regularizado)** | Quadrática → Linear (controlado por λ) |

### Seleção do Hiperparâmetro λ (Friedman)

* **Método:** K-Fold Cross Validation (K = 5)
* **λ testados:** {0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}
* **λ ótimo encontrado:** `0.01` (acurácia média de validação: 99,65%)

### Validação

* **Método:** Simulação de Monte Carlo
* **Rodadas:** R = 500
* **Split:** 80% treino / 20% teste
* **Métrica:** Acurácia (taxa de acerto)

### Resultados (Monte Carlo — R=500 rodadas)

| Modelo | Média | Desvio-Padrão | Maior | Menor |
| ------ | ----- | -------------- | ----- | ----- |
| MQO Tradicional | 72,39% | 0,66% | 74,10% | 70,33% |
| Gaussiano (QDA) | 96,70% | 0,17% | 97,17% | 96,14% |
| Gaussiano (Cov. todo cj. treino — LDA) | 96,27% | 0,18% | 96,73% | 95,72% |
| Gaussiano (Cov. Agregada) | 94,85% | 0,27% | 95,56% | 93,97% |
| Naive Bayes | 96,56% | 0,17% | 97,04% | 95,98% |
| **Friedman (λ=0.01)** | **99,65%** | **0,05%** | **99,82%** | **99,50%** |

> **Conclusão:** O classificador de Friedman com λ=0.01 alcançou o melhor desempenho (99,65%), confirmando que fronteiras quadráticas são essenciais para este problema não-linearmente separável. O MQO, limitado a fronteiras lineares, obteve apenas 72,39%.

---

## 📊 Como Executar

### Tarefa de Regressão

```bash
python src/1_REGRESSAO.py
```

**Saídas geradas:**

* Gráfico de espalhamento (velocidade × potência)
* Gráfico de ajuste dos 7 modelos
* Tabelas de MSE e R² no terminal

### Tarefa de Classificação

```bash
python src/2_CLASSIFICACAO.py
```

**Saídas geradas:**

* Gráfico de espalhamento EMG por expressão facial
* Painéis com fronteiras de decisão dos 6 modelos
* Curva de acurácia × λ (K-Fold)
* Histograma de acurácias (Monte Carlo)
* Tabelas de resultados no terminal

> **Atenção:** Os scripts assumem que os datasets estão em `../data/` relativamente à pasta `src/`. Ajuste os caminhos caso execute de outro diretório.

---

## 📄 Relatório

O arquivo [`report/relatorio_IA.pdf`](./report/relatorio_IA.pdf) contém o relatório completo no formato de artigo científico, incluindo:

* **Resumo** — síntese dos métodos e principais resultados
* **Metodologia** — descrição detalhada de cada modelo e protocolo de validação
* **Resultados** — tabelas, gráficos e análises quantitativas
* **Conclusões** — discussão crítica sobre o desempenho de cada abordagem

---

## 📐 Fundamentos Matemáticos

### MQO Tradicional

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

### MQO Regularizado (Tikhonov / Ridge)

$$\hat{\beta} = (X^TX + \lambda I^*)^{-1}X^Ty$$

onde $I^*$ é a matriz identidade com o elemento (0,0) zerado (não penaliza o intercepto).

### Classificador Gaussiano — Log-verossimilhança

$$\log P(C_k \mid x) \propto \log p(x \mid C_k) + \log \pi_k$$

### Classificador de Friedman

$$\Sigma_k^{(\lambda)} = (1-\lambda)\hat{\Sigma}_k + \lambda\hat{\Sigma}_{pool}$$

---

## 🔬 Observações Técnicas

* Todos os modelos foram implementados **manualmente com NumPy** (sem scikit-learn ou similares).
* A regularização de Tikhonov **não penaliza o intercepto** (elemento [0,0] da identidade zerado).
* O classificador gaussiano utiliza log-probabilidades para evitar underflow numérico.
* A matriz de covariância recebe uma regularização numérica de `1e-6` na diagonal para garantir inversibilidade.