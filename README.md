# 🎮 GameTox: Aumento Sintético de Dados via LLM para Classificação de Toxicidade sob Desbalanceamento Extremo

Este repositório contém o código-fonte, dados e pipelines experimentais do projeto de pesquisa voltado ao combate do desbalanceamento extremo de classes na detecção automática de toxicidade em chats de jogos multiplayer (*World of Tanks* / Corpus **GameTox**).

A abordagem investiga o impacto do **Aumento Sintético de Dados Semânticos** (*Data Augmentation*) utilizando o modelo **Qwen 2.5 7B Instruct** executado localmente (quantizado em 4-bit) com técnicas de *Few-Shot Prompting* restritivas.

---

## 📌 Visão Geral do Projeto

A moderação automática de chats em jogos digitais enfrenta dois grandes desafios:
1. **Complexidade Linguística:** Mensagens curtas, ruidosas, com gírias (*kys*, *fgt*, *noob*), *typos* e ausência de pontuação.
2. **Desbalanceamento Extremo de Classes:** As categorias de infração grave — Discurso de Ódio (*Hate*), Ameaças (*Threats*) e Extremismo (*Extremism*) — representam menos de 1% das interações reais.

Este projeto avalia a hipótese de utilizar Grandes Modelos de Linguagem (LLMs) para gerar amostras sintéticas realistas e encorpar as classes raras, comparando o desempenho de classificadores sob representações esparsas (**TF-IDF**) e densas (**Word2Vec**).

---

## 📊 Principais Resultados

A avaliação foi realizada sobre um conjunto de **Teste Real e Isolado ($n = 4.986$)**, sem vazamento de dados (*Data Leakage*):

| Algoritmo | Baseline (TF-IDF) | Augmented (TF-IDF) | Baseline (W2V) | Augmented (W2V) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (RF)** | $0{,}3892$ | **$0{,}4133$** | $0{,}3568$ | **$0{,}1503$** / **$0{,}3766$*** |
| **Linear SVM** | $0{,}4265$ | $0{,}4151$ | $0{,}3067$ | $0{,}2750$ |
| **Logistic Regression (LR)** | $0{,}4277$ | $0{,}3663$ | $0{,}2112$ | $0{,}2316$ |
| **Naive Bayes (NB)** | $0{,}2849$ | $0{,}2849$ | $0{,}0948$ | $0{,}0827$ |

*\*Observação: Modelos não lineares como o Random Forest beneficiaram-se significativamente da diversidade semântica sintética, registrando ganhos de até +7,14% no Macro F1-Score.*

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.12
* **Ambiente de Execução:** Google Colab (GPU NVIDIA T4 16GB)
* **LLM Local:** `Qwen/Qwen2.5-7B-Instruct` (via `transformers` e `bitsandbytes` em 4-bits)
* **Machine Learning & NLP:** `scikit-learn`, `gensim` (Word2Vec), `pandas`, `numpy`
* **Visualização & Gráficos:** `matplotlib`, `seaborn`, `TikZ` (LaTeX)

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
Instale as dependências necessárias no seu ambiente ou notebook:

```bash
pip install transformers accelerate bitsandbytes torch pandas scikit-learn gensim
