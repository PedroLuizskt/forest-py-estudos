# 🌳 forest-py-estudos

> **Estudos avançados em Machine Learning aplicado à Engenharia Florestal**
> Versão expandida e didática do pacote [`fptools`](https://github.com/RichterV/fptools), com foco em **Redes Neurais (MLP e CNN)**, relatórios analíticos aprofundados e análises gráficas robustas.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/status-em%20desenvolvimento-yellow">
</p>

---

## 📋 Sumário

- [Sobre](#-sobre)
- [Estrutura](#-estrutura)
- [Plano de Estudos](#-plano-de-estudos)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Validação](#-validação)
- [Autor](#-autor)

---

## 🎯 Sobre

Este repositório é um material de estudos **profissional e didático** que reproduz e expande os tópicos do pacote `fptools` (Msc. Vinicius Richter), adicionando:

- 🧠 **Redes Neurais Artificiais (MLP)** para volumetria, hipsometria e classificação de sítio
- 🖼️ **Redes Neurais Convolucionais (CNN)** para classificação de uso/cobertura e segmentação de copas
- 📊 **Análises gráficas profundas** (diagnósticos de resíduos, curvas de aprendizado, Grad-CAM, etc.)
- 📑 **Relatórios analíticos** comparando abordagens clássicas vs. deep learning
- 🧪 **Testes automatizados** com cobertura ≥80%
- ⚙️ **Reprodutibilidade total** via configs YAML e seeds determinísticas

## 🗂️ Estrutura

```
forest-py-estudos/
├── src/forestpy/        # Pacote instalável: from forestpy import ...
│   ├── data/            # Loaders, validadores e geradores de dados
│   ├── dendrometria/    # Volume, afilamento, hipsometria, sortimentos
│   ├── inventario/      # Amostragem e distribuição diamétrica
│   ├── sitio/           # Curvas de site e índices de produtividade
│   ├── ml/              # MLP, CNN, métricas e avaliação
│   ├── viz/             # Toolkit de visualização
│   └── utils/           # Config, I/O, logging, reprodutibilidade
├── notebooks/           # 12 sessões cronológicas de estudo
├── data/                # raw / interim / processed / external
├── models/              # Modelos treinados (.pt, .pkl)
├── reports/             # Figuras, tabelas e análises markdown
├── configs/             # Hiperparâmetros YAML por experimento
├── scripts/             # CLIs reproduzíveis
└── tests/               # pytest
```

## 📚 Plano de Estudos

| # | Sessão | Tipo |
|---|--------|------|
| 01 | Introdução aos Dados Florestais | Conceitual |
| 02 | EDA do PEF Vinhedo | Análise gráfica |
| 03 | Dendrometria Clássica | Modelagem clássica |
| 04 | Inventário e Amostragem | Estatística |
| 05 | Volumetria Clássica — Baseline | Regressão |
| 06 | 🧠 **Volumetria com MLP (PyTorch)** | Deep Learning |
| 07 | 🧠 **Hipsometria com Redes Neurais** | Deep Learning |
| 08 | 🧠 **Classificação de Sítio via DL** | Deep Learning |
| 09 | Distribuição Diamétrica: Weibull vs MLP | Comparativo |
| 10 | 🖼️ **CNN para Sensoriamento Remoto** | Deep Learning |
| 11 | 🖼️ **U-Net para Segmentação de Copas** | Deep Learning |
| 12 | 📊 Relatório Executivo Final | Consolidação |

## 🚀 Instalação

### Pré-requisitos
- Python **3.12.x**
- Git
- (Opcional) GPU CUDA para acelerar as sessões 10 e 11

### Setup

```powershell
# Clona o repositório
git clone https://github.com/PedroLuizskt/forest-py-estudos.git
cd forest-py-estudos

# Cria e ativa o ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux/Mac

# Instala em modo editável (com dev tools)
pip install -e ".[dev]"

# Registra o kernel para Jupyter
python -m ipykernel install --user --name forestpy --display-name "Python (forestpy)"
```

### Verificação rápida

```bash
python -c "import forestpy; print(forestpy.__version__)"
pytest -v
```

## 💻 Como Usar

### Via notebooks (didático)
```bash
jupyter lab notebooks/
```
Execute as sessões em ordem cronológica (01 → 12).

### Via pacote (programático)
```python
from forestpy.dendrometria.volume import schumacher_hall
from forestpy.ml.mlp.trainer import MLPTrainer
from forestpy.viz.diagnostics import plot_residuals

v = schumacher_hall(dap=20.0, h=15.0)
```

### Via CLI (scripts)
```bash
python scripts/train_mlp.py --config configs/mlp_volumetria.yaml
python scripts/evaluate.py --model models/mlp/volumetria_v1.pt
```

## 🧪 Validação

```bash
make test       # roda pytest com cobertura
make lint       # ruff + black
make notebooks  # executa notebooks ponta-a-ponta (CI-friendly)
make validate   # tudo acima
```

## 👤 Autor

**Pedro Luiz R. Vaz de Melo**
Engenheiro Florestal | Cientista de Dados | Desenvolvedor GIS

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://linkedin.com/in/seu-perfil)
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/PedroLuizskt)

---

## 🙏 Inspiração

Este projeto é uma versão expandida e didática do pacote [`fptools`](https://github.com/RichterV/fptools) de Msc. Vinicius Richter, com permissão e gratidão pela contribuição original à comunidade florestal Python brasileira.
