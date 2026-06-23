# forest-py-estudos

**Aplicação de Machine Learning e Redes Neurais à Engenharia Florestal: estudo aprofundado em Python**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-223%20passing-success)](tests/)
[![Sessions](https://img.shields.io/badge/sessions-12%2F12%20complete-success)](notebooks/)
[![Code style](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)

---

## Sumário

- [Resumo](#resumo)
- [Tese Central](#tese-central)
- [Motivação e Inspiração](#motivação-e-inspiração)
- [Diferenciais Metodológicos](#diferenciais-metodológicos)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Trilha Didática Completa](#trilha-didática-completa)
- [Resultados Consolidados](#resultados-consolidados)
- [Requisitos e Instalação](#requisitos-e-instalação)
- [Utilização](#utilização)
- [Validação e Qualidade de Código](#validação-e-qualidade-de-código)
- [Reprodutibilidade](#reprodutibilidade)
- [Referências](#referências)
- [Citação](#citação)
- [Autor](#autor)
- [Licença](#licença)

---

## Resumo

Este repositório consolida um estudo aplicado de **Machine Learning e Deep Learning** ao domínio da **Engenharia Florestal**, com ênfase em mensuração florestal, inventário, classificação de sítio, distribuição diamétrica e análises geoespaciais (incluindo segmentação semântica). O material é estruturado como um pacote Python instalável (`forestpy`), acompanhado de **doze sessões didáticas** completas em formato de notebook que documentam o raciocínio metodológico desde a análise exploratória até a modelagem com redes neurais densas (MLP) e convolucionais (CNN, U-Net).

Os datasets são sintéticos e reproduzíveis, com distribuições estatísticas realistas de plantios de *Eucalyptus grandis*. A escolha por dados sintéticos garante reprodutibilidade total, independência de fontes externas e — principalmente — **controle sobre os mecanismos geradores**, permitindo investigar empiricamente em quais regimes os modelos paramétricos clássicos superam o aprendizado de máquina, e vice-versa, sem comprometer a validade metodológica das análises.

## Tese Central

A investigação conduzida ao longo das doze sessões resultou em uma tese metodológica clara, sustentada por seis experimentos comparativos rigorosos:

> **Não existe paradigma universalmente superior em ciência de dados aplicada. Cada um vence em um regime estrutural específico, identificável a priori por características do problema (forma funcional, tamanho amostral, riqueza de features, dimensionalidade da saída). O profissional sênior diagnostica esse regime antes de escolher a ferramenta.**

Esta tese é demonstrada empiricamente — não apenas afirmada — através de comparações estatisticamente fundamentadas com validação cruzada e intervalos de confiança bootstrap em cada sessão.

## Motivação e Inspiração

O projeto tem como inspiração direta o pacote [`fptools`](https://github.com/RichterV/fptools) (Vinicius Richter, M.Sc.), referência na comunidade florestal Python brasileira para processamento de inventários, ajuste de modelos hipsométricos e volumétricos, classificação de sítio e prognose de produção.

A presente obra propõe-se a **estender** o escopo daquele material em três frentes:

1. **Modelagem com Redes Neurais.** Implementação sistemática de arquiteturas MLP, CNN e U-Net em PyTorch para tarefas tradicionalmente abordadas por regressão paramétrica, com **comparação rigorosa e estatisticamente fundamentada** contra os modelos clássicos.
2. **Diagnóstico metodológico explícito.** Cada sessão documenta não apenas o desempenho dos modelos, mas também as razões pelas quais um paradigma supera o outro em cada regime experimental (forma funcional, escassez de dados, riqueza de features, estrutura espacial).
3. **Análises gráficas e estatísticas robustas.** Inclusão de painéis de diagnóstico (resíduos, QQ-plots, curvas de aprendizado, matrizes de confusão, intervalos de confiança bootstrap, métricas de segmentação IoU/Dice) e visualizações geoespaciais aprofundadas.

## Diferenciais Metodológicos

| Aspecto | Abordagem |
|---|---|
| Reprodutibilidade | Seeds determinísticas, configurações YAML versionadas, ambiente isolado via `pyproject.toml` |
| Validação | Suíte de **223 testes automatizados** com pytest, cobertura monitorada |
| Engenharia de software | Estrutura `src/`-layout, linting com Ruff, formatação com Black, pre-commit hooks |
| Avaliação estatística | Validação cruzada k-fold + bootstrap para intervalos de confiança em todas as comparações |
| Honestidade científica | Resultados reportados independentemente do paradigma favorecido; análise das causas de vitória/derrota do ML em cada regime |
| Documentação | Docstrings completas (estilo Google), referências bibliográficas inline, exemplos doctest |

## Arquitetura do Projeto

```
forest-py-estudos/
├── src/forestpy/             Pacote Python instalável
│   ├── data/                 Loaders, geradores tabulares e de chips raster
│   │   ├── loaders.py        Dataset principal (PEF Vinhedo sintético)
│   │   ├── chips.py          Chips multi-banda para classificação
│   │   └── canopy_chips.py   Chips com máscaras para segmentação
│   ├── dendrometria/         Volume, hipsometria, ajuste paramétrico
│   ├── inventario/           Amostragem (AAS, AE) e distribuição diamétrica (Weibull)
│   ├── sitio/                Curvas de crescimento e índices de produtividade
│   ├── ml/                   Métricas, avaliação, MLP e CNN em PyTorch
│   │   ├── metrics.py        Regressão, classificação e segmentação (IoU, Dice)
│   │   ├── evaluation.py     k-fold CV + bootstrap
│   │   ├── preprocessing.py  Normalização reproduzível
│   │   ├── encoders.py       One-Hot Encoding
│   │   ├── mlp/              MLPRegressor, MLPClassifier, trainers
│   │   └── cnn/              SimpleCNN, CNNTrainer, UNet, UNetTrainer
│   ├── viz/                  Toolkit de visualização (matplotlib, seaborn)
│   │   ├── eda.py, diagnostics.py
│   │   ├── classification.py
│   │   └── segmentation.py   Visualizações tríplices chip/observado/predito
│   └── utils/                Configuração, logging, I/O, reprodutibilidade
├── notebooks/                12 sessões cronológicas completas
├── data/                     raw / interim / processed / external
├── models/                   Pesos de modelos treinados
├── reports/                  Figuras, tabelas e análises técnicas em Markdown
├── configs/                  Hiperparâmetros YAML por experimento
├── scripts/                  CLIs reproduzíveis
└── tests/                    Suíte pytest (223 testes)
```

A separação entre o pacote (`src/forestpy/`) e os notebooks consumidores garante que cada sessão didática utilize as mesmas funções testadas que estariam em produção, evitando a duplicação de código típica de projetos exclusivamente em notebook.

## Trilha Didática Completa

| # | Tema | Métodos Principais |
|:---:|---|---|
| 01 | Introdução aos dados florestais | Contextualização, dicionário de variáveis |
| 02 | Análise exploratória de dados (EDA) | Distribuições, correlações, detecção de outliers |
| 03 | Dendrometria clássica | Ajuste de Schumacher-Hall e Spurr via mínimos quadrados |
| 04 | Inventário e amostragem | AAS e AE (Cochran 1977) com FPC e suficiência amostral |
| 05 | Volumetria clássica — baseline | Validação cruzada 5-fold e bootstrap para IC |
| 06 | Volumetria com Redes Neurais (MLP) | MLP em PyTorch, comparação estatística com baseline |
| 07 | Hipsometria com Redes Neurais | MLP multi-feature (DAP + idade + classe) |
| 08 | Classificação de sítio via Deep Learning | MLP multiclasse vs Árvore vs Majoritário; análise confiança/cobertura |
| 09 | Distribuição diamétrica: Weibull vs MLP | Modelo paramétrico vs neural multi-output em prognose |
| 10 | CNN para sensoriamento remoto | Chips raster 4-bandas, MLP em features vs CNN end-to-end |
| 11 | U-Net para segmentação de copas | Encoder-decoder com skip connections, BCE+Dice loss |
| 12 | Relatório executivo final | Síntese consolidada, padrões metodológicos, recomendações práticas |

## Resultados Consolidados

Um dos diferenciais centrais deste projeto é o **diagnóstico metodológico explícito** em cada sessão. Em vez de assumir que redes neurais superam modelos clássicos por padrão, cada comparação é conduzida em condições rigorosas (validação cruzada, bootstrap, critério estatístico explícito) e o resultado é discutido **independentemente do paradigma favorecido**.

| Sessão | Tarefa | Vencedor | Razão Estatística |
|:---:|---|---|---|
| 06 | Volumetria (DAP+H → V) | Schumacher-Hall | Forma funcional **correta** — paramétrico é o estimador de máxima verossimilhança |
| 07 | Hipsometria (DAP → H) | **MLP multi-feature** | Vantagem de **53% no RMSE** ao incorporar idade e classe de sítio |
| 08 | Classificação tabular de sítio | Empate (Árvore ≈ MLP) | Dados tabulares moderados, conforme Shwartz-Ziv & Armon (2022) |
| 09 | Distribuição diamétrica | Weibull | Forma adequada + escassez de dados (30 parcelas) |
| 10 | Sensoriamento remoto (chips) | MLP em features manuais | Espectros já discriminam; CNN exige mais dados |
| 11 | Segmentação semântica | **U-Net** | **Vantagem de +0.72 no IoU** — saída espacial densa é estrutural |

### Padrão observado

Em **três sessões** o paradigma clássico venceu (06, 09, 10) — quando a forma funcional é adequada, dados são escassos ou features bem-engenheiradas. Em **uma sessão** houve empate técnico (08). Em **duas sessões** o ML venceu (07, 11) — quando há mais informação no input ou a saída é espacialmente estruturada.

A magnitude dos ganhos é reveladora: a maior derrota do ML foi na volumetria (-120%), enquanto a maior vitória foi na segmentação (+400%). **As magnitudes refletem a importância do regime estrutural, não da arquitetura.**

> A síntese metodológica está sumarizada em uma frase: *"Modelos paramétricos vencem quando a forma funcional é correta e/ou os dados são escassos; modelos neurais vencem quando há mais informação no input ou a saída é estruturada espacialmente."*

## Requisitos e Instalação

### Pré-requisitos

- Python **3.12.x**
- Git
- (Opcional) GPU com suporte CUDA para acelerar treinamento de CNNs e U-Net

### Instalação

```bash
git clone https://github.com/PedroLuizskt/forest-py-estudos.git
cd forest-py-estudos

# Criação de ambiente virtual isolado
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows (PowerShell)
# source .venv/bin/activate      # Linux/macOS

# Instalação em modo editável com dependências de desenvolvimento
pip install -e ".[dev]"

# PyTorch (CPU — para GPU, ajuste o índice conforme docs.pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Registro do kernel para uso em Jupyter
python -m ipykernel install --user --name forestpy --display-name "Python (forestpy)"
```

### Verificação

```bash
python -c "import forestpy; print(forestpy.__version__)"
pytest -v --no-cov          # 223 testes, ~10s (sem U-Net) ou ~2min (suite completa)
```

## Utilização

### Notebooks didáticos

```bash
jupyter lab notebooks/
```

As sessões devem ser executadas em ordem cronológica (01 → 12) para que o encadeamento conceitual e os artefatos intermediários (modelos ajustados, métricas) estejam disponíveis nas sessões subsequentes.

### Uso programático do pacote

```python
from forestpy.data.loaders import load_pef_vinhedo
from forestpy.dendrometria.fitting import fit_model, compare_models
from forestpy.ml.evaluation import kfold_cv, bootstrap_metric
from forestpy.ml.mlp import MLPRegressor, MLPTrainer
from forestpy.ml.cnn import UNet, UNetTrainer
from forestpy.utils import set_seed

set_seed(42)
df = load_pef_vinhedo(synthetic_fallback=True)

# Ajuste paramétrico clássico
resultado = fit_model("schumacher_hall", df["volume"], df["dap"], df["h"])
print(resultado.summary())

# Segmentação com U-Net
from forestpy.data.canopy_chips import generate_segmentation_chips
ds = generate_segmentation_chips(n_chips=120, chip_size=64, seed=42)
model = UNet(in_channels=4, out_channels=1, chip_size=64)
trainer = UNetTrainer(model, learning_rate=1e-3)
# trainer.fit(...) — ver notebook 11
```

### Execução via Makefile

```bash
make test         # pytest com cobertura
make lint         # análise estática via Ruff
make format       # formatação automática (Black + Ruff --fix)
make validate     # pipeline completo de validação
make notebooks    # execução automatizada de todos os notebooks
```

## Validação e Qualidade de Código

A suíte de testes cobre os módulos críticos do pacote:

| Módulo | Testes |
|---|:---:|
| `data.loaders` | 7 |
| `data.chips` (sensoriamento remoto) | 13 |
| `data.canopy_chips` (máscaras para segmentação) | 7 |
| `dendrometria.volume` | 16 |
| `dendrometria.hipsometria` | 18 |
| `dendrometria.fitting` | 10 |
| `inventario.amostragem` | 19 |
| `inventario.distribuicao` (Weibull) | 15 |
| `ml.metrics` (regressão + classificação + segmentação) | 36 |
| `ml.evaluation` (k-fold + bootstrap) | 12 |
| `ml.preprocessing` | 7 |
| `ml.encoders` | 8 |
| `ml.mlp` (regressor + classifier) | 19 |
| `ml.cnn` (SimpleCNN + U-Net) | 26 |
| `viz` (style, eda, diagnostics, trees) | 10 |
| **Total** | **223** |

Padrões adotados:

- **Linting**: Ruff configurado para Python 3.12 com regras Pyflakes, pycodestyle, isort, pyupgrade e bugbear.
- **Formatação**: Black com `line-length=100`.
- **Pre-commit**: hooks automáticos para limpeza de trailing whitespace, validação de YAML/TOML e formatação.
- **Type hints**: utilizados em todas as assinaturas públicas.

## Reprodutibilidade

Para garantir a reprodução fiel dos resultados:

1. **Seeds determinísticas** são fixadas via `forestpy.utils.set_seed(42)` no início de cada notebook e script.
2. **Configurações de experimentos** ficam armazenadas em `configs/*.yaml` e são referenciadas explicitamente nos *model cards*.
3. **Dependências travadas** podem ser geradas com `pip freeze > requirements.lock` ao final de cada sessão.
4. **Dados sintéticos** são gerados com seed controlada, garantindo que qualquer execução produza o mesmo dataset.
5. **Geradores estruturais** (loaders, chips, canopy_chips) são determinísticos por construção e parametrizam explicitamente todas as fontes de aleatoriedade.

## Referências

Bailey, R. L., & Dell, T. R. (1973). Quantifying diameter distributions with the Weibull function. *Forest Science*, 19(2), 97–104.

Box, G. E. P. (1976). Science and statistics. *Journal of the American Statistical Association*, 71(356), 791–799.

Campos, J. C. C., & Leite, H. G. (2017). *Mensuração Florestal: Perguntas e Respostas* (5ª ed.). Editora UFV.

Cochran, W. G. (1977). *Sampling Techniques* (3ª ed.). John Wiley & Sons.

Curtis, R. O. (1967). Height-diameter and height-diameter-age equations for second-growth Douglas-fir. *Forest Science*, 13(4), 365–375.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer.

Husch, B., Beers, T. W., & Kershaw, J. A., Jr. (2003). *Forest Mensuration* (4ª ed.). John Wiley & Sons.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3D Vision (3DV)*.

Péllico Netto, S., & Brena, D. A. (1997). *Inventário Florestal*. Editora Universitária UFPR.

Richter, V. (2024). *fptools: Forest Python Tools* [Software]. Disponível em: https://github.com/RichterV/fptools

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, LNCS 9351, 234–241.

Schumacher, F. X., & Hall, F. S. (1933). Logarithmic expression of timber-tree volume. *Journal of Agricultural Research*, 47(9), 719–734.

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion*, 81, 84–90.

Spurr, S. H. (1952). *Forest Inventory*. Ronald Press.

Weinstein, B. G., Marconi, S., Bohlman, S., Zare, A., & White, E. (2019). Individual tree-crown detection in RGB imagery using semi-supervised deep learning neural networks. *Remote Sensing*, 11(11), 1309.

## Citação

Caso este material seja útil em trabalhos acadêmicos ou profissionais, sugere-se a citação:

```bibtex
@software{vazdemelo2026forestpyestudos,
  author       = {Vaz de Melo, Pedro Luiz R.},
  title        = {forest-py-estudos: Aplicação de Machine Learning e Redes Neurais à Engenharia Florestal},
  year         = {2026},
  url          = {https://github.com/PedroLuizskt/forest-py-estudos},
  note         = {Versão 0.1.0}
}
```

## Autor

**Pedro Luiz R. Vaz de Melo**
Engenheiro Florestal (UFSJ, 2025) | Cientista de Dados Geoespaciais

- GitHub: [@PedroLuizskt](https://github.com/PedroLuizskt)
- Email: pedroschuldiner@outlook.com

## Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para os termos completos.

---

> *"All models are wrong, but some are useful."* — George Box (1976)
