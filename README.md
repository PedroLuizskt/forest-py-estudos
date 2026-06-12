# forest-py-estudos

**Aplicação de Machine Learning e Redes Neurais à Engenharia Florestal: estudo aprofundado em Python**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-197%20passing-success)](tests/)
[![Code style](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)

---

## Sumário

- [Resumo](#resumo)
- [Motivação e Inspiração](#motivação-e-inspiração)
- [Diferenciais Metodológicos](#diferenciais-metodológicos)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Plano de Estudos e Estado Atual](#plano-de-estudos-e-estado-atual)
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

Este repositório consolida um estudo aplicado de **Machine Learning e Deep Learning** ao domínio da **Engenharia Florestal**, com ênfase em mensuração florestal, inventário, classificação de sítio, distribuição diamétrica e análises geoespaciais. O material é estruturado como um pacote Python instalável (`forestpy`), acompanhado de doze sessões didáticas em formato de notebook que documentam o raciocínio metodológico, desde a análise exploratória até a modelagem com redes neurais densas (MLP) e convolucionais (CNN).

O dataset principal é uma base sintética com distribuições estatísticas realistas de plantios de *Eucalyptus grandis*, compatível com o formato do dataset PEF Vinhedo (SP). A escolha por dados sintéticos garante reprodutibilidade total, independência de fontes externas e controle sobre os mecanismos geradores — permitindo investigar empiricamente em quais regimes os modelos paramétricos clássicos superam o aprendizado de máquina, e vice-versa, sem comprometer a validade metodológica das análises.

## Motivação e Inspiração

O projeto tem como inspiração direta o pacote [`fptools`](https://github.com/RichterV/fptools) (Vinicius Richter, M.Sc.), referência na comunidade florestal Python brasileira para processamento de inventários, ajuste de modelos hipsométricos e volumétricos, classificação de sítio e prognose de produção.

A presente obra propõe-se a **estender** o escopo daquele material em três frentes:

1. **Modelagem com Redes Neurais.** Implementação sistemática de arquiteturas MLP e CNN em PyTorch para tarefas tradicionalmente abordadas por regressão paramétrica, com **comparação rigorosa e estatisticamente fundamentada** contra os modelos clássicos.
2. **Diagnóstico metodológico explícito.** Cada sessão documenta não apenas o desempenho dos modelos, mas também as razões pelas quais um paradigma supera o outro em cada regime experimental (forma funcional, escassez de dados, riqueza de features, estrutura espacial).
3. **Análises gráficas robustas.** Inclusão de painéis de diagnóstico (resíduos, QQ-plots, curvas de aprendizado, matrizes de confusão, intervalos de confiança bootstrap) e visualizações geoespaciais aprofundadas.

## Diferenciais Metodológicos

| Aspecto | Abordagem |
|---|---|
| Reprodutibilidade | Seeds determinísticas, configurações YAML versionadas, ambiente isolado via `pyproject.toml` |
| Validação | Suíte de **197 testes automatizados** com pytest, cobertura monitorada |
| Engenharia de software | Estrutura `src/`-layout, linting com Ruff, formatação com Black, pre-commit hooks |
| Avaliação estatística | Validação cruzada k-fold + bootstrap para intervalos de confiança em todas as comparações |
| Honestidade científica | Resultados reportados independentemente do resultado favorável; análise das causas de vitória/derrota do ML em cada regime |
| Documentação | Docstrings completas (estilo Google), referências bibliográficas inline, exemplos doctest |

## Arquitetura do Projeto

```
forest-py-estudos/
├── src/forestpy/             Pacote Python instalável
│   ├── data/                 Loaders, validadores, geradores de dados e chips raster
│   ├── dendrometria/         Volume, hipsometria, ajuste de modelos paramétricos
│   ├── inventario/           Amostragem (AAS, estratificada) e distribuição diamétrica (Weibull)
│   ├── sitio/                Curvas de crescimento e índices de produtividade
│   ├── ml/                   Métricas, avaliação (CV, bootstrap), MLP e CNN em PyTorch
│   │   ├── metrics.py        Regressão e classificação
│   │   ├── evaluation.py     k-fold CV + bootstrap
│   │   ├── preprocessing.py  Normalização reproduzível
│   │   ├── encoders.py       One-Hot Encoding
│   │   ├── mlp/              MLPRegressor, MLPClassifier, trainers
│   │   └── cnn/              SimpleCNN, CNNTrainer
│   ├── viz/                  Toolkit de visualização (matplotlib, seaborn)
│   └── utils/                Configuração, logging, I/O, reprodutibilidade
├── notebooks/                12 sessões cronológicas de estudo
├── data/                     raw / interim / processed / external
├── models/                   Pesos de modelos treinados
├── reports/                  Figuras, tabelas e análises técnicas em Markdown
├── configs/                  Hiperparâmetros YAML por experimento
├── scripts/                  CLIs reproduzíveis
└── tests/                    Suíte pytest (197 testes)
```

A separação entre o pacote (`src/forestpy/`) e os notebooks consumidores garante que cada sessão didática utilize as mesmas funções testadas que estariam em produção, evitando a duplicação de código típica de projetos exclusivamente em notebook.

## Plano de Estudos e Estado Atual

| # | Tema | Métodos Principais | Status |
|:---:|---|---|:---:|
| 01 | Introdução aos dados florestais | Contextualização, dicionário de variáveis | ✓ |
| 02 | Análise exploratória de dados (EDA) | Distribuições, correlações, detecção de outliers | ✓ |
| 03 | Dendrometria clássica | Ajuste de Schumacher-Hall e Spurr via mínimos quadrados | ✓ |
| 04 | Inventário e amostragem | Amostragem aleatória simples e estratificada (Cochran 1977) | ✓ |
| 05 | Volumetria clássica — baseline | Validação cruzada 5-fold e bootstrap para IC | ✓ |
| 06 | Volumetria com Redes Neurais (MLP) | MLP em PyTorch, comparação estatística com baseline | ✓ |
| 07 | Hipsometria com Redes Neurais | MLP multi-feature (DAP + idade + classe) | ✓ |
| 08 | Classificação de sítio via Deep Learning | MLP multiclasse vs Árvore de Decisão vs Majoritário | ✓ |
| 09 | Distribuição diamétrica: Weibull vs MLP | Modelo paramétrico vs neural multi-output | ✓ |
| 10 | CNN para sensoriamento remoto | Chips raster 4-bandas, MLP em features vs CNN end-to-end | ✓ |
| 11 | U-Net para segmentação de copas | Segmentação semântica pixel-a-pixel | em desenvolvimento |
| 12 | Relatório executivo final | Síntese comparativa de todos os modelos | em desenvolvimento |

## Resultados Consolidados

Um dos diferenciais centrais deste projeto é o **diagnóstico metodológico explícito** em cada sessão. Em vez de assumir que redes neurais superam modelos clássicos por padrão, cada comparação é conduzida em condições rigorosas (validação cruzada, bootstrap, critério estatístico explícito) e o resultado é discutido **independentemente do paradigma favorecido**.

| Sessão | Tarefa | Vencedor | Razão Estatística |
|:---:|---|---|---|
| 06 | Volumetria (DAP+H → V) | Schumacher-Hall | Forma funcional **correta** — paramétrico é o estimador de máxima verossimilhança |
| 07 | Hipsometria (DAP → H) | **MLP multi-feature** | Vantagem de **53% no RMSE** ao incorporar idade e classe de sítio |
| 08 | Classificação tabular de sítio | Empate (Árvore ≈ MLP) | Dados tabulares moderados, conforme Shwartz-Ziv & Armon (2022) |
| 09 | Distribuição diamétrica | Weibull | Forma adequada + escassez de dados (30 parcelas) |
| 10 | Sensoriamento remoto (chips) | MLP em features manuais | Espectros já discriminam; CNN exige mais dados |

O padrão emergente é coerente com a literatura: **o aprendizado profundo brilha em regimes específicos** — abundância de dados, alta dimensionalidade de entrada, estrutura espacial ou textual rica, e baixa adequação de formas paramétricas conhecidas. Em problemas tabulares com forma funcional bem estabelecida e dados moderados, modelos clássicos frequentemente vencem ou empatam. **Diagnosticar o regime correto é a habilidade central do cientista de dados aplicado.**

## Requisitos e Instalação

### Pré-requisitos

- Python **3.12.x**
- Git
- (Opcional) GPU com suporte CUDA para acelerar treinamento de CNNs

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

# PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Registro do kernel para uso em Jupyter
python -m ipykernel install --user --name forestpy --display-name "Python (forestpy)"
```

### Verificação

```bash
python -c "import forestpy; print(forestpy.__version__)"
pytest -v --no-cov          # 197 testes, ~10s
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
from forestpy.utils import set_seed

set_seed(42)
df = load_pef_vinhedo(synthetic_fallback=True)

# Ajuste paramétrico clássico
resultado = fit_model("schumacher_hall", df["volume"], df["dap"], df["h"])
print(resultado.summary())

# Comparação entre modelos
ranking = compare_models(
    ["schumacher_hall", "spurr"],
    df["volume"], df["dap"], df["h"],
)
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
| `dendrometria.volume` | 16 |
| `dendrometria.hipsometria` | 18 |
| `dendrometria.fitting` | 10 |
| `inventario.amostragem` | 19 |
| `inventario.distribuicao` (Weibull) | 15 |
| `ml.metrics` (regressão + classificação) | 28 |
| `ml.evaluation` (k-fold + bootstrap) | 12 |
| `ml.preprocessing` | 7 |
| `ml.encoders` | 8 |
| `ml.mlp` (regressor + classifier) | 19 |
| `ml.cnn` | 12 |
| `viz` (style, eda, diagnostics, trees) | 10 |
| **Total** | **197** |

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
5. **Geradores estruturais** (loaders, chips) são determinísticos por construção e parametrizam explicitamente todas as fontes de aleatoriedade.

## Referências

Bailey, R. L., & Dell, T. R. (1973). Quantifying diameter distributions with the Weibull function. *Forest Science*, 19(2), 97–104.

Campos, J. C. C., & Leite, H. G. (2017). *Mensuração Florestal: Perguntas e Respostas* (5ª ed.). Editora UFV.

Cochran, W. G. (1977). *Sampling Techniques* (3ª ed.). John Wiley & Sons.

Curtis, R. O. (1967). Height-diameter and height-diameter-age equations for second-growth Douglas-fir. *Forest Science*, 13(4), 365–375.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2ª ed.). Springer.

Husch, B., Beers, T. W., & Kershaw, J. A., Jr. (2003). *Forest Mensuration* (4ª ed.). John Wiley & Sons.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

Péllico Netto, S., & Brena, D. A. (1997). *Inventário Florestal*. Editora Universitária UFPR.

Richter, V. (2024). *fptools: Forest Python Tools* [Software]. Disponível em: https://github.com/RichterV/fptools

Schumacher, F. X., & Hall, F. S. (1933). Logarithmic expression of timber-tree volume. *Journal of Agricultural Research*, 47(9), 719–734.

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion*, 81, 84–90.

Spurr, S. H. (1952). *Forest Inventory*. Ronald Press.

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

> *"As coisas só acontecem se os dados que coletamos puderem informar e inspirar aqueles que estão em posição de fazer a diferença."* — Dr. Mike Schmoker
