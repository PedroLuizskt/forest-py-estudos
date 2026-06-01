# forest-py-estudos

**Aplicação de Machine Learning e Redes Neurais à Engenharia Florestal: estudo aprofundado em Python**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-81%20passing-success)](tests/)
[![Code style](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)

---

## Sumário

- [Resumo](#resumo)
- [Motivação e Inspiração](#motivação-e-inspiração)
- [Diferenciais Metodológicos](#diferenciais-metodológicos)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Plano de Estudos](#plano-de-estudos)
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

Este repositório consolida um estudo aplicado de **Machine Learning e Deep Learning** ao domínio da **Engenharia Florestal**, com ênfase em mensuração florestal, inventário, classificação de sítio e análises geoespaciais. O material é estruturado como um pacote Python instalável (`forestpy`), acompanhado de doze sessões didáticas em formato de notebook que documentam o raciocínio metodológico, desde a análise exploratória até a modelagem com redes neurais densas (MLP) e convolucionais (CNN).

O dataset principal é uma base sintética com distribuições estatísticas realistas de plantios de *Eucalyptus grandis*, compatível com o formato do dataset PEF Vinhedo (SP). A escolha por dados sintéticos garante reprodutibilidade total e independência de fontes externas, sem comprometer a validade metodológica das análises.

## Motivação e Inspiração

O projeto tem como inspiração direta o pacote [`fptools`](https://github.com/RichterV/fptools) (Vinicius Richter, M.Sc.), referência na comunidade florestal Python brasileira para processamento de inventários, ajuste de modelos hipsométricos e volumétricos, classificação de sítio e prognose de produção.

A presente obra propõe-se a **estender** o escopo daquele material em três frentes:

1. **Modelagem com Redes Neurais.** Implementação sistemática de arquiteturas MLP e CNN em PyTorch para tarefas tradicionalmente abordadas por regressão paramétrica, com comparação direta contra os modelos clássicos.
2. **Relatórios analíticos aprofundados.** Cada sessão produz documentação técnica em Markdown contendo diagnósticos estatísticos, comparações de desempenho e discussão crítica dos resultados.
3. **Análises gráficas robustas.** Inclusão de painéis de diagnóstico (resíduos, QQ-plots, curvas de aprendizado, interpretabilidade via Grad-CAM) e visualizações geoespaciais.

## Diferenciais Metodológicos

| Aspecto | Abordagem |
|---|---|
| Reprodutibilidade | Seeds determinísticas, configurações YAML versionadas, ambiente isolado via `pyproject.toml` |
| Validação | Suíte de 81 testes automatizados com pytest, cobertura monitorada |
| Engenharia de software | Estrutura `src/`-layout, linting com Ruff, formatação com Black, pre-commit hooks |
| Modelagem | Baselines clássicos sempre presentes para comparação justa com modelos neurais |
| Documentação | Docstrings completas (estilo Google), referências bibliográficas inline, exemplos doctest |

## Arquitetura do Projeto

```
forest-py-estudos/
├── src/forestpy/             Pacote Python instalável
│   ├── data/                 Loaders, validadores e geradores de dados
│   ├── dendrometria/         Volume, afilamento, hipsometria, sortimentos, ajuste
│   ├── inventario/           Amostragem e distribuições diamétricas
│   ├── sitio/                Curvas de crescimento e índices de produtividade
│   ├── ml/                   Métricas, MLP (PyTorch) e CNN
│   ├── viz/                  Toolkit de visualização (matplotlib, seaborn)
│   └── utils/                Configuração, logging, I/O, reprodutibilidade
├── notebooks/                12 sessões cronológicas de estudo
├── data/                     raw / interim / processed / external
├── models/                   Pesos de modelos treinados
├── reports/                  Figuras, tabelas e análises técnicas em Markdown
├── configs/                  Hiperparâmetros YAML por experimento
├── scripts/                  CLIs reproduzíveis
└── tests/                    Suíte pytest
```

A separação entre o pacote (`src/forestpy/`) e os notebooks consumidores garante que cada sessão didática utilize as mesmas funções testadas que estariam em produção, evitando a duplicação de código típica de projetos exclusivamente em notebook.

## Plano de Estudos

| Sessão | Tema | Métodos Principais |
|:---:|---|---|
| 01 | Introdução aos dados florestais | Contextualização, dicionário de variáveis |
| 02 | Análise exploratória de dados (EDA) | Distribuições, correlações, detecção de outliers |
| 03 | Dendrometria clássica | Ajuste de Schumacher-Hall e Spurr via mínimos quadrados |
| 04 | Inventário e amostragem | Amostragem aleatória simples e estratificada |
| 05 | Volumetria clássica — baseline | Regressão paramétrica e diagnóstico de resíduos |
| 06 | Volumetria com Redes Neurais (MLP) | MLP em PyTorch, comparação com baseline |
| 07 | Hipsometria com Redes Neurais | MLP multi-output por espécie e classe |
| 08 | Classificação de sítio via Deep Learning | MLP para classificação multiclasse |
| 09 | Distribuição diamétrica: Weibull vs. MLP | Comparação paramétrico-neural |
| 10 | CNN para sensoriamento remoto | Classificação de uso/cobertura, transfer learning |
| 11 | U-Net para segmentação de copas | Segmentação semântica de copas individuais |
| 12 | Relatório executivo final | Síntese comparativa de todos os modelos |

## Requisitos e Instalação

### Pré-requisitos

- Python **3.12.x**
- Git
- (Opcional) GPU com suporte CUDA para acelerar treinamento de CNNs (sessões 10–11)

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

# Registro do kernel para uso em Jupyter
python -m ipykernel install --user --name forestpy --display-name "Python (forestpy)"
```

### Verificação

```bash
python -c "import forestpy; print(forestpy.__version__)"
pytest -v --no-cov
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
from forestpy.viz.diagnostics import plot_residuals

df = load_pef_vinhedo(synthetic_fallback=True)

# Ajuste de Schumacher-Hall ao volume observado
resultado = fit_model("schumacher_hall", df["volume"], df["dap"], df["h"])
print(resultado.summary())

# Comparação entre modelos volumétricos
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
| `data.loaders` | 8 |
| `dendrometria.volume` | 16 |
| `dendrometria.hipsometria` | 18 |
| `dendrometria.fitting` | 10 |
| `ml.metrics` | 19 |
| `viz` (style, eda, diagnostics, trees) | 10 |
| **Total** | **81** |

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

## Referências

Husch, B., Beers, T. W., & Kershaw, J. A., Jr. (2003). *Forest Mensuration* (4ª ed.). John Wiley & Sons.

Campos, J. C. C., & Leite, H. G. (2017). *Mensuração Florestal: Perguntas e Respostas* (5ª ed.). Editora UFV.

Schumacher, F. X., & Hall, F. S. (1933). Logarithmic expression of timber-tree volume. *Journal of Agricultural Research*, 47(9), 719–734.

Spurr, S. H. (1952). *Forest Inventory*. Ronald Press.

Curtis, R. O. (1967). Height-diameter and height-diameter-age equations for second-growth Douglas-fir. *Forest Science*, 13(4), 365–375.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Richter, V. (2024). *fptools: Forest Python Tools* [Software]. Disponível em: https://github.com/RichterV/fptools

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

> *"In God we trust; all others must bring data."* — W. Edwards Deming
