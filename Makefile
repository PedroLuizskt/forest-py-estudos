# ══════════════════════════════════════════════════════════════
#  forest-py-estudos — Comandos de Desenvolvimento
# ══════════════════════════════════════════════════════════════

.PHONY: help install install-dev test lint format clean notebooks figures validate kernel

help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════╗"
	@echo "  ║  forest-py-estudos — Forestry × ML in Python     ║"
	@echo "  ╚══════════════════════════════════════════════════╝"
	@echo ""
	@echo "  📦 Setup"
	@echo "    make install        Instala o pacote em modo editável"
	@echo "    make install-dev    Instala com dependências de desenvolvimento"
	@echo "    make kernel         Registra o kernel Jupyter (Python forestpy)"
	@echo ""
	@echo "  🧪 Qualidade"
	@echo "    make test           Roda pytest com cobertura"
	@echo "    make lint           Roda ruff (linter)"
	@echo "    make format         Aplica black + ruff --fix"
	@echo "    make validate       Lint + test + execução dos notebooks"
	@echo ""
	@echo "  📓 Notebooks"
	@echo "    make notebooks      Executa todos os notebooks ponta-a-ponta"
	@echo "    make figures        Regenera figuras canônicas em reports/figures/"
	@echo ""
	@echo "  🧹 Limpeza"
	@echo "    make clean          Remove caches, builds e checkpoints"
	@echo ""

# ── Setup ───────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

kernel:
	python -m ipykernel install --user --name forestpy --display-name "Python (forestpy)"

# ── Qualidade ───────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

validate: lint test
	@echo "✅ Validação completa"

# ── Notebooks ───────────────────────────────────────────────────
notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

figures:
	python scripts/generate_figures.py

# ── Limpeza ─────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .ruff_cache .mypy_cache
	@echo "🧹 Caches limpos"
