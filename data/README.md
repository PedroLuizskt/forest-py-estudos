# 📂 Diretório de Dados

Estrutura inspirada no [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

| Pasta        | Conteúdo                                                                            |
|--------------|-------------------------------------------------------------------------------------|
| `raw/`       | **Imutável.** Dados originais como recebidos da fonte. Nunca editar diretamente.    |
| `interim/`   | Dados em estágios intermediários de transformação (limpeza, junções).               |
| `processed/` | Dataset final pronto para modelagem (`.parquet` recomendado).                       |
| `external/`  | Fontes externas (MapBiomas, SICAR, IBGE/SIDRA) baixadas via scripts ou APIs.        |

## Fontes esperadas

- **PEF Vinhedo (SP):** plantios de *Eucalyptus grandis* — dataset principal do projeto.
- **MapBiomas:** coleção de uso/cobertura do solo (raster 30 m) — sessão 10.
- **SICAR:** geometrias de imóveis rurais cadastrados — sessão 11 (opcional).

## Versionamento

Arquivos pesados (`.csv`, `.tif`, `.gpkg`) ficam **fora** do Git via `.gitignore`. Recomenda-se [DVC](https://dvc.org) para versionar grandes datasets quando o projeto crescer.
