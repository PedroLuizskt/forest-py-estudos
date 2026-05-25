# 🤖 Modelos Treinados

Pesos e artefatos de modelos. Arquivos binários ficam fora do Git via `.gitignore`.

| Pasta        | Conteúdo                                        |
|--------------|-------------------------------------------------|
| `classical/` | scikit-learn / XGBoost (`.pkl` via joblib)      |
| `mlp/`       | Redes neurais densas PyTorch (`.pt`)            |
| `cnn/`       | Redes convolucionais PyTorch (`.pt`)            |

## Convenção de nomes

```
<tarefa>_<arquitetura>_v<versao>.<ext>
```

Exemplos:
- `volumetria_mlp_v1.pt`
- `copas_unet_v2.pt`
- `sitio_rf_v1.pkl`

Cada modelo treinado deve gerar um **Model Card** em `reports/analytical/` com:
- Métricas (validação cruzada)
- Hiperparâmetros (referência ao YAML em `configs/`)
- Comparação com baseline
