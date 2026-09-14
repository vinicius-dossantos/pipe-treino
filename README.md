# pipe-treino

Pipeline de treino parametrizado, com tracking de experimentos e model
registry via MLflow. Demonstração ao vivo — dataset `load_wine`, nativo
do scikit-learn, escolhido por ser pequeno; o foco é a automação do
pipeline, não a qualidade do modelo.

## Estrutura

```
pipe-treino/
├── README.md
├── requirements.txt      # dependências fixadas (lockfile)
├── config.yaml            # hiperparâmetros e configuração, fora do código
├── .github/workflows/
│   └── deploy.yml          # CI/CD: smoke test do modelo em staging
└── src/
    ├── train.py              # treino + tracking + registro no MLflow
    ├── promote_to_staging.py # promove uma versão do modelo no registry
    ├── fix_mlruns_paths.py    # corrige caminhos absolutos do mlruns versionado
    └── smoke_test.py          # valida o modelo em staging (local e no CI)
```

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Rodar o pipeline

A partir da raiz do repositório:

```bash
python src/train.py
```

Treina o modelo com os parâmetros de `config.yaml`, loga parâmetros,
métricas e o artefato do modelo no MLflow, e registra uma nova versão
em `wine-classifier` no Model Registry.

## 3. Checkpoint de rastreabilidade — comparar duas execuções

Sobrescrevendo hiperparâmetros via linha de comando (sem editar o config):

```bash
python src/train.py --n_estimators 100 --max_depth 5
python src/train.py --n_estimators 5 --max_depth 1
```

Cada execução recebe um `run_id` próprio e uma nova versão do modelo.

## 4. Ver os resultados na UI do MLflow

```bash
mlflow ui --backend-store-uri mlruns
```

Abra `http://localhost:5000`, entre no experimento `wine-classifier` e
compare as execuções lado a lado — parâmetros, métricas e artefatos.

## 5. Promover a melhor versão para staging

**Automático** — escolhe a versão com a melhor métrica entre as já registradas:

```bash
python src/promote_to_staging.py
```

**Manual** — você escolhe a versão pelo número visto na UI do MLflow:

```bash
python src/promote_to_staging.py --version <numero_da_versao>
```

Por padrão a métrica usada é `accuracy`; para usar outra (ex: `f1_macro`):

```bash
python src/promote_to_staging.py --metric f1_macro
```


> Nota: `transition_model_version_stage` (estágios `Staging`/`Production`)
> está marcado como deprecated no MLflow a partir da versão 2.9, em favor
> de aliases e tags. Optei por manter estágios por serem mais didáticos
> para introduzir o conceito de registry; vale mencionar em aula que essa
> API deve mudar nas próximas versões do MLflow.

## Nota sobre `setuptools` no requirements.txt

O `mlflow==2.14.1` ainda depende de `pkg_resources`, que vem do
`setuptools` — mas versões recentes do `python -m venv` deixaram de
instalar `setuptools` por padrão dentro do venv. Por isso o
`requirements.txt` fixa `setuptools<81` explicitamente (a partir da 81,
o `pkg_resources` está sendo removido do `setuptools`). Sem essa linha,
`import mlflow` falha com `ModuleNotFoundError: No module named
'pkg_resources'` em ambientes onde o `setuptools` não vem pré-instalado.

## 6. Deploy automático (Parte 3)

O workflow `.github/workflows/deploy.yml` roda em todo push para `main`
(ou manualmente, via `workflow_dispatch`) e executa um **smoke test**:
carrega o modelo marcado como `Staging` e valida que ele gera previsões
válidas para um pequeno lote de exemplo.

Rodar o mesmo smoke test localmente, sem precisar do GitHub Actions:

```bash
python src/fix_mlruns_paths.py
python src/smoke_test.py
```

Se não houver nenhuma versão em `Staging`, o script falha com código de
saída 1 — é esse mesmo código de saída que faz o job do GitHub Actions
aparecer como falho.

### ⚠️ Ressalva importante sobre o `mlruns/` versionado

Para esta demonstração, `mlruns/` **está commitado no repositório** (o
`.gitignore` não ignora mais essa pasta) — é assim que o runner do
GitHub Actions, que não tem acesso à sua máquina, consegue enxergar o
mesmo registry que você usa localmente.

**Isso não é prática recomendada para um pipeline real.** Numa
arquitetura de produção, o tracking/registry do MLflow fica num
servidor compartilhado e acessível pela rede — por exemplo, um MLflow
gerenciado (Databricks, ou similar) ou um `tracking_uri` apontando para
um banco Postgres com artefatos em S3/GCS — nunca dentro do próprio
repositório git. Vale deixar isso explícito para a turma: aqui trocamos
"infraestrutura compartilhada" por "arquivo versionado" só para caber
numa aula, sem precisar de credenciais de nuvem.

O `src/fix_mlruns_paths.py` existe só por causa dessa gambiarra: o
MLflow grava caminho absoluto de máquina dentro dos `meta.yaml` do
`mlruns/`, então toda vez que esse repositório é clonado ou copiado
para um lugar novo (seu Mac, o runner do GitHub, a máquina de outra
pessoa), os caminhos gravados apontam para uma pasta que não existe
ali. O script reescreve esses caminhos para a pasta `mlruns/` local,
a partir de onde o repositório está agora — e roda tanto localmente
quanto como primeiro passo do workflow do GitHub Actions. Num
pipeline real, com um tracking server remoto, esse problema
simplesmente não existiria.
