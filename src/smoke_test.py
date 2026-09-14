"""
smoke_test.py

Valida que o modelo marcado como Staging no registry carrega
corretamente e gera previsões válidas para um pequeno lote de exemplo.

Rodado tanto localmente quanto pelo pipeline de CI/CD (GitHub Actions)
como critério de aprovação antes de considerar o deploy bem-sucedido.
"""

import sys

import mlflow
from sklearn.datasets import load_wine

MODEL_NAME = "wine-classifier"
STAGE = "Staging"
VALID_CLASSES = {0, 1, 2}
SAMPLE_SIZE = 5


def main() -> int:
    mlflow.set_tracking_uri("mlruns")

    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    print(f"Carregando modelo de {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        print(f"FALHA: não foi possível carregar o modelo — {exc}")
        return 1

    dataset = load_wine(as_frame=True)
    sample = dataset.data.iloc[:SAMPLE_SIZE]

    predictions = model.predict(sample)

    if len(predictions) != SAMPLE_SIZE:
        print(f"FALHA: esperado {SAMPLE_SIZE} previsões, recebido {len(predictions)}")
        return 1

    invalid = [p for p in predictions if int(p) not in VALID_CLASSES]
    if invalid:
        print(f"FALHA: previsões fora do intervalo esperado: {invalid}")
        return 1

    print(f"OK — {SAMPLE_SIZE} previsões válidas: {list(predictions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
