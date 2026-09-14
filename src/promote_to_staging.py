"""
promote_to_staging.py

Promove uma versão do modelo registrado no MLflow Model Registry
para o estágio "Staging".

Duas formas de uso:
  - manual:      --version 2            (promove a versão informada)
  - automática:  sem --version          (promove a versão com a melhor
                                          métrica entre as já registradas)
"""

import argparse

from mlflow import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promove uma versão do modelo para um estágio do registry."
    )
    parser.add_argument("--model_name", default="wine-classifier")
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Número da versão a promover. Se omitido, a versão com a "
        "melhor métrica (--metric) é escolhida automaticamente.",
    )
    parser.add_argument(
        "--metric",
        default="accuracy",
        help="Métrica usada para escolher a melhor versão quando --version não é informado.",
    )
    parser.add_argument("--stage", default="Staging")
    return parser.parse_args()


def find_best_version(
    client: MlflowClient, model_name: str, metric_key: str
) -> tuple[int, float]:
    """Percorre as versões registradas e retorna a de maior valor na métrica dada."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise SystemExit(f"Nenhuma versão registrada para o modelo '{model_name}'.")

    best_version = None
    best_value = float("-inf")
    for mv in versions:
        run = client.get_run(mv.run_id)
        value = run.data.metrics.get(metric_key)
        if value is not None and value > best_value:
            best_value = value
            best_version = int(mv.version)

    if best_version is None:
        raise SystemExit(
            f"Nenhuma versão de '{model_name}' possui a métrica '{metric_key}' registrada."
        )

    return best_version, best_value


def main() -> None:
    args = parse_args()
    client = MlflowClient()

    if args.version is not None:
        version = args.version
        print(f"Versão informada manualmente: v{version}")
    else:
        version, value = find_best_version(client, args.model_name, args.metric)
        print(f"Melhor versão por {args.metric}: v{version} ({args.metric}={value:.4f})")

    client.transition_model_version_stage(
        name=args.model_name,
        version=version,
        stage=args.stage,
        archive_existing_versions=True,
    )

    print(f"{args.model_name} v{version} → {args.stage}")


if __name__ == "__main__":
    main()

