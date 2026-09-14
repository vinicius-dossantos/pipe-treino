"""
train.py

Pipeline de treino parametrizado, com tracking de experimentos e
registro de modelo no MLflow Model Registry.

Dataset: load_wine (scikit-learn) — escolhido por ser pequeno e
nativo do scikit-learn. O foco deste script é a automação do
pipeline (parametrização, reprodutibilidade, tracking, registry),
não a qualidade do modelo em si.
"""

import argparse
import random

import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino parametrizado do classificador de vinhos."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Caminho do arquivo de configuração."
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Sobrescreve model.n_estimators do config (útil para comparar execuções ao vivo).",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Sobrescreve model.max_depth do config.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # overrides pontuais via linha de comando — permitem comparar
    # execuções ao vivo sem editar o config a cada rodada
    if args.n_estimators is not None:
        config["model"]["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        config["model"]["max_depth"] = args.max_depth

    set_seed(config["data"]["random_state"])

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    dataset = load_wine(as_frame=True)
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    with mlflow.start_run():
        model_params = config["model"]
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("random_state", config["data"]["random_state"])

        model = RandomForestClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            random_state=config["data"]["random_state"],
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=config["registry"]["model_name"],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"run_id={run_id}")
        print(f"accuracy={accuracy:.4f}  f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()
