"""
fix_mlruns_paths.py

O MLflow grava caminhos absolutos dentro dos meta.yaml de mlruns/
(artifact_uri, artifact_location, source, storage_location). Como
mlruns/ está versionado nesta demo (ver README), esses caminhos
apontam para a máquina onde o treino rodou originalmente — não para
a máquina atual. Este script reescreve todos esses caminhos para
apontar para o mlruns/ local, a partir da raiz do repositório.

Rodar sempre que: clonar/copiar o repositório em uma máquina nova,
ou como primeiro passo antes do smoke test em um runner de CI —
por isso ele entra no workflow do GitHub Actions antes do
smoke_test.py.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = REPO_ROOT / "mlruns"

FIELD_PATTERN = re.compile(
    r"^(artifact_uri|artifact_location|source|storage_location):\s*(.*?/mlruns)(/.*)?$",
    re.MULTILINE,
)


def fix_file(path: Path) -> bool:
    original = path.read_text()

    def replace(match: re.Match) -> str:
        key = match.group(1)
        suffix = match.group(3) or ""
        return f"{key}: {MLRUNS_DIR}{suffix}"

    updated = FIELD_PATTERN.sub(replace, original)
    if updated != original:
        path.write_text(updated)
        return True
    return False


def main() -> None:
    if not MLRUNS_DIR.exists():
        print(f"mlruns/ não encontrado em {MLRUNS_DIR} — nada para corrigir.")
        return

    changed = 0
    for meta_file in MLRUNS_DIR.rglob("meta.yaml"):
        if fix_file(meta_file):
            changed += 1

    print(f"Caminhos corrigidos em {changed} arquivo(s), apontando para {MLRUNS_DIR}")


if __name__ == "__main__":
    main()
