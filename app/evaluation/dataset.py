from pathlib import Path

from app.schemas.models import EvaluationRecord


def load_dataset(path: str | Path) -> list[EvaluationRecord]:
    return [EvaluationRecord.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]