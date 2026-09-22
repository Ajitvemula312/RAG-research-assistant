import re

from app.schemas.models import Evidence, SourceMetadata


def chunk_text(text: str, metadata: SourceMetadata, chunk_size: int = 900, overlap: int = 120) -> list[Evidence]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        content = " ".join(words[start : start + chunk_size]).strip()
        if not content:
            continue
        page_part = f":p{metadata.page_number}" if metadata.page_number is not None else ""
        chunk_id = f"{metadata.document_id}{page_part}:c{start // step}"
        chunks.append(Evidence(text=content, metadata=metadata.model_copy(update={"chunk_id": chunk_id})))
        if start + chunk_size >= len(words):
            break
    return chunks