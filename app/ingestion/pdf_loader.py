from pathlib import Path

from app.schemas.models import SourceMetadata


def load_pdf(path: str | Path, source_url: str | None = None) -> list[tuple[str, SourceMetadata]]:
    file = Path(path)
    from pypdf import PdfReader

    reader = PdfReader(str(file))
    document_id = file.stem
    return [(page.extract_text() or "", SourceMetadata(document_id=document_id, title=file.stem, source="pdf", url=source_url, chunk_id=f"{document_id}:{number}", page_number=number + 1)) for number, page in enumerate(reader.pages)]