import argparse
from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/papers")
    parser.add_argument("--manifest", default="data/web/source_urls.txt")
    args = parser.parse_args()
    pipeline = IngestionPipeline()
    files = sorted(Path(args.path).glob("*.pdf"))
    manifest = Path(args.manifest)
    urls = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()] if manifest.exists() else []
    for file in files:
        prefix = file.name.split("_", 1)[0]
        index = int(prefix) - 1 if prefix.isdigit() else -1
        source_url = urls[index] if 0 <= index < len(urls) else None
        print(f"ingesting {file}: {len(pipeline.ingest_pdf(file, source_url))} chunks")
    print(f"processed {len(files)} PDF documents")


if __name__ == "__main__":
    main()