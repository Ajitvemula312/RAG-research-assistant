import argparse

from app.ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    args = parser.parse_args()
    pipeline = IngestionPipeline()
    for url in args.urls:
        print(f"ingesting {url}: {len(pipeline.ingest_url(url))} chunks")


if __name__ == "__main__":
    main()