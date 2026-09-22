from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.schemas.models import SourceMetadata


def load_web(url: str, timeout: float = 20.0) -> tuple[str, SourceMetadata]:
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for node in soup(["script", "style", "nav"]):
        node.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else urlparse(url).netloc
    return soup.get_text(" ", strip=True), SourceMetadata(document_id=url, title=title, source="web", url=url, chunk_id=url)