import httpx


class LLMProvider:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaLLM(LLMProvider):
    def __init__(self, model: str = "llama3", temperature: float = 0.1, timeout: float = 60.0, base_url: str = "http://localhost:11434") -> None:
        self.model, self.temperature, self.timeout, self.base_url = model, temperature, timeout, base_url

    def generate(self, prompt: str) -> str:
        response = httpx.post(f"{self.base_url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": self.temperature}}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["response"]


class DeterministicLLM(LLMProvider):
    """A safe test provider; it only quotes supplied evidence."""

    def generate(self, prompt: str) -> str:
        evidence = prompt.split("EVIDENCE:\n", 1)[-1].strip()
        return "Insufficient evidence to answer confidently.\n\n" + evidence[:600]