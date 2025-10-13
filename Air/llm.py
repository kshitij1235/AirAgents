# Air/llm.py
from functools import lru_cache
from langchain_core.messages import BaseMessage


class LLM:
    HOT_PARAMS = {"temperature", "max_tokens", "num_ctx"}

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        deployment_name: str = None,
        azure_endpoint: str = None,
        api_version: str = "2024-02-01",
        num_ctx: int = 2048,
        **kwargs,
    ):
        self.provider = provider.lower()
        self.llm = None
        self.config = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "deployment_name": deployment_name,
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
            "num_ctx": num_ctx,
            **kwargs,
        }

        self.set_provider(self.provider, **self.config)

    # -------------------------------------------------------------------------
    # LRU cached provider builder
    # -------------------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=8)
    def _build_provider(provider: str, **kwargs):
        provider = provider.lower()

        if provider == "openai":
            from langchain_openai import AzureChatOpenAI, ChatOpenAI

            return ChatOpenAI(
                model=kwargs.get("model", "gpt-4o-mini"),
                api_key=kwargs.get("api_key"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )

        elif provider == "azure":
            return AzureChatOpenAI(
                deployment_name=kwargs.get("deployment_name"),
                api_key=kwargs.get("api_key"),
                azure_endpoint=kwargs.get("azure_endpoint"),
                api_version=kwargs.get("api_version", "2024-02-01"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                model=kwargs.get("model", "llama2"),
                temperature=kwargs.get("temperature", 0.7),
                num_ctx=kwargs.get("num_ctx", 2048),
            )

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=kwargs.get("model", "gemini-1.5-pro"),
                google_api_key=kwargs.get("api_key"),
                temperature=kwargs.get("temperature", 0.7),
                max_output_tokens=kwargs.get("max_tokens", 1024),
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # -------------------------------------------------------------------------
    def set_provider(self, provider: str, **kwargs):
        """Rebuild provider, removing duplicate provider keys."""
        self.provider = provider.lower()
        self.config = kwargs
        cfg = {k: v for k, v in kwargs.items() if k != "provider"}
        self.llm = self._build_provider(self.provider, **cfg)

    def reconfigure(self, **kwargs):
        """Hot-swap configuration values dynamically."""
        updates = {**self.config, **kwargs}
        hot_changes = {k: v for k, v in updates.items() if k in self.HOT_PARAMS}
        cold_changes = {k: v for k, v in updates.items() if k not in self.HOT_PARAMS}

        # If cold parameters changed, rebuild
        if cold_changes != {k: self.config.get(k) for k in cold_changes}:
            self.set_provider(self.provider, **updates)
        else:
            for k, v in hot_changes.items():
                if hasattr(self.llm, k):
                    setattr(self.llm, k, v)
            self.config.update(updates)

    # -------------------------------------------------------------------------
    def predict(self, messages) -> BaseMessage:
        """Standard blocking call to model."""
        if not self.llm:
            raise RuntimeError("No LLM configured.")
        return self.llm.invoke(messages)

    async def predict_async(self, messages) -> BaseMessage:
        """async blocking call to model."""
        if not self.llm:
            raise RuntimeError("No LLM configured.")
        return self.llm.invoke(messages)

    def stream_predict(self, messages):
        """Stream output from the model."""
        if not self.llm:
            raise RuntimeError("No LLM configured.")
        for chunk in self.llm.stream(messages):
            yield chunk
