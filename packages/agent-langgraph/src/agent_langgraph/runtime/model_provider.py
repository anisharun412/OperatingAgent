from common.config import AgentConfig
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr


def _ollama_base_url(value: str | None) -> str | None:
    """Normalize user-entered Ollama URLs to the server root.

    LangChain's Ollama client appends ``/api`` itself. Passing an API path here
    produces the confusing ``404 page not found`` response from Ollama.
    """
    if not value:
        return None
    return value.rstrip("/").removesuffix("/api") or None


class ModelProvider:
    """
    Provides the LLM used by the LangGraph nodes.

    A single model instance is shared across graph invocations.
    Nodes decide how to use the model; ModelProvider only owns
    model construction/access.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._model = self._create_model(config)

    @staticmethod
    def _create_model(config: AgentConfig) -> BaseChatModel:
        """
        Centralized model construction.

        Replace this implementation with the model backend
        used by the project, e.g. Ollama, OpenAI, Groq, etc.
        """

        provider = config.llm.provider.strip().lower()

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.llm.model,
                temperature=config.llm.temperature,
                top_p=config.llm.top_p,
                num_predict=config.llm.max_tokens,
                base_url=_ollama_base_url(config.llm.base_url),
                client_kwargs={"timeout": config.llm.timeout_seconds},
            )

        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                timeout=config.llm.timeout_seconds,
                model_kwargs={"top_p": config.llm.top_p},
                api_key=SecretStr(config.llm.api_key),
                base_url=(config.llm.base_url or None),
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model_name=config.llm.model,
                timeout=config.llm.timeout_seconds,
                temperature=config.llm.temperature,
                max_tokens_to_sample=config.llm.max_tokens,
                top_p=config.llm.top_p,
                api_key=SecretStr(config.llm.api_key),
                base_url=(config.llm.base_url or None),
                stop=None,
            )

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.llm.model,
                temperature=config.llm.temperature,
                top_p=config.llm.top_p,
                timeout=config.llm.timeout_seconds,
                max_completion_tokens=config.llm.max_tokens,
                api_key=SecretStr(config.llm.api_key),
                base_url=(config.llm.base_url or None),
            )

        raise NotImplementedError(
            f"unsupported LLM provider: {config.llm.provider!r}"
        )

    def get_model(self) -> BaseChatModel:
        """Return the shared chat model."""
        return self._model
