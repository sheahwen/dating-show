"""Shared model client creation helpers."""

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient


def create_model_client(config: dict | None, default_client: str = "ollama", default_model: str = "llama3.2"):
    """
    Create a chat model client from a simple config dict.

    Expected config keys:
      - client: "ollama" | "openai"
      - model: model name
    """
    client_type = default_client
    model_name = default_model

    if config:
        client_type = config.get("client", default_client)
        model_name = config.get("model", default_model)

    if client_type == "ollama":
        return OllamaChatCompletionClient(model=model_name)
    if client_type == "openai":
        return OpenAIChatCompletionClient(model=model_name)

    raise ValueError(f"Unsupported client type: {client_type}")

