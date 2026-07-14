from __future__ import annotations

from typing import Type

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import src.config as config


class LLMClient:
    """
    Provider-agnostic LLM client.

    Supports:

    • Google Gemini
    • OpenAI
    • Anthropic
    • Groq
    • Ollama
    """

    def __init__(self):

        self._llm = None

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> BaseChatModel:

        provider = config.LLM_PROVIDER.lower()

        if provider == "google":
            return ChatGoogleGenerativeAI(model=config.LLM_MODEL,
                                          temperature=config.LLM_TEMPERATURE,
                                          google_api_key=config.GOOGLE_API_KEY,)

        if provider == "openai":
            return ChatOpenAI(model=config.LLM_MODEL,
                              temperature=config.LLM_TEMPERATURE,
                              api_key=config.OPENAI_API_KEY,)

        if provider == "anthropic":
            return ChatAnthropic(model=config.LLM_MODEL, 
                                 temperature=config.LLM_TEMPERATURE,
                                 api_key=config.ANTHROPIC_API_KEY,)

        if provider == "groq":
            return ChatGroq(model=config.LLM_MODEL, 
                            temperature=config.LLM_TEMPERATURE, 
                            api_key=config.GROQ_API_KEY,)

        if provider == "ollama":
            return ChatOllama(model=config.LLM_MODEL, 
                              temperature=config.LLM_TEMPERATURE,)

        raise ValueError(
            f"Unsupported LLM provider: {provider}"
        )

    def generate(self, messages, response_schema: Type[BaseModel]) -> BaseModel:
        """
        Generate a structured response.
        """

        structured_llm = self.llm.with_structured_output(response_schema)

        return structured_llm.invoke(messages)