from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from src.prompts.system_prompt import SYSTEM_PROMPT
from src.llm.models import SupportTicket

class PromptBuilder:
    """
    Provider-agnostic prompt builder.

    Compatible with

    - Gemini
    - GPT
    - Claude
    - Groq
    - Ollama
    """

    def __init__(self) -> None:

        self.template = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{user_prompt}"),
            ]
        )

    def build(
        self,
        *,
        ticket: SupportTicket,
        context: list[dict[str, Any]],
    ) -> list[BaseMessage]:
        """
        Build chat messages for the LLM.
        """

        user_prompt = self._build_user_prompt(
            issue=ticket.issue,
            subject=ticket.subject,
            company=ticket.company,
            context=context,
        )

        return self.template.format_messages(
            user_prompt=user_prompt,
        )

    def _build_user_prompt(
        self,
        *,
        issue: str,
        subject: str,
        company: str,
        context: list[dict[str, Any]],
    ) -> str:

        context_text = self._format_context(context)

        product_areas = sorted(
            {
                chunk["product_area"]
                for chunk in context
            }
        )

        return f"""
# SUPPORT TICKET

Company
-------
{company or "Not Provided"}

Subject
-------
{subject or "Not Provided"}

Issue
-----
{issue}


# RETRIEVAL SUMMARY

Retrieved Documents
-------------------
{len(context)}

Retrieved Product Areas
-----------------------
{", ".join(product_areas) if product_areas else "Unknown"}


# RETRIEVED DOCUMENTATION

{context_text}
"""

    @staticmethod
    def _format_context(
        context: list[dict[str, Any]],
    ) -> str:

        documents = []

        for idx, chunk in enumerate(context, start=1):

            documents.append(
f"""## Document {idx}

Company:
{chunk["company"]}

Product Area:
{chunk["product_area"]}

Title:
{chunk["title"]}

Section:
{chunk["section"]}

URL:
{chunk["url"]}

Content:
{chunk["text"]}
"""
            )

        return "\n\n".join(documents)