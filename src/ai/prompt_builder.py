from __future__ import annotations

from typing import Any
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from src.ai.system_prompt import SYSTEM_PROMPT
from src.ai.models import SupportTicket

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

        self.template = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), 
                                                          ("human", "{user_prompt}")])


    def build(
        self,
        *,
        ticket: SupportTicket,
        context: list[dict[str, Any]],
        retrieval_required: bool = True,
        routing_reason: str = "",
    ) -> list[BaseMessage]:
        """
        Build chat messages for the LLM.
        """
        if not retrieval_required:
            user_prompt = self._build_routing_prompt(
                issue=ticket.issue,
                subject=ticket.subject,
                company=ticket.company,
                routing_reason=routing_reason,
            )
        else:
            user_prompt = self._build_user_prompt(
                issue=ticket.issue,
                subject=ticket.subject,
                company=ticket.company,
                context=context,
            )
        return self.template.format_messages(user_prompt=user_prompt)

    def _build_routing_prompt(
        self,
        *,
        issue: str,
        subject: str,
        company: str,
        routing_reason: str,
    ) -> str:
        return f"""
                # SUPPORT TICKET (ROUTING MODE)

                Company
                -------
                {company or "Not Provided"}

                Subject
                -------
                {subject or "Not Provided"}

                Issue
                -----
                {issue}

                Routing Reason
                --------------
                {routing_reason or "Retrieval skipped by routing gate."}
                """


    def _build_user_prompt(self, *, issue: str, subject: str, company: str, context: list[dict[str, Any]],) -> str:

        context_text = self._format_context(context)
        product_areas = sorted({chunk["product_area"] for chunk in context})
        
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
            doc_str = (
                f"## Document {idx}\n\n"
                f"Company:\n{chunk.get('company', '')}\n\n"
                f"Product Area:\n{chunk.get('product_area', '')}\n\n"
                f"Title:\n{chunk.get('title', '')}\n\n"
                f"Section:\n{chunk.get('section', '')}\n\n"
                f"URL:\n{chunk.get('url', '')}\n\n"
                f"Content:\n{chunk.get('text', '')}"
            )
            documents.append(doc_str)
        return "\n\n".join(documents)