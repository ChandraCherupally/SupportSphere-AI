"""
Evaluation framework module for SupportSphere AI.
"""

from __future__ import annotations

import sys
import types

# Enterprise stub to bypass VertexAI dependency warning in langchain-community used by Ragas.
# This ensures Ragas loads correctly in any workspace configuration without installing extra packages.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    mock_module = types.ModuleType("langchain_community.chat_models.vertexai")
    mock_module.ChatVertexAI = object  # type: ignore
    sys.modules["langchain_community.chat_models.vertexai"] = mock_module
