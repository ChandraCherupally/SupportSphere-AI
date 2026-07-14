from __future__ import annotations

from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from src.graph.nodes import (
    input_check_node,
    retrieve_node,
    retrieval_check_node,
    generate_node,
    output_check_node,
)
from src.graph.state import SupportState


def build_graph():
    """
    Build and compile the SupportSphere workflow.

    Workflow

        START
          │
          ▼
      Input Check
          │
          ▼
       Retrieve
          │
          ▼
    Retrieval Check
          │
          ▼
       Generate
          │
          ▼
      Output Check
          │
          ▼
         END
    """

    workflow = StateGraph(SupportState)

    workflow.add_node(
        "input_check",
        input_check_node,
    )

    workflow.add_node(
        "retrieve",
        retrieve_node,
    )

    workflow.add_node(
        "retrieval_check",
        retrieval_check_node,
    )

    workflow.add_node(
        "generate",
        generate_node,
    )

    workflow.add_node(
        "output_check",
        output_check_node,
    )

    # Wire nodes
    workflow.add_edge(
        START,
        "input_check",
    )

    workflow.add_edge(
        "input_check",
        "retrieve",
    )

    workflow.add_edge(
        "retrieve",
        "retrieval_check",
    )

    workflow.add_edge(
        "retrieval_check",
        "generate",
    )

    workflow.add_edge(
        "generate",
        "output_check",
    )

    workflow.add_edge(
        "output_check",
        END,
    )

    return workflow.compile()


# =============================================================================
# Singleton Compiled Graph
# =============================================================================

support_graph = build_graph()