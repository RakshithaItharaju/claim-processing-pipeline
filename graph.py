"""LangGraph workflow.

START -> segregator -> (id_agent | discharge_agent | bill_agent, in parallel) -> aggregator -> END
"""
from __future__ import annotations

import operator
from functools import partial
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import (
    aggregator_node,
    bill_agent_node,
    discharge_agent_node,
    id_agent_node,
    segregator_node,
)
from .llm import LLM


class ClaimState(TypedDict, total=False):
    claim_id: str
    pages: list[dict]                      # every page's text (only the segregator reads this)
    classifications: dict[int, str]        # page -> doc type
    routed: dict[str, list[dict]]          # agent -> ONLY the pages assigned to it
    id_result: dict[str, Any]
    discharge_result: dict[str, Any]
    bill_result: dict[str, Any]
    warnings: Annotated[list[str], operator.add]
    result: dict[str, Any]


def build_graph(llm: LLM):
    g = StateGraph(ClaimState)

    g.add_node("segregator", partial(segregator_node, llm=llm))
    g.add_node("id_agent", partial(id_agent_node, llm=llm))
    g.add_node("discharge_agent", partial(discharge_agent_node, llm=llm))
    g.add_node("bill_agent", partial(bill_agent_node, llm=llm))
    g.add_node("aggregator", partial(aggregator_node, llm_name=getattr(llm, "name", "")))

    g.add_edge(START, "segregator")
    # fan-out: segregator -> the three extraction agents (run in parallel)
    for agent in ("id_agent", "discharge_agent", "bill_agent"):
        g.add_edge("segregator", agent)
    # fan-in: aggregator waits for ALL three
    g.add_edge(["id_agent", "discharge_agent", "bill_agent"], "aggregator")
    g.add_edge("aggregator", END)
    return g.compile()
