"""MCP server exposing the Qloo taste graph to any MCP-capable agent.

Run with ``taste-mcp`` (stdio). Needs ``QLOO_API_KEY`` in the environment. The typed functions below exist so the MCP
SDK can publish argument schemas; every one of them dispatches to the shared registry, so MCP clients and the built-in
agent run the same code with the same argument checks.
"""
from __future__ import annotations

from typing import List, Optional

from .qloo import QlooClient
from .registry import TOOLS_BY_NAME, call_tool


def _args(**kwargs) -> dict:
    return {key: value for key, value in kwargs.items() if value is not None}


def build_server(client: Optional[QlooClient] = None):
    try:  # mcp 2.x
        from mcp.server.mcpserver import MCPServer
    except ModuleNotFoundError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as MCPServer

    qloo = client or QlooClient()
    mcp = MCPServer("taste-mcp")

    def describe(name: str) -> str:
        return TOOLS_BY_NAME[name].description

    @mcp.tool(description=describe("find_entities"))
    def find_entities(names: List[str], types: Optional[List[str]] = None) -> dict:
        return call_tool(qloo, "find_entities", _args(names=names, types=types))

    @mcp.tool(description=describe("find_tags"))
    def find_tags(query: str, take: int = 10) -> dict:
        return call_tool(qloo, "find_tags", _args(query=query, take=take))

    @mcp.tool(description=describe("list_audiences"))
    def list_audiences(category: str, take: int = 50) -> dict:
        return call_tool(qloo, "list_audiences", _args(category=category, take=take))

    @mcp.tool(description=describe("recommend"))
    def recommend(
        target_type: str,
        entity_ids: Optional[List[str]] = None,
        tag_ids: Optional[List[str]] = None,
        audience_ids: Optional[List[str]] = None,
        city: Optional[str] = None,
        take: int = 8,
    ) -> dict:
        return call_tool(qloo, "recommend", _args(target_type=target_type, entity_ids=entity_ids, tag_ids=tag_ids,
                                                  audience_ids=audience_ids, city=city, take=take))

    @mcp.tool(description=describe("bridge_tastes"))
    def bridge_tastes(seeds: List[str], target_type: str, city: Optional[str] = None, take: int = 8) -> dict:
        return call_tool(qloo, "bridge_tastes", _args(seeds=seeds, target_type=target_type, city=city, take=take))

    @mcp.tool(description=describe("score_candidates"))
    def score_candidates(target_type: str, entity_ids: List[str], candidate_ids: List[str]) -> dict:
        return call_tool(qloo, "score_candidates", _args(target_type=target_type, entity_ids=entity_ids,
                                                         candidate_ids=candidate_ids))

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
