"""
agents/shared/agent_bus.py
----------------------------
Re-export AgentBus from scripts/agent_bus.py for convenience.

Usage:
    from agents.shared.agent_bus import get_bus, AgentBus
"""
from scripts.agent_bus import AgentBus, get_bus  # noqa: F401

__all__ = ["AgentBus", "get_bus"]
