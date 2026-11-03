"""
support_agent
=============
ShopCo customer-support AI agent: the System Under Test (SUT).

Public API:
    from support_agent import SupportAgent, AgentResponse, Settings
"""

from support_agent.agent import SupportAgent
from support_agent.config import Settings
from support_agent.schemas import AgentResponse

__all__ = ["SupportAgent", "AgentResponse", "Settings"]
__version__ = "0.1.0"