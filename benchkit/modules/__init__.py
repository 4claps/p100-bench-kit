"""Test modules. Each module declares its parameters (with defaults the UI
renders), estimates its duration, checks it can run, and runs blocking code
in the runner's worker thread through a RunContext.
"""

from .agent import AgentModule
from .offload import OffloadModule
from .server import ServerModule
from .speed import SpeedModule

# run order: quick hardware numbers first, the long agent battery last
MODULE_ORDER = ["speed", "server", "offload", "agent"]
MODULES = {m.name: m for m in (SpeedModule(), ServerModule(), AgentModule(), OffloadModule())}

__all__ = ["MODULES", "MODULE_ORDER"]
