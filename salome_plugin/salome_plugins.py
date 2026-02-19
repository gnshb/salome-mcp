"""SALOME plugin entrypoint for MCP bridge controls."""

import salome_pluginsmanager

from mcp_port_plugin import (
    bridge_status,
    install_live_status_widget,
    install_quit_hook,
    start_bridge_default,
    start_bridge_dialog,
    stop_bridge,
)

# Try to show the live status badge immediately when plugin is loaded.
install_live_status_widget()
install_quit_hook()

salome_pluginsmanager.AddFunction(
    "MCP Bridge/Start (default)",
    "Open the MCP bridge on the configured default host/port",
    start_bridge_default,
)

salome_pluginsmanager.AddFunction(
    "MCP Bridge/Start (choose host and port)",
    "Open the MCP bridge and choose host/port interactively",
    start_bridge_dialog,
)

salome_pluginsmanager.AddFunction(
    "MCP Bridge/Status",
    "Show whether MCP bridge is currently running",
    bridge_status,
)

salome_pluginsmanager.AddFunction(
    "MCP Bridge/Stop",
    "Close the MCP bridge port",
    stop_bridge,
)
