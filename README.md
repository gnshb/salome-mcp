# SalomeMCP

# Note: This is experimental! Use at your own risk!

SALOME Model Context Protocol for agentinc use

## Overview
This repository provides three components:
1. SALOME GUI plugin to start/stop the local bridge port.
2. SALOME-side bridge (`salome_bridge.py`) that executes GEOM/SMESH operations.
3. MCP server (`salome-mcp`) that agent clients connect to.

Runtime model:
1. Start SALOME GUI.
2. Start bridge from `Tools -> Plugins -> MCP Bridge`.
3. Start your MCP client (configured to run `salome-mcp`).
4. Agent tool calls flow: `Agent -> MCP server -> SALOME bridge -> SALOME GUI session`.

## Requirements
1. SALOME 9.x
2. Python 3.10+
3. `uv`

# Setup

## First time
### 1) Install Python dependencies
```bash
cd /absolute/path/to/salome-mcp
uv sync
```

### 2) Install SALOME plugin files
```bash
mkdir -p ~/.config/salome/Plugins
cp salome_plugin/salome_plugins.py ~/.config/salome/Plugins/
cp salome_plugin/mcp_port_plugin.py ~/.config/salome/Plugins/
cp salome_bridge.py ~/.config/salome/Plugins/
```

## Configure MCP Client
### Claude Desktop example
```json
{
  "mcpServers": {
    "salome": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/salome-mcp",
        "salome-mcp"
      ],
      "env": {
        "SALOME_HOST": "localhost",
        "SALOME_PORT": "1234"
      }
    }
  }
}
```

### Claude Code CLI example
```bash
claude mcp add salome uv run --directory /absolute/path/to/salome-mcp salome-mcp
```

## GUI Usage Guide
Menu path: `Tools -> Plugins -> MCP Bridge`

### `Start (default)`
1. Tries default host/port (`localhost:1234`).
2. If unavailable, prompts for a port.

### `Start (choose host and port)`
1. Prompts for host and port immediately.
2. Use when you want a fixed non-default port.

### `Status`
Shows current bridge state and connection hint.

### `Stop`
Stops bridge and closes the listening socket.

### Status badge meanings
1. `MCP Bridge: STARTING host:port` - bridge startup in progress.
2. `MCP Bridge: RUNNING host:port` - bridge ready.
3. `MCP Bridge: STOPPED` - bridge not running.

## Workflow
1. Start SALOME.
2. In GUI: `MCP Bridge -> Start (default)`.
3. Start/open your MCP agent client.
4. Run `check_salome_status`.
5. Work normally.
6. End session with `MCP Bridge -> Stop`.

## Tool Coverage
### Session
- `ping_salome`, `check_salome_status`, `get_study_info`, `get_scene_summary`, `list_study_objects`

### Geometry (GEOM)
- Primitive creation, transforms, copy/duplicate, rename/delete
- Boolean ops (`fuse`, `cut`, `common`)
- Groups (generic, surface, volume)
- Partition, explode, import/export, object inspection

### Mesh (SMESH)
- Import/export
- Mesh setup with hypotheses
- Compute and mesh info

### Advanced
- `execute_salome_code` (raw Python execution in SALOME process)

## Note
`execute_salome_code` executes arbitrary Python in the SALOME process. Keep bridge access local and trusted.
