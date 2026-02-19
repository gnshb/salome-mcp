# SalomeMCP

---
### Note: This is experimental! Use at your own risk!
---

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

## Setup

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

## Sample 

Prompt:

```text
lets do these step by step: 1. make a cylinder r = 4 h =8 and place it along x axis 2. make two more cylinders of r = 2 and h = 4 3. place these cylinders on either of the circular ends 4. carve a cylinder out of these smaller ones to make two shells of rout = 2 and rin = 1 5. fuse all 6. create the single vol group 7. make surface groups inlet outlet and walls 8. make a mesh with netgen 12 2d 3d very fine mesh. 9. compute mesh and report its stats
```

![image](./sample.png)
