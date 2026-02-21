import json
import logging
import os
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from mcp.server.fastmcp import Context, FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SalomeMCPServer")

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1234


@dataclass
class SalomeConnection:
    host: str
    port: int
    sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        if self.sock:
            return True

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info("Connected to SALOME bridge at %s:%s", self.host, self.port)
            return True
        except Exception as exc:  # pragma: no cover - runtime network failure path
            logger.error("Failed to connect to SALOME bridge: %s", exc)
            self.sock = None
            return False

    def disconnect(self) -> None:
        if not self.sock:
            return
        try:
            self.sock.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.error("Error disconnecting from SALOME bridge: %s", exc)
        finally:
            self.sock = None

    def _receive_full_response(self, buffer_size: int = 8192) -> bytes:
        if not self.sock:
            raise ConnectionError("Socket is not connected")

        chunks = []
        self.sock.settimeout(120.0)

        while True:
            chunk = self.sock.recv(buffer_size)
            if not chunk:
                if chunks:
                    break
                raise ConnectionError("SALOME bridge closed connection")

            chunks.append(chunk)
            payload = b"".join(chunks)
            try:
                json.loads(payload.decode("utf-8"))
                return payload
            except json.JSONDecodeError:
                continue

        payload = b"".join(chunks)
        json.loads(payload.decode("utf-8"))
        return payload

    def send_command(self, command_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to SALOME bridge")

        command = {"type": command_type, "params": params or {}}

        try:
            assert self.sock is not None
            self.sock.sendall(json.dumps(command).encode("utf-8"))
            response_data = self._receive_full_response()
            response = json.loads(response_data.decode("utf-8"))

            if response.get("status") == "error":
                raise RuntimeError(response.get("message", "Unknown SALOME error"))

            return response.get("result", {})
        except Exception:
            self.disconnect()
            raise


_salome_connection: Optional[SalomeConnection] = None


def get_salome_connection() -> SalomeConnection:
    global _salome_connection

    if _salome_connection is not None:
        try:
            _salome_connection.send_command("ping")
            return _salome_connection
        except Exception as exc:
            logger.warning("Existing SALOME connection is stale: %s", exc)
            _salome_connection.disconnect()
            _salome_connection = None

    host = os.getenv("SALOME_HOST", DEFAULT_HOST)
    port = int(os.getenv("SALOME_PORT", DEFAULT_PORT))

    _salome_connection = SalomeConnection(host=host, port=port)
    if not _salome_connection.connect():
        _salome_connection = None
        raise RuntimeError(
            "Could not connect to SALOME bridge. Start MCP Bridge from SALOME GUI plugin first."
        )

    return _salome_connection


def _run(command: str, params: Optional[Dict[str, Any]] = None) -> str:
    result = get_salome_connection().send_command(command, params or {})
    return json.dumps(result, indent=2)


@asynccontextmanager
async def server_lifespan(_server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    logger.info("SALOME MCP server starting")
    try:
        try:
            get_salome_connection()
            logger.info("SALOME bridge reachable at startup")
        except Exception as exc:
            logger.warning("SALOME bridge not reachable at startup: %s", exc)
        yield {}
    finally:
        global _salome_connection
        if _salome_connection:
            _salome_connection.disconnect()
            _salome_connection = None
        logger.info("SALOME MCP server stopped")


mcp = FastMCP("SalomeMCP", lifespan=server_lifespan)


@mcp.tool()
def ping_salome(ctx: Context) -> str:
    """Check connectivity to the SALOME bridge process."""
    _ = ctx
    result = get_salome_connection().send_command("ping")
    return result.get("message", "SALOME bridge reachable")


@mcp.tool()
def check_salome_status(ctx: Context) -> str:
    """
    Standard health check for bridge accessibility and active study.

    Returns host/port, reachability, ping message, and study metadata when available.
    """
    _ = ctx
    host = os.getenv("SALOME_HOST", DEFAULT_HOST)
    port = int(os.getenv("SALOME_PORT", DEFAULT_PORT))

    status: Dict[str, Any] = {
        "host": host,
        "port": port,
        "reachable": False,
        "ping": None,
        "study": None,
    }

    try:
        conn = get_salome_connection()
        ping = conn.send_command("ping")
        status["reachable"] = True
        status["ping"] = ping.get("message", "SALOME bridge reachable")

        try:
            status["study"] = conn.send_command("get_study_info")
        except Exception as exc:
            status["study_error"] = str(exc)
    except Exception as exc:
        status["error"] = str(exc)

    return json.dumps(status, indent=2)


@mcp.tool()
def get_study_info(ctx: Context) -> str:
    """Return high-level information about the current SALOME study."""
    _ = ctx
    return _run("get_study_info")


@mcp.tool()
def get_scene_summary(ctx: Context, limit_per_component: int = 200) -> str:
    """Return GEOM and SMESH scene summary for the active study."""
    _ = ctx
    return _run("get_scene_summary", {"limit_per_component": limit_per_component})


@mcp.tool()
def list_study_objects(ctx: Context, component: Optional[str] = None, limit: int = 200) -> str:
    """List study objects, optionally filtered by component (GEOM, SMESH, ...)."""
    _ = ctx
    return _run("list_study_objects", {"component": component, "limit": limit})


@mcp.tool()
def get_object_info(ctx: Context, object_ref: str, precise_bbox: bool = False) -> str:
    """Return GEOM object info: name, bbox, basic properties, and subshape counts."""
    _ = ctx
    return _run("get_object_info", {"object_ref": object_ref, "precise_bbox": precise_bbox})


@mcp.tool()
def create_box(ctx: Context, dx: float, dy: float, dz: float, name: str = "Box") -> str:
    """Create a GEOM box by dimensions."""
    _ = ctx
    return _run("create_box", {"dx": dx, "dy": dy, "dz": dz, "name": name})


@mcp.tool()
def create_cylinder(ctx: Context, radius: float, height: float, name: str = "Cylinder") -> str:
    """Create a GEOM cylinder at origin."""
    _ = ctx
    return _run("create_cylinder", {"radius": radius, "height": height, "name": name})


@mcp.tool()
def create_sphere(ctx: Context, radius: float, name: str = "Sphere") -> str:
    """Create a GEOM sphere at origin."""
    _ = ctx
    return _run("create_sphere", {"radius": radius, "name": name})


@mcp.tool()
def boolean_operation(
    ctx: Context,
    operation: str,
    base_object: str,
    tool_objects: List[str],
    result_name: str = "BooleanResult",
) -> str:
    """
    Perform boolean operation on GEOM objects.

    operation: fuse | cut | common
    base_object/tool_objects: entry IDs or object names
    """
    _ = ctx
    return _run(
        "boolean_operation",
        {
            "operation": operation,
            "base_object": base_object,
            "tool_objects": tool_objects,
            "result_name": result_name,
        },
    )


@mcp.tool()
def fuse_objects(
    ctx: Context,
    base_object: str,
    tool_objects: List[str],
    result_name: str = "Fuse",
) -> str:
    """Fuse GEOM objects (explicit convenience wrapper for boolean fuse)."""
    _ = ctx
    return _run(
        "fuse_objects",
        {
            "base_object": base_object,
            "tool_objects": tool_objects,
            "result_name": result_name,
        },
    )


@mcp.tool()
def cut_objects(
    ctx: Context,
    base_object: str,
    tool_objects: List[str],
    result_name: str = "Cut",
) -> str:
    """Cut GEOM objects (base minus tools)."""
    _ = ctx
    return _run(
        "cut_objects",
        {
            "base_object": base_object,
            "tool_objects": tool_objects,
            "result_name": result_name,
        },
    )


@mcp.tool()
def common_objects(
    ctx: Context,
    base_object: str,
    tool_objects: List[str],
    result_name: str = "Common",
) -> str:
    """Intersect GEOM objects (common region)."""
    _ = ctx
    return _run(
        "common_objects",
        {
            "base_object": base_object,
            "tool_objects": tool_objects,
            "result_name": result_name,
        },
    )


@mcp.tool()
def copy_object(ctx: Context, source_ref: str, name: str = "Copy") -> str:
    """Create a GEOM copy of an object."""
    _ = ctx
    return _run("copy_object", {"source_ref": source_ref, "name": name})


@mcp.tool()
def duplicate_object(
    ctx: Context,
    source_ref: str,
    count: int = 1,
    name_prefix: Optional[str] = None,
) -> str:
    """Create N GEOM duplicates from one source object."""
    _ = ctx
    return _run(
        "duplicate_object",
        {
            "source_ref": source_ref,
            "count": count,
            "name_prefix": name_prefix,
        },
    )


@mcp.tool()
def translate_object(
    ctx: Context,
    source_ref: str,
    dx: float,
    dy: float,
    dz: float,
) -> str:
    """Translate a GEOM object in place by vector components (no new copy)."""
    _ = ctx
    return _run(
        "translate_object",
        {
            "source_ref": source_ref,
            "dx": dx,
            "dy": dy,
            "dz": dz,
        },
    )


@mcp.tool()
def rotate_object(
    ctx: Context,
    source_ref: str,
    angle_degrees: float,
    axis: str = "Z",
) -> str:
    """Rotate a GEOM object in place around X/Y/Z axis (degrees, no new copy)."""
    _ = ctx
    return _run(
        "rotate_object",
        {
            "source_ref": source_ref,
            "angle_degrees": angle_degrees,
            "axis": axis,
        },
    )


@mcp.tool()
def rename_object(ctx: Context, object_ref: str, new_name: str) -> str:
    """Rename GEOM study object (entry ID or existing name)."""
    _ = ctx
    return _run("rename_object", {"object_ref": object_ref, "new_name": new_name})


@mcp.tool()
def delete_object(ctx: Context, object_ref: str, with_children: bool = True) -> str:
    """Delete GEOM study object by entry/name."""
    _ = ctx
    return _run("delete_object", {"object_ref": object_ref, "with_children": with_children})


@mcp.tool()
def list_subshapes(
    ctx: Context,
    shape_ref: str,
    subshape_type: str = "FACE",
    sorted_centres: bool = True,
) -> str:
    """List subshape IDs (FACE/EDGE/VERTEX/SOLID/...) for a GEOM object."""
    _ = ctx
    return _run(
        "list_subshapes",
        {
            "shape_ref": shape_ref,
            "subshape_type": subshape_type,
            "sorted_centres": sorted_centres,
        },
    )


@mcp.tool()
def create_group(
    ctx: Context,
    shape_ref: str,
    subshape_type: str,
    subshape_ids: List[int],
    name: str = "Group",
) -> str:
    """Create GEOM group from subshape IDs."""
    _ = ctx
    return _run(
        "create_group",
        {
            "shape_ref": shape_ref,
            "subshape_type": subshape_type,
            "subshape_ids": subshape_ids,
            "name": name,
        },
    )


@mcp.tool()
def create_surface_group(
    ctx: Context,
    shape_ref: str,
    subshape_ids: List[int],
    name: str = "SurfaceGroup",
) -> str:
    """Create FACE group convenience wrapper."""
    _ = ctx
    return _run(
        "create_surface_group",
        {
            "shape_ref": shape_ref,
            "subshape_ids": subshape_ids,
            "name": name,
        },
    )


@mcp.tool()
def create_volume_group(
    ctx: Context,
    shape_ref: str,
    subshape_ids: List[int],
    name: str = "VolumeGroup",
) -> str:
    """Create SOLID group convenience wrapper."""
    _ = ctx
    return _run(
        "create_volume_group",
        {
            "shape_ref": shape_ref,
            "subshape_ids": subshape_ids,
            "name": name,
        },
    )


@mcp.tool()
def make_partition(
    ctx: Context,
    object_refs: List[str],
    tool_refs: Optional[List[str]] = None,
    result_name: str = "Partition",
    shape_type: str = "SOLID",
    keep_non_limit_shapes: bool = False,
) -> str:
    """Create GEOM partition from objects and optional tools."""
    _ = ctx
    return _run(
        "make_partition",
        {
            "object_refs": object_refs,
            "tool_refs": tool_refs or [],
            "result_name": result_name,
            "shape_type": shape_type,
            "keep_non_limit_shapes": keep_non_limit_shapes,
        },
    )


@mcp.tool()
def explode_shape(
    ctx: Context,
    shape_ref: str,
    subshape_type: str = "FACE",
    result_prefix: str = "Exploded",
    add_to_study: bool = True,
    sorted_centres: bool = True,
) -> str:
    """Explode GEOM object to subshapes and optionally add them to study."""
    _ = ctx
    return _run(
        "explode_shape",
        {
            "shape_ref": shape_ref,
            "subshape_type": subshape_type,
            "result_prefix": result_prefix,
            "add_to_study": add_to_study,
            "sorted_centres": sorted_centres,
        },
    )


@mcp.tool()
def import_geometry(
    ctx: Context,
    filepath: str,
    format: str = "auto",
    name: Optional[str] = None,
    ignore_units: bool = False,
) -> str:
    """Import GEOM from BREP/STEP/IGES."""
    _ = ctx
    return _run(
        "import_geometry",
        {
            "filepath": filepath,
            "format": format,
            "name": name,
            "ignore_units": ignore_units,
        },
    )


@mcp.tool()
def export_geometry(
    ctx: Context,
    shape_ref: str,
    filepath: str,
    format: str = "auto",
    ascii_stl: bool = True,
    stl_deflection: float = 0.001,
) -> str:
    """Export GEOM object to BREP/STEP/IGES/STL."""
    _ = ctx
    return _run(
        "export_geometry",
        {
            "shape_ref": shape_ref,
            "filepath": filepath,
            "format": format,
            "ascii_stl": ascii_stl,
            "stl_deflection": stl_deflection,
        },
    )


@mcp.tool()
def import_mesh(ctx: Context, filepath: str, format: str = "auto", name: Optional[str] = None) -> str:
    """Import mesh from MED/UNV/STL/CGNS/GMF."""
    _ = ctx
    return _run("import_mesh", {"filepath": filepath, "format": format, "name": name})


@mcp.tool()
def export_mesh(
    ctx: Context,
    mesh_ref: str,
    filepath: str,
    format: str = "auto",
    ascii_stl: bool = True,
    auto_dimension: bool = True,
) -> str:
    """Export mesh to MED/UNV/STL."""
    _ = ctx
    return _run(
        "export_mesh",
        {
            "mesh_ref": mesh_ref,
            "filepath": filepath,
            "format": format,
            "ascii_stl": ascii_stl,
            "auto_dimension": auto_dimension,
        },
    )


@mcp.tool()
def create_mesh(
    ctx: Context,
    shape_ref: str,
    mesh_name: str = "Mesh",
    segment_count: int = 10,
    max_element_area: Optional[float] = None,
    max_element_volume: Optional[float] = None,
    surface_algorithm: str = "triangle",
    volume_algorithm: str = "tetrahedron",
) -> str:
    """
    Create a mesh definition on GEOM object using standard hypotheses.

    surface_algorithm: triangle | quadrangle | none
    volume_algorithm: tetrahedron | hexahedron | none
    """
    _ = ctx
    return _run(
        "create_mesh",
        {
            "shape_ref": shape_ref,
            "mesh_name": mesh_name,
            "segment_count": segment_count,
            "max_element_area": max_element_area,
            "max_element_volume": max_element_volume,
            "surface_algorithm": surface_algorithm,
            "volume_algorithm": volume_algorithm,
        },
    )


@mcp.tool()
def compute_mesh(ctx: Context, mesh_ref: str) -> str:
    """Compute mesh and return basic mesh statistics."""
    _ = ctx
    return _run("compute_mesh", {"mesh_ref": mesh_ref})


@mcp.tool()
def get_mesh_info(ctx: Context, mesh_ref: str) -> str:
    """Return mesh counts and detailed SMESH info."""
    _ = ctx
    return _run("get_mesh_info", {"mesh_ref": mesh_ref})


@mcp.tool()
def execute_salome_code(ctx: Context, code: str) -> str:
    """
    Execute arbitrary Python code in the SALOME process.

    The executed code can set a `result` variable that will be returned.
    """
    _ = ctx
    return _run("execute_code", {"code": code})


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
