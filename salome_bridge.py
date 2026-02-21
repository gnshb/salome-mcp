#!/usr/bin/env python3
"""Socket bridge to expose SALOME GUI operations to an external MCP server."""

import io
import json
import logging
import math
import os
import socket
import threading
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("SalomeBridge")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1234


class SalomeRuntime:
    """SALOME API wrapper with persistent runtime state."""

    def __init__(self) -> None:
        self.exec_globals: Dict[str, Any] = {}
        self._initialized = False
        self._meshes_by_name: Dict[str, Any] = {}

    def initialize(self) -> None:
        if self._initialized:
            return

        import salome  # type: ignore

        salome.salome_init()
        self._ensure_gui_session(salome)
        self.exec_globals["salome"] = salome

        try:
            from salome.geom import geomBuilder  # type: ignore

            self.exec_globals["geomBuilder"] = geomBuilder
            self.exec_globals["geompy"] = geomBuilder.New()
        except Exception as exc:  # pragma: no cover - depends on SALOME modules
            logger.warning("Could not initialize GEOM module: %s", exc)

        try:
            import SMESH  # type: ignore
            from salome.smesh import smeshBuilder  # type: ignore

            self.exec_globals["SMESH"] = SMESH
            self.exec_globals["smeshBuilder"] = smeshBuilder
            self.exec_globals["smesh"] = smeshBuilder.New()
        except Exception as exc:  # pragma: no cover - depends on SALOME modules
            logger.warning("Could not initialize SMESH module: %s", exc)

        self._initialized = True
        logger.info("SALOME runtime initialized")

    @staticmethod
    def _ensure_gui_session(salome_module: Any) -> None:
        try:
            has_desktop = bool(salome_module.sg.hasDesktop())
        except Exception:
            has_desktop = False

        if not has_desktop:
            raise RuntimeError(
                "Headless SALOME bridge is disabled. Start bridge from SALOME GUI plugin."
            )

    def _refresh_gui(self) -> None:
        salome = self.exec_globals.get("salome")
        if not salome:
            return
        try:
            if salome.sg.hasDesktop():
                salome.sg.updateObjBrowser()
        except Exception:
            pass

    def _apply_visibility(
        self,
        show_entries: Optional[Sequence[str]] = None,
        hide_entries: Optional[Sequence[str]] = None,
    ) -> None:
        salome = self.exec_globals.get("salome")
        if not salome:
            return
        try:
            if not salome.sg.hasDesktop():
                return
        except Exception:
            return

        show = {entry for entry in (show_entries or []) if entry}
        hide = {entry for entry in (hide_entries or []) if entry}
        hide -= show

        for entry in hide:
            try:
                salome.sg.Erase(entry)
            except Exception:
                pass

        for entry in show:
            try:
                salome.sg.Display(entry)
            except Exception:
                pass

        self._refresh_gui()

    def _iter_component_sobjects(self, component: str, include_nested: bool = True) -> List[Any]:
        self.initialize()
        salome = self.exec_globals["salome"]
        study = salome.myStudy
        out: List[Any] = []

        def collect_children(parent: Any) -> None:
            child_iter = study.NewChildIterator(parent)
            while child_iter.More():
                child = child_iter.Value()
                child_iter.Next()
                out.append(child)
                if include_nested:
                    collect_children(child)

        comp_iter = study.NewComponentIterator()
        while comp_iter.More():
            comp_sobj = comp_iter.Value()
            comp_iter.Next()
            comp_name = comp_sobj.ComponentDataType() if hasattr(comp_sobj, "ComponentDataType") else ""
            if comp_name != component:
                continue
            collect_children(comp_sobj)

        return out

    @staticmethod
    def _sobj_entry(sobj: Any) -> str:
        if hasattr(sobj, "GetID"):
            return sobj.GetID()
        return ""

    @staticmethod
    def _sobj_name(sobj: Any) -> str:
        if hasattr(sobj, "GetName"):
            return sobj.GetName()
        return ""

    def _lookup_entry_by_name(self, component: str, name: str) -> Optional[str]:
        for sobj in self._iter_component_sobjects(component):
            if self._sobj_name(sobj) == name:
                entry = self._sobj_entry(sobj)
                if entry:
                    return entry
        return None

    def _object_from_entry(self, entry: str) -> Any:
        salome = self.exec_globals["salome"]
        try:
            return salome.IDToObject(entry)
        except Exception:
            return None

    def _resolve_shape(self, reference: str) -> Tuple[Any, str, str]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        ref = reference.strip()
        obj = self._object_from_entry(ref)
        entry = ref if obj is not None else ""

        if obj is None:
            entry = self._lookup_entry_by_name("GEOM", ref)
            if entry:
                obj = self._object_from_entry(entry)

        if obj is None:
            try:
                obj = geompy.GetObject(ref)
            except Exception:
                obj = None

        if obj is None:
            raise RuntimeError(f"GEOM object not found: {reference}")

        name = ref
        if entry:
            sobj = self.exec_globals["salome"].myStudy.FindObjectID(entry)
            if sobj is not None:
                name = self._sobj_name(sobj) or name

        return obj, entry, name

    @staticmethod
    def _extract_first_mesh(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return SalomeRuntime._extract_first_mesh(value[0])
        return value

    def _register_mesh(self, mesh: Any, name: Optional[str] = None) -> str:
        mesh_name = name or ""
        if not mesh_name:
            try:
                mesh_name = mesh.GetName()
            except Exception:
                mesh_name = ""

        if not mesh_name:
            mesh_name = f"Mesh_{len(self._meshes_by_name) + 1}"

        self._meshes_by_name[mesh_name] = mesh

        smesh = self.exec_globals.get("smesh")
        try:
            if smesh is not None:
                smesh.SetName(mesh.GetMesh(), mesh_name)
        except Exception:
            pass

        return mesh_name

    def _resolve_mesh(self, reference: str) -> Tuple[Any, str]:
        self.initialize()

        ref = reference.strip()
        if ref in self._meshes_by_name:
            mesh_obj = self._coerce_mesh_wrapper(self._meshes_by_name[ref])
            self._meshes_by_name[ref] = mesh_obj
            return mesh_obj, ref

        obj = self._object_from_entry(ref)
        if obj is not None and hasattr(obj, "NbNodes"):
            name = ref
            try:
                name = obj.GetName() or ref
            except Exception:
                pass
            mesh_obj = self._coerce_mesh_wrapper(obj)
            self._meshes_by_name[name] = mesh_obj
            return mesh_obj, name

        entry = self._lookup_entry_by_name("SMESH", ref)
        if entry:
            obj = self._object_from_entry(entry)
            if obj is not None and hasattr(obj, "NbNodes"):
                mesh_obj = self._coerce_mesh_wrapper(obj)
                self._meshes_by_name[ref] = mesh_obj
                return mesh_obj, ref

        raise RuntimeError(f"Mesh not found: {reference}")

    def _coerce_mesh_wrapper(self, mesh_obj: Any) -> Any:
        """Return a mesh object that supports high-level operations like Compute()."""
        if hasattr(mesh_obj, "Compute"):
            return mesh_obj

        smesh_engine = self.exec_globals.get("smesh")
        if smesh_engine is None:
            return mesh_obj

        try:
            # SALOME 9.x: smesh is the builder instance and Mesh(obj) wraps a CORBA mesh proxy.
            wrapped = smesh_engine.Mesh(mesh_obj)
            if hasattr(wrapped, "Compute"):
                return wrapped
        except Exception:
            pass

        return mesh_obj

    @staticmethod
    def _normalized_format(filepath: str, requested: str = "auto") -> str:
        if requested and requested.lower() != "auto":
            return requested.lower()

        ext = Path(filepath).suffix.lower()
        mapping = {
            ".brep": "brep",
            ".step": "step",
            ".stp": "step",
            ".iges": "iges",
            ".igs": "iges",
            ".stl": "stl",
            ".med": "med",
            ".unv": "unv",
            ".cgns": "cgns",
            ".mesh": "mesh",
            ".meshb": "mesh",
        }
        if ext in mapping:
            return mapping[ext]
        raise RuntimeError(f"Unsupported file extension: {ext}")

    def ping(self) -> Dict[str, Any]:
        self.initialize()
        return {"message": "SALOME bridge is connected"}

    def get_study_info(self) -> Dict[str, Any]:
        self.initialize()
        salome = self.exec_globals["salome"]
        study = salome.myStudy

        components: List[str] = []
        iterator = study.NewComponentIterator()
        while iterator.More():
            component = iterator.Value()
            iterator.Next()
            components.append(component.ComponentDataType())

        return {
            "study_name": study._get_Name() if hasattr(study, "_get_Name") else "",
            "component_count": len(components),
            "components": components,
        }

    def list_study_objects(self, component: Optional[str], limit: int) -> Dict[str, Any]:
        self.initialize()
        salome = self.exec_globals["salome"]
        study = salome.myStudy
        objects: List[Dict[str, str]] = []

        comp_iter = study.NewComponentIterator()
        while comp_iter.More():
            comp_sobj = comp_iter.Value()
            comp_iter.Next()

            comp_name = comp_sobj.ComponentDataType() if hasattr(comp_sobj, "ComponentDataType") else ""
            if component and comp_name != component:
                continue

            for sobj in self._iter_component_sobjects(comp_name):
                objects.append(
                    {
                        "component": comp_name,
                        "entry": self._sobj_entry(sobj),
                        "name": self._sobj_name(sobj),
                    }
                )
                if len(objects) >= limit:
                    return {"count": len(objects), "truncated": True, "objects": objects}

        return {"count": len(objects), "truncated": False, "objects": objects}

    def get_scene_summary(self, limit_per_component: int) -> Dict[str, Any]:
        self.initialize()
        geom_objects = self.list_study_objects("GEOM", limit_per_component)
        mesh_objects = self.list_study_objects("SMESH", limit_per_component)
        info = self.get_study_info()
        return {
            "study": info,
            "geom": geom_objects,
            "smesh": mesh_objects,
        }

    def create_box(self, dx: float, dy: float, dz: float, name: str) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        shape = geompy.MakeBoxDXDYDZ(dx, dy, dz)
        entry = geompy.addToStudy(shape, name)
        self._apply_visibility(show_entries=[entry])
        return {
            "message": f"Created box '{name}'",
            "entry": entry,
            "dimensions": {"dx": dx, "dy": dy, "dz": dz},
        }

    def create_cylinder(self, radius: float, height: float, name: str) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        shape = geompy.MakeCylinderRH(radius, height)
        entry = geompy.addToStudy(shape, name)
        self._apply_visibility(show_entries=[entry])
        return {
            "message": f"Created cylinder '{name}'",
            "entry": entry,
            "parameters": {"radius": radius, "height": height},
        }

    def create_sphere(self, radius: float, name: str) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        shape = geompy.MakeSphereR(radius)
        entry = geompy.addToStudy(shape, name)
        self._apply_visibility(show_entries=[entry])
        return {
            "message": f"Created sphere '{name}'",
            "entry": entry,
            "parameters": {"radius": radius},
        }

    def create_naca4_airfoil(
        self,
        code: str,
        chord: float,
        n_points: int,
        closed_te: bool,
        span: float,
        name: Optional[str],
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        naca = str(code).strip()
        if len(naca) != 4 or not naca.isdigit():
            raise RuntimeError("NACA code must be a 4-digit string (for example: '2412').")

        chord_value = float(chord)
        if chord_value <= 0.0:
            raise RuntimeError("chord must be > 0")

        sample_count = int(n_points)
        if sample_count < 5:
            raise RuntimeError("n_points must be >= 5")
        span_value = float(span)
        if span_value < 0.0:
            raise RuntimeError("span must be >= 0")

        m = int(naca[0]) / 100.0
        p = int(naca[1]) / 10.0
        t = int(naca[2:]) / 100.0
        if m > 0.0 and (p <= 0.0 or p >= 1.0):
            raise RuntimeError("Invalid camber location in NACA code; second digit must be 1..9 for cambered airfoils.")

        trailing_coeff = -0.1036 if bool(closed_te) else -0.1015

        x_values: List[float] = []
        for idx in range(sample_count):
            beta = math.pi * idx / (sample_count - 1)
            x_values.append(0.5 * (1.0 - math.cos(beta)))

        upper: List[Tuple[float, float, float]] = []
        lower: List[Tuple[float, float, float]] = []
        for x in x_values:
            yt = 5.0 * t * (
                0.2969 * math.sqrt(max(0.0, x))
                - 0.1260 * x
                - 0.3516 * x * x
                + 0.2843 * x * x * x
                + trailing_coeff * x * x * x * x
            )

            if m <= 0.0:
                yc = 0.0
                dyc_dx = 0.0
            elif x < p:
                yc = (m / (p * p)) * (2.0 * p * x - x * x)
                dyc_dx = (2.0 * m / (p * p)) * (p - x)
            else:
                one_minus_p = 1.0 - p
                yc = (m / (one_minus_p * one_minus_p)) * ((1.0 - 2.0 * p) + 2.0 * p * x - x * x)
                dyc_dx = (2.0 * m / (one_minus_p * one_minus_p)) * (p - x)

            theta = math.atan(dyc_dx)
            xu = (x - yt * math.sin(theta)) * chord_value
            yu = (yc + yt * math.cos(theta)) * chord_value
            xl = (x + yt * math.sin(theta)) * chord_value
            yl = (yc - yt * math.cos(theta)) * chord_value

            upper.append((xu, yu, 0.0))
            lower.append((xl, yl, 0.0))

        perimeter = list(reversed(upper)) + lower[1:]
        vertices = [geompy.MakeVertex(px, py, pz) for px, py, pz in perimeter]

        edges: List[Any] = []
        for idx in range(len(vertices) - 1):
            v1 = vertices[idx]
            v2 = vertices[idx + 1]
            try:
                edge = geompy.MakeLineTwoPnt(v1, v2)
            except Exception:
                edge = geompy.MakeEdge(v1, v2)
            edges.append(edge)

        first = perimeter[0]
        last = perimeter[-1]
        gap = math.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2 + (last[2] - first[2]) ** 2)
        if gap > max(1e-9, abs(chord_value) * 1e-9):
            try:
                closing_edge = geompy.MakeLineTwoPnt(vertices[-1], vertices[0])
            except Exception:
                closing_edge = geompy.MakeEdge(vertices[-1], vertices[0])
            edges.append(closing_edge)

        try:
            wire = geompy.MakeWire(edges, 1e-7)
        except TypeError:
            wire = geompy.MakeWire(edges)

        shape = wire
        shape_kind = "WIRE"
        try:
            shape = geompy.MakeFace(wire, True)
            shape_kind = "FACE"
        except TypeError:
            try:
                shape = geompy.MakeFace(wire, 1)
                shape_kind = "FACE"
            except Exception:
                pass
        except Exception:
            pass

        if span_value > 0.0:
            if shape_kind != "FACE":
                try:
                    shape = geompy.MakeFace(wire, True)
                    shape_kind = "FACE"
                except Exception:
                    raise RuntimeError("Could not build a face from the airfoil wire; cannot create 3D solid.")
            shape = geompy.MakePrismDXDYDZ(shape, 0.0, 0.0, span_value)
            shape_kind = "SOLID"

        shape_name = (name or f"NACA{naca}").strip() or f"NACA{naca}"
        entry = geompy.addToStudy(shape, shape_name)
        self._apply_visibility(show_entries=[entry])
        return {
            "message": f"Created NACA {naca} airfoil '{shape_name}'",
            "entry": entry,
            "shape_type": shape_kind,
            "code": naca,
            "parameters": {
                "chord": chord_value,
                "n_points": sample_count,
                "closed_te": bool(closed_te),
                "span": span_value,
                "m": m,
                "p": p,
                "t": t,
            },
            "point_count": len(perimeter),
        }

    def fuse_objects(
        self,
        base_object: str,
        tool_objects: Sequence[str],
        result_name: str,
    ) -> Dict[str, Any]:
        return self.boolean_operation("fuse", base_object, tool_objects, result_name)

    def cut_objects(
        self,
        base_object: str,
        tool_objects: Sequence[str],
        result_name: str,
    ) -> Dict[str, Any]:
        return self.boolean_operation("cut", base_object, tool_objects, result_name)

    def common_objects(
        self,
        base_object: str,
        tool_objects: Sequence[str],
        result_name: str,
    ) -> Dict[str, Any]:
        return self.boolean_operation("common", base_object, tool_objects, result_name)

    def copy_object(self, source_ref: str, name: str) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        shape, source_entry, _ = self._resolve_shape(source_ref)
        try:
            copied = geompy.MakeCopy(shape)
        except Exception:
            copied = geompy.MakeTranslation(shape, 0.0, 0.0, 0.0)

        entry = geompy.addToStudy(copied, name)
        self._apply_visibility(show_entries=[entry], hide_entries=[source_entry])
        return {
            "message": f"Created copy '{name}'",
            "entry": entry,
            "source": source_ref,
        }

    def duplicate_object(
        self,
        source_ref: str,
        count: int,
        name_prefix: Optional[str],
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")
        if count < 1:
            raise RuntimeError("count must be >= 1")

        shape, source_entry, source_name = self._resolve_shape(source_ref)
        prefix = (name_prefix or f"{source_name}_copy").strip()
        if not prefix:
            prefix = "Copy"

        items: List[Dict[str, Any]] = []
        for idx in range(count):
            copy_name = f"{prefix}_{idx + 1}"
            try:
                copied = geompy.MakeCopy(shape)
            except Exception:
                copied = geompy.MakeTranslation(shape, 0.0, 0.0, 0.0)
            entry = geompy.addToStudy(copied, copy_name)
            items.append({"name": copy_name, "entry": entry})

        self._apply_visibility(
            show_entries=[str(item["entry"]) for item in items],
            hide_entries=[source_entry],
        )
        return {
            "message": f"Created {count} duplicate(s) from '{source_ref}'",
            "source": source_ref,
            "count": count,
            "items": items,
        }

    def translate_object(
        self,
        source_ref: str,
        dx: float,
        dy: float,
        dz: float,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        dx_val = float(dx)
        dy_val = float(dy)
        dz_val = float(dz)
        shape, source_entry, source_name = self._resolve_shape(source_ref)

        translated_in_place = False
        errors: List[str] = []

        if hasattr(geompy, "TranslateDXDYDZ"):
            try:
                geompy.TranslateDXDYDZ(shape, dx_val, dy_val, dz_val, theCopy=False)
                translated_in_place = True
            except TypeError:
                try:
                    geompy.TranslateDXDYDZ(shape, dx_val, dy_val, dz_val, False)
                    translated_in_place = True
                except TypeError:
                    try:
                        geompy.TranslateDXDYDZ(shape, dx_val, dy_val, dz_val)
                        translated_in_place = True
                    except Exception as exc:
                        errors.append(f"TranslateDXDYDZ failed: {exc}")
                except Exception as exc:
                    errors.append(f"TranslateDXDYDZ failed: {exc}")
            except Exception as exc:
                errors.append(f"TranslateDXDYDZ failed: {exc}")

        if not translated_in_place and hasattr(geompy, "TranslateVector"):
            vector = geompy.MakeVectorDXDYDZ(dx_val, dy_val, dz_val)
            try:
                geompy.TranslateVector(shape, vector, theCopy=False)
                translated_in_place = True
            except TypeError:
                try:
                    geompy.TranslateVector(shape, vector, False)
                    translated_in_place = True
                except TypeError:
                    try:
                        geompy.TranslateVector(shape, vector)
                        translated_in_place = True
                    except Exception as exc:
                        errors.append(f"TranslateVector failed: {exc}")
                except Exception as exc:
                    errors.append(f"TranslateVector failed: {exc}")
            except Exception as exc:
                errors.append(f"TranslateVector failed: {exc}")

        if not translated_in_place:
            detail = "; ".join(errors) if errors else "No supported in-place translation API found."
            raise RuntimeError(f"Failed to translate object in place. {detail}")

        self._apply_visibility(show_entries=[source_entry])
        return {
            "message": f"Translated '{source_name}' in place",
            "entry": source_entry,
            "source": source_ref,
            "offset": {"dx": dx_val, "dy": dy_val, "dz": dz_val},
        }

    def rotate_object(
        self,
        source_ref: str,
        angle_degrees: float,
        axis: str,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        axis_key = axis.upper().strip()
        axis_map = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }
        if axis_key not in axis_map:
            raise RuntimeError("axis must be one of: X, Y, Z")

        shape, source_entry, source_name = self._resolve_shape(source_ref)
        axis_vec = geompy.MakeVectorDXDYDZ(*axis_map[axis_key])
        angle_radians = math.radians(float(angle_degrees))

        rotated_in_place = False
        errors: List[str] = []
        if hasattr(geompy, "Rotate"):
            try:
                geompy.Rotate(shape, axis_vec, angle_radians, theCopy=False)
                rotated_in_place = True
            except TypeError:
                try:
                    geompy.Rotate(shape, axis_vec, angle_radians, False)
                    rotated_in_place = True
                except TypeError:
                    try:
                        geompy.Rotate(shape, axis_vec, angle_radians)
                        rotated_in_place = True
                    except Exception as exc:
                        errors.append(f"Rotate failed: {exc}")
                except Exception as exc:
                    errors.append(f"Rotate failed: {exc}")
            except Exception as exc:
                errors.append(f"Rotate failed: {exc}")

        if not rotated_in_place:
            detail = "; ".join(errors) if errors else "No supported in-place rotation API found."
            raise RuntimeError(f"Failed to rotate object in place. {detail}")

        self._apply_visibility(show_entries=[source_entry])
        return {
            "message": f"Rotated '{source_name}' in place",
            "entry": source_entry,
            "source": source_ref,
            "axis": axis_key,
            "angle_degrees": float(angle_degrees),
            "angle_radians": angle_radians,
        }

    def rename_object(self, object_ref: str, new_name: str) -> Dict[str, Any]:
        self.initialize()
        salome = self.exec_globals["salome"]

        shape, entry, old_name = self._resolve_shape(object_ref)
        if not entry:
            raise RuntimeError(
                "Unable to rename object without study entry. Use an entry or a study object name."
            )

        sobj = salome.myStudy.FindObjectID(entry)
        renamed = False
        if sobj is not None:
            try:
                sobj.SetName(new_name)
                renamed = True
            except Exception:
                pass

            if not renamed:
                try:
                    builder = salome.myStudy.NewBuilder()
                    builder.SetName(sobj, new_name)
                    renamed = True
                except Exception:
                    pass

        if not renamed:
            try:
                shape.SetName(new_name)
                renamed = True
            except Exception:
                pass

        if not renamed:
            raise RuntimeError("Failed to rename object in study")

        self._refresh_gui()
        return {
            "message": f"Renamed object to '{new_name}'",
            "entry": entry,
            "old_name": old_name,
            "new_name": new_name,
        }

    def get_object_info(self, object_ref: str, precise_bbox: bool) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        shape, entry, name = self._resolve_shape(object_ref)
        bbox = geompy.BoundingBox(shape, bool(precise_bbox))
        props = geompy.BasicProperties(shape)
        what_is = geompy.WhatIs(shape)

        counts: Dict[str, int] = {}
        for kind in ("VERTEX", "EDGE", "FACE", "SOLID"):
            try:
                counts[kind] = int(geompy.NbShapes(shape, geompy.ShapeType[kind]))
            except Exception:
                counts[kind] = 0

        shape_type = None
        try:
            shape_type = int(shape.GetShapeType())
        except Exception:
            pass

        return {
            "object_ref": object_ref,
            "resolved_name": name,
            "entry": entry,
            "shape_type": shape_type,
            "what_is": what_is,
            "basic_properties": {
                "length": float(props[0]),
                "area": float(props[1]),
                "volume": float(props[2]),
            },
            "bounding_box": {
                "xmin": float(bbox[0]),
                "xmax": float(bbox[1]),
                "ymin": float(bbox[2]),
                "ymax": float(bbox[3]),
                "zmin": float(bbox[4]),
                "zmax": float(bbox[5]),
                "precise": bool(precise_bbox),
            },
            "subshape_counts": counts,
        }

    def delete_object(self, object_ref: str, with_children: bool) -> Dict[str, Any]:
        self.initialize()
        salome = self.exec_globals["salome"]

        _, entry, name = self._resolve_shape(object_ref)
        if not entry:
            raise RuntimeError(
                "Unable to delete object without study entry. Use an entry or a study object name."
            )

        study = salome.myStudy
        sobj = study.FindObjectID(entry)
        if sobj is None:
            raise RuntimeError(f"Study object not found for entry: {entry}")

        builder = study.NewBuilder()
        if with_children and hasattr(builder, "RemoveObjectWithChildren"):
            builder.RemoveObjectWithChildren(sobj)
        else:
            builder.RemoveObject(sobj)

        self._refresh_gui()
        return {
            "message": f"Deleted object '{name}'",
            "entry": entry,
            "with_children": bool(with_children),
        }

    def boolean_operation(
        self,
        operation: str,
        base_object: str,
        tool_objects: Sequence[str],
        result_name: str,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        base_shape, base_entry, _ = self._resolve_shape(base_object)
        resolved_tools = [self._resolve_shape(ref) for ref in tool_objects]
        tool_shapes = [item[0] for item in resolved_tools]
        tool_entries = [item[1] for item in resolved_tools]
        op = operation.lower().strip()

        if not tool_shapes:
            raise RuntimeError("At least one tool object is required")

        if op == "fuse":
            result = geompy.MakeFuseList([base_shape] + tool_shapes)
        elif op == "cut":
            result = geompy.MakeCutList(base_shape, tool_shapes, True)
        elif op == "common":
            result = geompy.MakeCommonList([base_shape] + tool_shapes)
        else:
            raise RuntimeError(f"Unsupported boolean operation: {operation}")

        entry = geompy.addToStudy(result, result_name)
        self._apply_visibility(
            show_entries=[entry],
            hide_entries=[base_entry] + tool_entries,
        )
        return {
            "message": f"Created boolean {op} '{result_name}'",
            "entry": entry,
            "operation": op,
            "base": base_object,
            "tools": list(tool_objects),
        }

    def list_subshapes(
        self,
        shape_ref: str,
        subshape_type: str,
        sorted_centres: bool,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        shape, _, _ = self._resolve_shape(shape_ref)
        shape_type_key = subshape_type.upper().strip()
        if shape_type_key not in geompy.ShapeType:
            raise RuntimeError(f"Unsupported subshape type: {subshape_type}")

        shape_type = geompy.ShapeType[shape_type_key]
        if sorted_centres:
            subs = geompy.SubShapeAllSortedCentres(shape, shape_type)
        else:
            subs = geompy.SubShapeAll(shape, shape_type)

        items: List[Dict[str, int]] = []
        for idx, sub in enumerate(subs):
            sub_id = int(geompy.GetSubShapeID(shape, sub))
            items.append({"index": idx, "id": sub_id})

        return {
            "shape": shape_ref,
            "subshape_type": shape_type_key,
            "count": len(items),
            "items": items,
        }

    def create_group(
        self,
        shape_ref: str,
        subshape_type: str,
        subshape_ids: Sequence[int],
        name: str,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        shape, _, _ = self._resolve_shape(shape_ref)
        shape_type_key = subshape_type.upper().strip()
        if shape_type_key not in geompy.ShapeType:
            raise RuntimeError(f"Unsupported subshape type: {subshape_type}")

        group = geompy.CreateGroup(shape, geompy.ShapeType[shape_type_key])
        geompy.UnionIDs(group, [int(i) for i in subshape_ids])

        if hasattr(geompy, "addToStudyInFather"):
            entry = geompy.addToStudyInFather(shape, group, name)
        else:
            entry = geompy.addToStudy(group, name)

        self._refresh_gui()
        return {
            "message": f"Created group '{name}'",
            "entry": entry,
            "shape": shape_ref,
            "subshape_type": shape_type_key,
            "subshape_ids": [int(i) for i in subshape_ids],
        }

    def create_groups(
        self,
        shape_ref: str,
        groups: Sequence[Dict[str, Any]],
        replace_existing: bool,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        salome = self.exec_globals.get("salome")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")
        if salome is None:
            raise RuntimeError("SALOME runtime is not available")

        if not isinstance(groups, (list, tuple)) or not groups:
            raise RuntimeError("groups must be a non-empty list")

        shape, _, _ = self._resolve_shape(shape_ref)
        created: List[Dict[str, Any]] = []
        removed: List[Dict[str, str]] = []

        study = salome.myStudy
        builder = study.NewBuilder()

        for index, raw_group in enumerate(groups):
            if not isinstance(raw_group, dict):
                raise RuntimeError(f"groups[{index}] must be an object")

            name = str(raw_group.get("name", f"Group_{index + 1}")).strip()
            if not name:
                raise RuntimeError(f"groups[{index}].name must not be empty")

            shape_type_key = str(raw_group.get("subshape_type", "FACE")).upper().strip()
            if shape_type_key not in geompy.ShapeType:
                raise RuntimeError(
                    f"groups[{index}].subshape_type '{shape_type_key}' is not supported"
                )

            raw_ids = raw_group.get("subshape_ids", [])
            if raw_ids is None:
                raw_ids = []
            if not isinstance(raw_ids, (list, tuple)):
                raise RuntimeError(f"groups[{index}].subshape_ids must be a list")
            subshape_ids = [int(i) for i in raw_ids]

            if replace_existing:
                same_name = list(study.FindObjectByName(name, "GEOM"))
                for sobj in same_name:
                    entry = self._sobj_entry(sobj)
                    if hasattr(builder, "RemoveObjectWithChildren"):
                        builder.RemoveObjectWithChildren(sobj)
                    else:
                        builder.RemoveObject(sobj)
                    removed.append({"name": name, "entry": entry})

            group = geompy.CreateGroup(shape, geompy.ShapeType[shape_type_key])
            if subshape_ids:
                geompy.UnionIDs(group, subshape_ids)

            if hasattr(geompy, "addToStudyInFather"):
                entry = geompy.addToStudyInFather(shape, group, name)
            else:
                entry = geompy.addToStudy(group, name)

            created.append(
                {
                    "name": name,
                    "entry": entry,
                    "subshape_type": shape_type_key,
                    "subshape_count": len(subshape_ids),
                    "subshape_ids": subshape_ids,
                }
            )

        self._refresh_gui()
        return {
            "message": f"Created {len(created)} group(s)",
            "shape": shape_ref,
            "replace_existing": bool(replace_existing),
            "removed": removed,
            "groups": created,
        }

    def create_surface_group(
        self,
        shape_ref: str,
        subshape_ids: Sequence[int],
        name: str,
    ) -> Dict[str, Any]:
        return self.create_group(shape_ref, "FACE", subshape_ids, name)

    def create_volume_group(
        self,
        shape_ref: str,
        subshape_ids: Sequence[int],
        name: str,
    ) -> Dict[str, Any]:
        return self.create_group(shape_ref, "SOLID", subshape_ids, name)

    def make_partition(
        self,
        object_refs: Sequence[str],
        tool_refs: Sequence[str],
        result_name: str,
        shape_type: str,
        keep_non_limit_shapes: bool,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        resolved_objects = [self._resolve_shape(ref) for ref in object_refs]
        resolved_tools = [self._resolve_shape(ref) for ref in tool_refs]
        objects = [item[0] for item in resolved_objects]
        tools = [item[0] for item in resolved_tools]
        hidden_entries = [item[1] for item in resolved_objects + resolved_tools]
        if not objects:
            raise RuntimeError("At least one object is required for partition")

        shape_type_key = shape_type.upper().strip() if shape_type else "SOLID"

        try:
            result = geompy.MakePartition(
                objects,
                tools,
                [],
                [],
                geompy.ShapeType[shape_type_key],
                0,
                [],
                0,
                KeepNonlimitShapes=1 if keep_non_limit_shapes else 0,
            )
        except Exception:
            result = geompy.MakePartition(objects, tools)

        entry = geompy.addToStudy(result, result_name)
        self._apply_visibility(show_entries=[entry], hide_entries=hidden_entries)
        return {
            "message": f"Created partition '{result_name}'",
            "entry": entry,
            "objects": list(object_refs),
            "tools": list(tool_refs),
            "shape_type": shape_type_key,
        }

    def explode_shape(
        self,
        shape_ref: str,
        subshape_type: str,
        result_prefix: str,
        add_to_study: bool,
        sorted_centres: bool,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        shape, _, _ = self._resolve_shape(shape_ref)
        shape_type_key = subshape_type.upper().strip()
        if shape_type_key not in geompy.ShapeType:
            raise RuntimeError(f"Unsupported subshape type: {subshape_type}")

        shape_type = geompy.ShapeType[shape_type_key]
        if sorted_centres:
            subs = geompy.SubShapeAllSortedCentres(shape, shape_type)
        else:
            subs = geompy.SubShapeAll(shape, shape_type)

        exploded: List[Dict[str, Any]] = []
        for idx, sub in enumerate(subs):
            sub_id = int(geompy.GetSubShapeID(shape, sub))
            item: Dict[str, Any] = {"index": idx, "id": sub_id}
            if add_to_study:
                sub_name = f"{result_prefix}_{idx + 1}"
                if hasattr(geompy, "addToStudyInFather"):
                    entry = geompy.addToStudyInFather(shape, sub, sub_name)
                else:
                    entry = geompy.addToStudy(sub, sub_name)
                item["entry"] = entry
                item["name"] = sub_name
            exploded.append(item)

        self._refresh_gui()
        return {
            "shape": shape_ref,
            "subshape_type": shape_type_key,
            "count": len(exploded),
            "items": exploded,
        }

    def import_geometry(
        self,
        filepath: str,
        fmt: str,
        name: Optional[str],
        ignore_units: bool,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        fmt_norm = self._normalized_format(filepath, fmt)

        if fmt_norm == "brep":
            shape = geompy.ImportBREP(filepath)
        elif fmt_norm == "step":
            try:
                shape = geompy.ImportSTEP(filepath, bool(ignore_units))
            except TypeError:
                shape = geompy.ImportSTEP(filepath)
        elif fmt_norm == "iges":
            try:
                shape = geompy.ImportIGES(filepath, bool(ignore_units))
            except TypeError:
                shape = geompy.ImportIGES(filepath)
        else:
            raise RuntimeError(f"Unsupported geometry import format: {fmt_norm}")

        obj_name = name or f"Imported_{Path(filepath).stem}"
        entry = geompy.addToStudy(shape, obj_name)
        self._apply_visibility(show_entries=[entry])
        return {
            "message": f"Imported geometry '{obj_name}'",
            "entry": entry,
            "format": fmt_norm,
            "filepath": filepath,
        }

    def export_geometry(
        self,
        shape_ref: str,
        filepath: str,
        fmt: str,
        ascii_stl: bool,
        stl_deflection: float,
    ) -> Dict[str, Any]:
        self.initialize()
        geompy = self.exec_globals.get("geompy")
        if geompy is None:
            raise RuntimeError("GEOM is not available in this SALOME session")

        shape, _, _ = self._resolve_shape(shape_ref)
        fmt_norm = self._normalized_format(filepath, fmt)

        if fmt_norm == "brep":
            geompy.ExportBREP(shape, filepath)
        elif fmt_norm == "step":
            geompy.ExportSTEP(shape, filepath)
        elif fmt_norm == "iges":
            geompy.ExportIGES(shape, filepath)
        elif fmt_norm == "stl":
            try:
                geompy.ExportSTL(shape, filepath, bool(ascii_stl), float(stl_deflection))
            except TypeError:
                geompy.ExportSTL(shape, filepath, bool(ascii_stl))
        else:
            raise RuntimeError(f"Unsupported geometry export format: {fmt_norm}")

        return {
            "message": "Geometry exported",
            "shape": shape_ref,
            "format": fmt_norm,
            "filepath": filepath,
        }

    def import_mesh(self, filepath: str, fmt: str, name: Optional[str]) -> Dict[str, Any]:
        self.initialize()
        smesh = self.exec_globals.get("smesh")
        if smesh is None:
            raise RuntimeError("SMESH is not available in this SALOME session")

        fmt_norm = self._normalized_format(filepath, fmt)

        if fmt_norm == "med":
            meshes, status = smesh.CreateMeshesFromMED(filepath)
            mesh = self._extract_first_mesh(meshes)
            if not status:
                raise RuntimeError("CreateMeshesFromMED returned unsuccessful status")
        elif fmt_norm == "unv":
            mesh = smesh.CreateMeshesFromUNV(filepath)
        elif fmt_norm == "stl":
            mesh = smesh.CreateMeshesFromSTL(filepath)
        elif fmt_norm == "cgns":
            meshes, status = smesh.CreateMeshesFromCGNS(filepath)
            mesh = self._extract_first_mesh(meshes)
            if not status:
                raise RuntimeError("CreateMeshesFromCGNS returned unsuccessful status")
        elif fmt_norm == "mesh":
            mesh = smesh.CreateMeshesFromGMF(filepath)[0]
        else:
            raise RuntimeError(f"Unsupported mesh import format: {fmt_norm}")

        if mesh is None:
            raise RuntimeError("Imported mesh is empty")

        mesh_name = self._register_mesh(mesh, name)
        self._refresh_gui()
        return {
            "message": f"Imported mesh '{mesh_name}'",
            "mesh_name": mesh_name,
            "format": fmt_norm,
            "filepath": filepath,
        }

    def export_mesh(
        self,
        mesh_ref: str,
        filepath: str,
        fmt: str,
        ascii_stl: bool,
        auto_dimension: bool,
    ) -> Dict[str, Any]:
        self.initialize()
        mesh, mesh_name = self._resolve_mesh(mesh_ref)
        fmt_norm = self._normalized_format(filepath, fmt)

        if fmt_norm == "med":
            try:
                mesh.ExportMED(filepath, autoDimension=bool(auto_dimension))
            except TypeError:
                try:
                    mesh.ExportMED(filepath, 0)
                except TypeError:
                    mesh.ExportMED(filepath)
        elif fmt_norm == "unv":
            mesh.ExportUNV(filepath)
        elif fmt_norm == "stl":
            mesh.ExportSTL(filepath, bool(ascii_stl))
        else:
            raise RuntimeError(f"Unsupported mesh export format: {fmt_norm}")

        return {
            "message": "Mesh exported",
            "mesh": mesh_name,
            "format": fmt_norm,
            "filepath": filepath,
        }

    def create_mesh(
        self,
        shape_ref: str,
        mesh_name: str,
        segment_count: int,
        max_element_area: Optional[float],
        max_element_volume: Optional[float],
        surface_algorithm: str,
        volume_algorithm: str,
    ) -> Dict[str, Any]:
        self.initialize()
        smesh = self.exec_globals.get("smesh")
        if smesh is None:
            raise RuntimeError("SMESH is not available in this SALOME session")

        shape, _, _ = self._resolve_shape(shape_ref)
        mesh = smesh.Mesh(shape, mesh_name)

        if segment_count > 0:
            algo_1d = mesh.Segment()
            algo_1d.NumberOfSegments(int(segment_count))

        surf_algo_name = surface_algorithm.lower().strip()
        if surf_algo_name == "triangle":
            algo_2d = mesh.Triangle()
            if max_element_area is not None:
                algo_2d.MaxElementArea(float(max_element_area))
        elif surf_algo_name == "quadrangle":
            mesh.Quadrangle()
        elif surf_algo_name == "none":
            pass
        else:
            raise RuntimeError(f"Unsupported surface_algorithm: {surface_algorithm}")

        vol_algo_name = volume_algorithm.lower().strip()
        if vol_algo_name in ("tetra", "tetrahedron"):
            algo_3d = mesh.Tetrahedron()
            if max_element_volume is not None:
                algo_3d.MaxElementVolume(float(max_element_volume))
        elif vol_algo_name in ("hexa", "hexahedron"):
            mesh.Hexahedron()
        elif vol_algo_name == "none":
            pass
        else:
            raise RuntimeError(f"Unsupported volume_algorithm: {volume_algorithm}")

        registered = self._register_mesh(mesh, mesh_name)
        self._refresh_gui()
        return {
            "message": f"Created mesh definition '{registered}'",
            "mesh_name": registered,
            "shape": shape_ref,
            "segment_count": int(segment_count),
            "surface_algorithm": surf_algo_name,
            "volume_algorithm": vol_algo_name,
        }

    def create_mesh_with_hypotheses(
        self,
        shape_ref: str,
        mesh_name: str,
        algorithm: str,
        hypotheses: Optional[Dict[str, Any]],
        compute: bool,
    ) -> Dict[str, Any]:
        self.initialize()
        smesh = self.exec_globals.get("smesh")
        smesh_builder = self.exec_globals.get("smeshBuilder")
        if smesh is None or smesh_builder is None:
            raise RuntimeError("SMESH is not available in this SALOME session")

        shape, _, _ = self._resolve_shape(shape_ref)
        mesh = smesh.Mesh(shape, mesh_name)

        algo_key = str(algorithm).lower().strip().replace("-", "_").replace(" ", "_")

        netgen_algorithms = {
            "netgen_1d2d3d": ("Tetrahedron", "NETGEN_1D2D3D"),
            "netgen_2d3d": ("Tetrahedron", "NETGEN_2D3D"),
            "netgen_1d2d": ("Triangle", "NETGEN_1D2D"),
            "netgen_2d": ("Triangle", "NETGEN_2D"),
        }
        aliases = {
            "netgen": "netgen_1d2d3d",
            "tetra": "tetrahedron",
            "hexa": "hexahedron",
            "quad": "quadrangle",
        }
        algo_key = aliases.get(algo_key, algo_key)

        if algo_key in netgen_algorithms:
            method_name, builder_const = netgen_algorithms[algo_key]
            algo_factory = getattr(mesh, method_name, None)
            builder_algo = getattr(smesh_builder, builder_const, None)
            if algo_factory is None or builder_algo is None:
                raise RuntimeError(
                    f"Algorithm '{algorithm}' is not available in this SALOME build"
                )
            algo = algo_factory(algo=builder_algo)
        elif algo_key == "tetrahedron":
            algo = mesh.Tetrahedron()
        elif algo_key == "hexahedron":
            algo = mesh.Hexahedron()
        elif algo_key == "triangle":
            algo = mesh.Triangle()
        elif algo_key == "quadrangle":
            algo = mesh.Quadrangle()
        else:
            raise RuntimeError(
                "Unsupported algorithm. Use one of: "
                "netgen_1d2d3d, netgen_2d3d, netgen_1d2d, netgen_2d, "
                "tetrahedron, hexahedron, triangle, quadrangle"
            )

        hyp = hypotheses or {}
        if not isinstance(hyp, dict):
            raise RuntimeError("hypotheses must be an object (dictionary)")

        applied: Dict[str, Any] = {}
        ignored: Dict[str, str] = {}

        if "segment_count" in hyp and hyp.get("segment_count") is not None:
            seg_count = int(hyp["segment_count"])
            if seg_count > 0:
                seg_algo = mesh.Segment()
                seg_algo.NumberOfSegments(seg_count)
                applied["segment_count"] = seg_count
            else:
                raise RuntimeError("hypotheses.segment_count must be > 0 when provided")

        if "max_element_area" in hyp and hyp.get("max_element_area") is not None:
            if hasattr(algo, "MaxElementArea"):
                value = float(hyp["max_element_area"])
                algo.MaxElementArea(value)
                applied["max_element_area"] = value
            else:
                ignored["max_element_area"] = "Algorithm does not support MaxElementArea"

        if "max_element_volume" in hyp and hyp.get("max_element_volume") is not None:
            if hasattr(algo, "MaxElementVolume"):
                value = float(hyp["max_element_volume"])
                algo.MaxElementVolume(value)
                applied["max_element_volume"] = value
            else:
                ignored["max_element_volume"] = "Algorithm does not support MaxElementVolume"

        params = None
        if hasattr(algo, "Parameters"):
            try:
                params = algo.Parameters()
            except Exception:
                params = None

        if "fineness" in hyp and hyp.get("fineness") is not None:
            if params is None or not hasattr(params, "SetFineness"):
                ignored["fineness"] = "Algorithm does not expose NETGEN parameters"
            else:
                fineness_name = str(hyp["fineness"]).lower().strip().replace("-", "_")
                fineness_aliases = {
                    "very_coarse": "VeryCoarse",
                    "coarse": "Coarse",
                    "moderate": "Moderate",
                    "medium": "Moderate",
                    "fine": "Fine",
                    "very_fine": "VeryFine",
                    "user_defined": "UserDefined",
                    "custom": "UserDefined",
                }
                const_name = fineness_aliases.get(fineness_name)
                if not const_name:
                    raise RuntimeError(
                        "Unsupported fineness value. Use one of: very_coarse, coarse, "
                        "moderate, fine, very_fine, user_defined"
                    )
                const_value = getattr(smesh_builder, const_name, None)
                if const_value is None:
                    raise RuntimeError(f"SMESH fineness constant '{const_name}' is unavailable")
                params.SetFineness(const_value)
                applied["fineness"] = fineness_name

        netgen_number_settings = {
            "max_size": ("SetMaxSize", float),
            "min_size": ("SetMinSize", float),
            "growth_rate": ("SetGrowthRate", float),
            "nb_seg_per_edge": ("SetNbSegPerEdge", int),
            "nb_seg_per_radius": ("SetNbSegPerRadius", int),
            "chordal_error": ("SetChordalError", float),
        }
        netgen_flag_settings = {
            "second_order": "SetSecondOrder",
            "optimize": "SetOptimize",
            "use_surface_curvature": "SetUseSurfaceCurvature",
            "fuse_edges": "SetFuseEdges",
            "quad_allowed": "SetQuadAllowed",
            "chordal_error_enabled": "SetChordalErrorEnabled",
        }

        for key, (method_name, caster) in netgen_number_settings.items():
            if key not in hyp or hyp.get(key) is None:
                continue
            if params is None or not hasattr(params, method_name):
                ignored[key] = "Algorithm does not expose NETGEN parameter method"
                continue
            value = caster(hyp[key])
            getattr(params, method_name)(value)
            applied[key] = value

        for key, method_name in netgen_flag_settings.items():
            if key not in hyp or hyp.get(key) is None:
                continue
            if params is None or not hasattr(params, method_name):
                ignored[key] = "Algorithm does not expose NETGEN parameter method"
                continue
            value = 1 if bool(hyp[key]) else 0
            getattr(params, method_name)(value)
            applied[key] = bool(hyp[key])

        known_hyp_keys = {
            "segment_count",
            "max_element_area",
            "max_element_volume",
            "fineness",
            "max_size",
            "min_size",
            "growth_rate",
            "nb_seg_per_edge",
            "nb_seg_per_radius",
            "chordal_error",
            "second_order",
            "optimize",
            "use_surface_curvature",
            "fuse_edges",
            "quad_allowed",
            "chordal_error_enabled",
        }
        unknown_hypotheses = sorted([str(k) for k in hyp.keys() if k not in known_hyp_keys])

        registered = self._register_mesh(mesh, mesh_name)
        compute_success = None
        mesh_info = None
        if compute:
            compute_success = bool(mesh.Compute())
            mesh_info = self.get_mesh_info(registered)

        self._refresh_gui()
        return {
            "message": f"Created mesh definition '{registered}'",
            "mesh_name": registered,
            "shape": shape_ref,
            "algorithm": algo_key,
            "hypotheses_applied": applied,
            "hypotheses_ignored": ignored,
            "unknown_hypotheses": unknown_hypotheses,
            "computed": bool(compute),
            "compute_success": compute_success,
            "mesh": mesh_info,
        }

    def get_mesh_info(self, mesh_ref: str) -> Dict[str, Any]:
        self.initialize()
        smesh = self.exec_globals.get("smesh")
        mesh, mesh_name = self._resolve_mesh(mesh_ref)

        info = {
            "mesh_name": mesh_name,
            "nodes": int(mesh.NbNodes()) if hasattr(mesh, "NbNodes") else 0,
            "edges": int(mesh.NbEdges()) if hasattr(mesh, "NbEdges") else 0,
            "faces": int(mesh.NbFaces()) if hasattr(mesh, "NbFaces") else 0,
            "volumes": int(mesh.NbVolumes()) if hasattr(mesh, "NbVolumes") else 0,
            "triangles": int(mesh.NbTriangles()) if hasattr(mesh, "NbTriangles") else 0,
            "quadrangles": int(mesh.NbQuadrangles()) if hasattr(mesh, "NbQuadrangles") else 0,
            "tetrahedrons": int(mesh.NbTetras()) if hasattr(mesh, "NbTetras") else 0,
            "hexahedrons": int(mesh.NbHexas()) if hasattr(mesh, "NbHexas") else 0,
        }

        detailed: Dict[str, int] = {}
        if smesh is not None and hasattr(smesh, "GetMeshInfo"):
            try:
                mesh_arg = mesh.GetMesh() if hasattr(mesh, "GetMesh") else mesh
                raw = smesh.GetMeshInfo(mesh_arg)
                for key, value in raw.items():
                    detailed[str(key)] = int(value)
            except Exception:
                detailed = {}

        info["detailed"] = detailed
        return info

    def compute_mesh(self, mesh_ref: str) -> Dict[str, Any]:
        self.initialize()
        mesh, mesh_name = self._resolve_mesh(mesh_ref)
        if not hasattr(mesh, "Compute"):
            raise RuntimeError(
                "Resolved mesh object does not support Compute(). Recreate or reselect mesh."
            )
        success = bool(mesh.Compute())
        self._refresh_gui()

        return {
            "message": f"Computed mesh '{mesh_name}'" if success else f"Mesh compute failed for '{mesh_name}'",
            "success": success,
            "mesh": self.get_mesh_info(mesh_name),
        }

    def execute_code(self, code: str) -> Dict[str, Any]:
        self.initialize()

        stream = io.StringIO()
        local_scope: Dict[str, Any] = {}

        with redirect_stdout(stream):
            exec(code, self.exec_globals, local_scope)

        result_value = local_scope.get("result")
        return {
            "stdout": stream.getvalue(),
            "result": repr(result_value),
        }


class SalomeBridgeServer:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self.runtime = SalomeRuntime()
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.last_error: Optional[str] = None
        self._start_cancelled = False
        self._state_lock = threading.Lock()

    def start(self) -> None:
        with self._state_lock:
            if self.running:
                return
            self.last_error = None
            self._start_cancelled = False

        try:
            # Refuse to run in headless sessions.
            self.runtime.initialize()

            with self._state_lock:
                if self._start_cancelled:
                    logger.info(
                        "SALOME bridge start cancelled before socket bind on %s:%s",
                        self.host,
                        self.port,
                    )
                    return
                self.running = True

            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)
        except Exception as exc:
            self.last_error = str(exc)
            logger.error(
                "Failed to start SALOME bridge on %s:%s: %s",
                self.host,
                self.port,
                exc,
            )
            self.running = False
            if self.server_socket:
                try:
                    self.server_socket.close()
                except Exception:
                    pass
                self.server_socket = None
            return

        logger.info("SALOME bridge listening on %s:%s", self.host, self.port)

        try:
            while self.running:
                try:
                    client, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.running:
                        logger.error("Bridge socket unexpectedly closed")
                    break

                logger.debug("Client connected from %s:%s", addr[0], addr[1])
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client,),
                    daemon=True,
                )
                thread.start()
        finally:
            self.stop()

    def stop(self) -> None:
        with self._state_lock:
            self._start_cancelled = True
            self.running = False
            if self.server_socket:
                try:
                    self.server_socket.close()
                except Exception:
                    pass
                self.server_socket = None
        logger.info("SALOME bridge stopped")

    def _handle_client(self, client: socket.socket) -> None:
        buffer = b""
        client.settimeout(None)

        try:
            while self.running:
                data = client.recv(8192)
                if not data:
                    break

                buffer += data
                try:
                    command = json.loads(buffer.decode("utf-8"))
                    buffer = b""
                except json.JSONDecodeError:
                    continue

                response = self._execute_command(command)
                client.sendall(json.dumps(response).encode("utf-8"))
        except Exception as exc:
            logger.error("Client handling error: %s", exc)
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        command_type = command.get("type")
        params = command.get("params", {})

        def _translate_handler(p: Dict[str, Any]) -> Dict[str, Any]:
            if "result_name" in p:
                raise RuntimeError(
                    "translate_object no longer accepts 'result_name'; transforms are now in place."
                )
            return self.runtime.translate_object(
                str(p["source_ref"]),
                float(p.get("dx", 0.0)),
                float(p.get("dy", 0.0)),
                float(p.get("dz", 0.0)),
            )

        def _rotate_handler(p: Dict[str, Any]) -> Dict[str, Any]:
            if "result_name" in p:
                raise RuntimeError(
                    "rotate_object no longer accepts 'result_name'; transforms are now in place."
                )
            return self.runtime.rotate_object(
                str(p["source_ref"]),
                float(p["angle_degrees"]),
                str(p.get("axis", "Z")),
            )

        handlers = {
            "ping": lambda _params: self.runtime.ping(),
            "get_study_info": lambda _params: self.runtime.get_study_info(),
            "get_scene_summary": lambda p: self.runtime.get_scene_summary(
                int(p.get("limit_per_component", 200))
            ),
            "list_study_objects": lambda p: self.runtime.list_study_objects(
                p.get("component"), int(p.get("limit", 200))
            ),
            "create_box": lambda p: self.runtime.create_box(
                float(p["dx"]), float(p["dy"]), float(p["dz"]), str(p.get("name", "Box"))
            ),
            "create_cylinder": lambda p: self.runtime.create_cylinder(
                float(p["radius"]), float(p["height"]), str(p.get("name", "Cylinder"))
            ),
            "create_sphere": lambda p: self.runtime.create_sphere(
                float(p["radius"]), str(p.get("name", "Sphere"))
            ),
            "create_naca4_airfoil": lambda p: self.runtime.create_naca4_airfoil(
                str(p["code"]),
                float(p.get("chord", 1.0)),
                int(p.get("n_points", 121)),
                bool(p.get("closed_te", True)),
                float(p.get("span", 0.01)),
                p.get("name"),
            ),
            "boolean_operation": lambda p: self.runtime.boolean_operation(
                str(p["operation"]),
                str(p["base_object"]),
                [str(x) for x in p.get("tool_objects", [])],
                str(p.get("result_name", "BooleanResult")),
            ),
            "fuse_objects": lambda p: self.runtime.fuse_objects(
                str(p["base_object"]),
                [str(x) for x in p.get("tool_objects", [])],
                str(p.get("result_name", "Fuse")),
            ),
            "cut_objects": lambda p: self.runtime.cut_objects(
                str(p["base_object"]),
                [str(x) for x in p.get("tool_objects", [])],
                str(p.get("result_name", "Cut")),
            ),
            "common_objects": lambda p: self.runtime.common_objects(
                str(p["base_object"]),
                [str(x) for x in p.get("tool_objects", [])],
                str(p.get("result_name", "Common")),
            ),
            "copy_object": lambda p: self.runtime.copy_object(
                str(p["source_ref"]),
                str(p.get("name", "Copy")),
            ),
            "duplicate_object": lambda p: self.runtime.duplicate_object(
                str(p["source_ref"]),
                int(p.get("count", 1)),
                p.get("name_prefix"),
            ),
            "translate_object": _translate_handler,
            "rotate_object": _rotate_handler,
            "rename_object": lambda p: self.runtime.rename_object(
                str(p["object_ref"]),
                str(p["new_name"]),
            ),
            "get_object_info": lambda p: self.runtime.get_object_info(
                str(p["object_ref"]),
                bool(p.get("precise_bbox", False)),
            ),
            "delete_object": lambda p: self.runtime.delete_object(
                str(p["object_ref"]),
                bool(p.get("with_children", True)),
            ),
            "list_subshapes": lambda p: self.runtime.list_subshapes(
                str(p["shape_ref"]),
                str(p.get("subshape_type", "FACE")),
                bool(p.get("sorted_centres", True)),
            ),
            "create_group": lambda p: self.runtime.create_group(
                str(p["shape_ref"]),
                str(p["subshape_type"]),
                [int(x) for x in p.get("subshape_ids", [])],
                str(p.get("name", "Group")),
            ),
            "create_groups": lambda p: self.runtime.create_groups(
                str(p["shape_ref"]),
                p.get("groups", []),
                bool(p.get("replace_existing", False)),
            ),
            "create_surface_group": lambda p: self.runtime.create_surface_group(
                str(p["shape_ref"]),
                [int(x) for x in p.get("subshape_ids", [])],
                str(p.get("name", "SurfaceGroup")),
            ),
            "create_volume_group": lambda p: self.runtime.create_volume_group(
                str(p["shape_ref"]),
                [int(x) for x in p.get("subshape_ids", [])],
                str(p.get("name", "VolumeGroup")),
            ),
            "make_partition": lambda p: self.runtime.make_partition(
                [str(x) for x in p.get("object_refs", [])],
                [str(x) for x in p.get("tool_refs", [])],
                str(p.get("result_name", "Partition")),
                str(p.get("shape_type", "SOLID")),
                bool(p.get("keep_non_limit_shapes", False)),
            ),
            "explode_shape": lambda p: self.runtime.explode_shape(
                str(p["shape_ref"]),
                str(p.get("subshape_type", "FACE")),
                str(p.get("result_prefix", "Exploded")),
                bool(p.get("add_to_study", True)),
                bool(p.get("sorted_centres", True)),
            ),
            "import_geometry": lambda p: self.runtime.import_geometry(
                str(p["filepath"]),
                str(p.get("format", "auto")),
                p.get("name"),
                bool(p.get("ignore_units", False)),
            ),
            "export_geometry": lambda p: self.runtime.export_geometry(
                str(p["shape_ref"]),
                str(p["filepath"]),
                str(p.get("format", "auto")),
                bool(p.get("ascii_stl", True)),
                float(p.get("stl_deflection", 0.001)),
            ),
            "import_mesh": lambda p: self.runtime.import_mesh(
                str(p["filepath"]),
                str(p.get("format", "auto")),
                p.get("name"),
            ),
            "export_mesh": lambda p: self.runtime.export_mesh(
                str(p["mesh_ref"]),
                str(p["filepath"]),
                str(p.get("format", "auto")),
                bool(p.get("ascii_stl", True)),
                bool(p.get("auto_dimension", True)),
            ),
            "create_mesh": lambda p: self.runtime.create_mesh(
                str(p["shape_ref"]),
                str(p.get("mesh_name", "Mesh")),
                int(p.get("segment_count", 10)),
                None if p.get("max_element_area") is None else float(p.get("max_element_area")),
                None
                if p.get("max_element_volume") is None
                else float(p.get("max_element_volume")),
                str(p.get("surface_algorithm", "triangle")),
                str(p.get("volume_algorithm", "tetrahedron")),
            ),
            "create_mesh_with_hypotheses": lambda p: self.runtime.create_mesh_with_hypotheses(
                str(p["shape_ref"]),
                str(p.get("mesh_name", "Mesh")),
                str(p.get("algorithm", "netgen_1d2d3d")),
                p.get("hypotheses"),
                bool(p.get("compute", False)),
            ),
            "compute_mesh": lambda p: self.runtime.compute_mesh(str(p["mesh_ref"])),
            "get_mesh_info": lambda p: self.runtime.get_mesh_info(str(p["mesh_ref"])),
            "execute_code": lambda p: self.runtime.execute_code(str(p["code"])),
        }

        handler = handlers.get(command_type)
        if handler is None:
            try:
                self.runtime._refresh_gui()
            except Exception:
                pass
            return {
                "status": "error",
                "message": f"Unknown command type: {command_type}",
            }

        try:
            result = handler(params)
            return {"status": "success", "result": result}
        except Exception as exc:
            logger.error("Command '%s' failed: %s", command_type, exc)
            logger.debug("%s", traceback.format_exc())
            return {"status": "error", "message": str(exc)}
        finally:
            try:
                self.runtime._refresh_gui()
            except Exception:
                pass


def main() -> None:
    host = os.getenv("SALOME_HOST", DEFAULT_HOST)
    port = int(os.getenv("SALOME_PORT", DEFAULT_PORT))

    server = SalomeBridgeServer(host=host, port=port)
    server.start()


if __name__ == "__main__":
    main()
