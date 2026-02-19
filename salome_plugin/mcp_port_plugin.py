"""SALOME GUI plugin helpers to manage MCP bridge ports."""

import os
import socket
import sys
import threading
import time
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from qtsalome import QInputDialog, QLabel, QMessageBox, QTimer

# Ensure repository root (which contains salome_bridge.py) is importable.
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import salome_bridge

DEFAULT_HOST = os.getenv("SALOME_MCP_BRIDGE_HOST", "localhost")
DEFAULT_PORT = int(os.getenv("SALOME_MCP_BRIDGE_PORT", "1234"))
START_TIMEOUT = float(os.getenv("SALOME_MCP_BRIDGE_START_TIMEOUT", "12.0"))
STATUS_POLL_INTERVAL_MS = int(os.getenv("SALOME_MCP_STATUS_POLL_MS", "3000"))


@dataclass
class BridgeState:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    server: Optional[SalomeBridgeServer] = None
    thread: Optional[threading.Thread] = None


_STATE = BridgeState()
_LOCK = threading.Lock()
_STATUS_LABEL: Optional[QLabel] = None
_STATUS_TIMER: Optional[QTimer] = None
_QUIT_HOOK_INSTALLED = False


def _desktop(context):
    try:
        return context.sg.getDesktop()
    except Exception:
        return None


def _desktop_from_app():
    try:
        import SalomePyQt

        return SalomePyQt.SalomePyQt().getDesktop()
    except Exception:
        return None


def _can_bind(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host == "localhost" else host
    try:
        addr_infos = socket.getaddrinfo(
            bind_host,
            port,
            0,
            socket.SOCK_STREAM,
            0,
            socket.AI_PASSIVE,
        )
    except Exception:
        return False

    for family, socktype, proto, _, sockaddr in addr_infos:
        sock = None
        try:
            sock = socket.socket(family, socktype, proto)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            return True
        except Exception:
            continue
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    return False


def _running() -> bool:
    return _STATE.server is not None and _STATE.server.running


def _starting() -> bool:
    return (
        _STATE.server is not None
        and not _STATE.server.running
        and _STATE.thread is not None
        and _STATE.thread.is_alive()
    )


def _status_text() -> str:
    if _running():
        return (
            f"Bridge is running on {_STATE.host}:{_STATE.port}\n"
            f"Use SALOME_PORT={_STATE.port} in your MCP client environment."
        )
    if _starting():
        return f"Bridge is starting on {_STATE.host}:{_STATE.port}..."
    return "Bridge is stopped."


def _status_line() -> str:
    if _running():
        return f"MCP Bridge: RUNNING {_STATE.host}:{_STATE.port}"
    if _starting():
        return f"MCP Bridge: STARTING {_STATE.host}:{_STATE.port}"
    return "MCP Bridge: STOPPED"


def _status_style() -> str:
    if _running():
        return "QLabel { color: #0b7a0b; font-weight: 600; }"
    if _starting():
        return "QLabel { color: #006b9a; font-weight: 600; }"
    return "QLabel { color: #9a1b1b; font-weight: 600; }"


def _refresh_status_widget() -> None:
    global _STATUS_LABEL
    if _STATUS_LABEL is None:
        return
    _STATUS_LABEL.setText(_status_line())
    _STATUS_LABEL.setStyleSheet(_status_style())


def install_live_status_widget(context=None) -> None:
    """Install/update persistent status-bar badge for bridge state."""
    global _STATUS_LABEL, _STATUS_TIMER

    desktop = _desktop(context) if context is not None else _desktop_from_app()
    if desktop is None:
        return

    try:
        status_bar = desktop.statusBar()
    except Exception:
        return

    if _STATUS_LABEL is None:
        label = status_bar.findChild(QLabel, "SalomeMCPBridgeStatusLabel")
        if label is None:
            label = QLabel("")
            label.setObjectName("SalomeMCPBridgeStatusLabel")
            status_bar.addPermanentWidget(label)
        _STATUS_LABEL = label

    if _STATUS_TIMER is None:
        timer = QTimer(_STATUS_LABEL)
        timer.setInterval(max(500, STATUS_POLL_INTERVAL_MS))
        timer.timeout.connect(_refresh_status_widget)
        timer.start()
        _STATUS_TIMER = timer

    _refresh_status_widget()


def _show_info(context, title: str, text: str) -> None:
    QMessageBox.information(_desktop(context), title, text)


def _show_error(context, title: str, text: str) -> None:
    QMessageBox.critical(_desktop(context), title, text)


def _stop_bridge_internal(show_info: bool, context=None) -> bool:
    install_live_status_widget(context)

    with _LOCK:
        server = _STATE.server
        thread = _STATE.thread
        if server is None:
            if show_info:
                _show_info(context, "SALOME MCP Bridge", "Bridge is already stopped.")
            return False

        server.stop()

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=1.0)

        _STATE.server = None
        _STATE.thread = None

    if show_info:
        _show_info(context, "SALOME MCP Bridge", "Bridge stopped. Port is now closed.")
    _refresh_status_widget()
    return True


def _on_app_about_to_quit() -> None:
    try:
        _stop_bridge_internal(show_info=False, context=None)
    except Exception:
        pass


def install_quit_hook() -> None:
    """Install one-time Qt aboutToQuit hook to ensure bridge port closes on exit."""
    global _QUIT_HOOK_INSTALLED
    if _QUIT_HOOK_INSTALLED:
        return

    app = None
    try:
        from qtsalome import QApplication  # type: ignore

        app = QApplication.instance()
    except Exception:
        app = None

    if app is None:
        try:
            from PyQt5.QtWidgets import QApplication  # type: ignore

            app = QApplication.instance()
        except Exception:
            app = None

    if app is None:
        try:
            from PyQt6.QtWidgets import QApplication  # type: ignore

            app = QApplication.instance()
        except Exception:
            app = None

    if app is None:
        return

    try:
        app.aboutToQuit.connect(_on_app_about_to_quit)
        _QUIT_HOOK_INSTALLED = True
    except Exception:
        pass


def _start_bridge(
    context,
    host: str,
    port: int,
    startup_note: Optional[str] = None,
    show_error_on_fail: bool = True,
) -> bool:
    install_live_status_widget(context)

    with _LOCK:
        if _running():
            _show_info(
                context,
                "SALOME MCP Bridge",
                f"Bridge is already running on {_STATE.host}:{_STATE.port}.",
            )
            return True
        if _starting():
            _show_info(
                context,
                "SALOME MCP Bridge",
                f"Bridge startup is already in progress on {_STATE.host}:{_STATE.port}.",
            )
            return True

        # Reload bridge module so Start picks up latest bridge code without SALOME restart.
        try:
            importlib.reload(salome_bridge)
        except Exception:
            pass

        server = salome_bridge.SalomeBridgeServer(host=host, port=port)
        thread = threading.Thread(
            target=server.start,
            daemon=True,
            name="SalomeMCPBridge",
        )
        thread.start()

        _STATE.host = host
        _STATE.port = port
        _STATE.server = server
        _STATE.thread = thread

        # Helpful defaults for sessions launched from the same environment.
        os.environ["SALOME_PORT"] = str(port)
        if host in ("0.0.0.0", "::"):
            os.environ["SALOME_HOST"] = "localhost"
        else:
            os.environ["SALOME_HOST"] = host

    def check_start(attempts_left: int) -> None:
        if _STATE.server is not server:
            return
        if server.last_error:
            with _LOCK:
                if _STATE.server is server:
                    _STATE.server = None
                    _STATE.thread = None
            if show_error_on_fail:
                _show_error(
                    context,
                    "SALOME MCP Bridge",
                    f"Failed to open {host}:{port}.\nReason: {server.last_error}",
                )
            _refresh_status_widget()
            return
        if server.running and server.server_socket is not None:
            _refresh_status_widget()
            return
        if attempts_left <= 0:
            server.stop()
            with _LOCK:
                if _STATE.server is server:
                    _STATE.server = None
                    _STATE.thread = None
            if show_error_on_fail:
                _show_error(
                    context,
                    "SALOME MCP Bridge",
                    f"Failed to open {host}:{port}.\nReason: Bridge start timed out",
                )
            _refresh_status_widget()
            return
        QTimer.singleShot(200, lambda: check_start(attempts_left - 1))

    check_start(max(5, int(START_TIMEOUT * 5)))

    message = f"Starting bridge on {host}:{port}.\n"
    if startup_note:
        message += f"{startup_note}\n"
    message += "This may take a few seconds."

    _show_info(context, "SALOME MCP Bridge", message)
    _refresh_status_widget()
    return True


def start_bridge_dialog(context) -> None:
    """Prompt for host/port and start MCP bridge."""
    install_live_status_widget(context)
    desktop = _desktop(context)

    host, ok = QInputDialog.getText(
        desktop,
        "SALOME MCP Bridge",
        "Host/interface to bind:",
        text=_STATE.host,
    )
    if not ok:
        return

    host = host.strip() or "localhost"

    port, ok = QInputDialog.getInt(
        desktop,
        "SALOME MCP Bridge",
        "Port to open:",
        value=_STATE.port,
        min=1024,
        max=65535,
    )
    if not ok:
        return

    _start_bridge(context, host, port)


def start_bridge_default(context) -> None:
    """Start bridge with default host/port without dialog."""
    install_live_status_widget(context)
    host = _STATE.host
    default_port = DEFAULT_PORT

    default_unavailable = not _can_bind(host, default_port)
    if not default_unavailable:
        _start_bridge(context, host, default_port, show_error_on_fail=True)
        return

    desktop = _desktop(context)
    port, ok = QInputDialog.getInt(
        desktop,
        "SALOME MCP Bridge",
        f"Default port {default_port} is unavailable.\nEnter a port:",
        value=max(1024, default_port + 1),
        min=1024,
        max=65535,
    )
    if not ok:
        return

    _start_bridge(
        context,
        host,
        port,
        startup_note=f"Default port {default_port} unavailable; using {port}.",
    )


def stop_bridge(context) -> None:
    """Stop MCP bridge and close the listening port."""
    _stop_bridge_internal(show_info=True, context=context)


def bridge_status(context) -> None:
    """Display current bridge status and connection hint."""
    install_live_status_widget(context)
    _refresh_status_widget()
    _show_info(context, "SALOME MCP Bridge", _status_text())
