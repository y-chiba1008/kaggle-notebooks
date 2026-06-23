"""実行環境を判定して情報を出力する."""

from __future__ import annotations

import os
import platform
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EnvironmentInfo:
    runtime: str
    runtime_detail: str
    is_kaggle: bool
    kaggle_run_type: str | None
    is_colab: bool
    is_jupyter: bool
    jupyter_frontend: str | None
    python_version: str
    python_executable: str
    platform: str
    hostname: str
    cwd: str
    extra: dict[str, Any]


def _get_ipython():
    try:
        from IPython import get_ipython

        return get_ipython()
    except ImportError:
        return None


def _get_connection_info() -> dict[str, Any]:
    try:
        import json
        from ipykernel import get_connection_file

        with open(get_connection_file()) as f:
            return json.load(f)
    except Exception:
        return {}


def _is_vscode_jupyter_kernel() -> bool:
    return "jvsc" in _get_connection_info().get("kernel_name", "")


def _detect_ide_from_process_tree() -> str | None:
    try:
        import psutil

        proc = psutil.Process()
        for _ in range(20):
            cmdline = " ".join(proc.cmdline()).lower()
            if "cursor-server" in cmdline:
                return "Cursor"
            if "vscode-server" in cmdline:
                return "VS Code"
            try:
                if "cursor-server" in proc.cwd().lower():
                    return "Cursor"
            except Exception:
                pass
            parent = proc.parent()
            if parent is None or parent.pid == proc.pid:
                break
            proc = parent
    except Exception:
        pass

    try:
        pid = os.getpid()
        for _ in range(20):
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).lower()
            if "cursor-server" in cmdline:
                return "Cursor"
            if "vscode-server" in cmdline:
                return "VS Code"
            ppid = int(Path(f"/proc/{pid}/stat").read_text().split()[3])
            if ppid <= 1 or ppid == pid:
                break
            pid = ppid
    except OSError:
        pass

    return None


def _is_cursor() -> bool:
    if any(
        os.environ.get(key)
        for key in (
            "CURSOR_TRACE_ID",
            "CURSOR_SESSION_ID",
            "CURSOR_CHANNEL",
            "CURSOR_AGENT",
            "CURSOR_LAYOUT",
        )
    ):
        return True

    for env_key in ("VSCODE_CWD", "VSCODE_L10N_BUNDLE_LOCATION", "PWD"):
        if "cursor" in os.environ.get(env_key, "").lower():
            return True

    if ".cursor-server" in os.environ.get("PATH", "").lower():
        return True

    return _detect_ide_from_process_tree() == "Cursor"


def _is_vscode_family() -> bool:
    if _is_vscode_jupyter_kernel():
        return True
    if any(os.environ.get(key) for key in ("VSCODE_PID", "VSCODE_CWD", "VSCODE_IPC_HOOK_CLI")):
        return True
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return True
    return _detect_ide_from_process_tree() in ("Cursor", "VS Code")


def _detect_jupyter_frontend(shell) -> str | None:
    if shell is None:
        return None

    module = shell.__class__.__module__
    if "google.colab" in module:
        return "Google Colab"

    shell_name = shell.__class__.__name__
    if shell_name != "ZMQInteractiveShell":
        return shell_name

    if os.environ.get("COLAB_RELEASE_TAG"):
        return "Google Colab"
    if _is_vscode_family():
        return "Cursor" if _is_cursor() else "VS Code"
    if os.environ.get("JPY_PARENT_PID"):
        return "Jupyter (classic / lab)"
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return "Databricks"
    return "Jupyter (unknown frontend)"


def _get_kernel_display_name(shell) -> str | None:
    if shell is None or not hasattr(shell, "kernel"):
        return None

    if display_name := getattr(shell.kernel, "display_name", None):
        return display_name

    try:
        from jupyter_client.kernelspec import KernelSpecManager

        ksm = KernelSpecManager()
        executable = Path(sys.executable).resolve()
        for name in ksm.find_kernel_specs():
            spec = ksm.get_kernel_spec(name)
            if spec.argv and Path(spec.argv[0]).resolve() == executable:
                return spec.display_name
    except Exception:
        pass

    return None


def detect_environment() -> EnvironmentInfo:
    kaggle_run_type = os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    is_kaggle = kaggle_run_type is not None or Path("/kaggle/input").exists()

    is_colab = False
    try:
        import google.colab  # noqa: F401

        is_colab = True
    except ImportError:
        is_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))

    shell = _get_ipython()
    is_jupyter = shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"
    frontend = _detect_jupyter_frontend(shell)

    if is_kaggle:
        runtime = "Kaggle Notebook"
        runtime_detail = f"run_type={kaggle_run_type or 'unknown'}"
    elif is_colab:
        runtime = "Google Colab"
        runtime_detail = os.environ.get("COLAB_RELEASE_TAG", "")
    elif is_jupyter:
        if frontend in ("Cursor", "VS Code"):
            runtime = frontend
            runtime_detail = "Notebook (ipykernel)"
        else:
            runtime = "Jupyter / IPython"
            runtime_detail = frontend or shell.__class__.__name__
    elif shell is not None:
        runtime = "IPython (terminal)"
        runtime_detail = shell.__class__.__name__
    else:
        runtime = "Local Python"
        runtime_detail = "script or REPL (no IPython)"

    extra: dict[str, Any] = {}

    if is_kaggle:
        extra["kaggle_working_exists"] = Path("/kaggle/working").exists()
        extra["kaggle_input_mounted"] = Path("/kaggle/input").exists()
        for key in ("KAGGLE_URL", "KAGGLE_USER", "KAGGLE_KERNEL_INTEGRATIONS"):
            if value := os.environ.get(key):
                extra[key.lower()] = value

    if wsl_distro := os.environ.get("WSL_DISTRO_NAME"):
        extra["wsl_distro"] = wsl_distro

    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        extra["virtual_env"] = os.environ.get("VIRTUAL_ENV", sys.prefix)

    if is_jupyter:
        try:
            from ipykernel import get_connection_file

            extra["kernel_connection_file"] = get_connection_file()
        except Exception:
            pass

        kernel_display = _get_kernel_display_name(shell)
        if kernel_display:
            extra["kernel_display_name"] = kernel_display

    project_root = next(
        (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()),
        None,
    )
    if project_root:
        extra["project_root"] = str(project_root)

    return EnvironmentInfo(
        runtime=runtime,
        runtime_detail=runtime_detail,
        is_kaggle=is_kaggle,
        kaggle_run_type=kaggle_run_type,
        is_colab=is_colab,
        is_jupyter=is_jupyter,
        jupyter_frontend=frontend,
        python_version=sys.version.split()[0],
        python_executable=sys.executable,
        platform=platform.platform(),
        hostname=socket.gethostname(),
        cwd=str(Path.cwd()),
        extra=extra,
    )


def print_environment_report() -> EnvironmentInfo:
    info = detect_environment()

    print("=" * 60)
    print("実行環境レポート")
    print("=" * 60)
    print(f"  ランタイム       : {info.runtime}")
    print(f"  詳細             : {info.runtime_detail}")
    print(f"  Kaggle           : {'Yes' if info.is_kaggle else 'No'}")
    if info.kaggle_run_type:
        print(f"  Kaggle run type  : {info.kaggle_run_type}")
    print(f"  Google Colab     : {'Yes' if info.is_colab else 'No'}")
    print(f"  Jupyter/IPython  : {'Yes' if info.is_jupyter else 'No'}")
    if info.jupyter_frontend:
        print(f"  Jupyter frontend : {info.jupyter_frontend}")
    print("-" * 60)
    print(f"  Python           : {info.python_version}")
    print(f"  実行ファイル     : {info.python_executable}")
    print(f"  OS / Platform    : {info.platform}")
    print(f"  ホスト名         : {info.hostname}")
    print(f"  作業ディレクトリ : {info.cwd}")
    if info.extra:
        print("-" * 60)
        print("  追加情報:")
        for key, value in info.extra.items():
            print(f"    {key}: {value}")
    print("=" * 60)

    return info

if __name__ == "__main__":
    info = print_environment_report()