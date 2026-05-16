"""Unified build script for Math-Anything desktop app.

Usage:
    python build.py --target sidecar      # Build Python sidecar EXE
    python build.py --target standalone    # Build standalone CLI+Server EXE
    python build.py --target tauri         # Build complete Tauri desktop app
    python build.py --target all           # Build everything
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CORE = ROOT / "math-anything" / "core"
APP = ROOT / "math-anything-app"
VERSION_FILE = CORE / "math_anything" / "version.py"


def get_version() -> str:
    ns = {}
    with open(VERSION_FILE) as f:
        exec(f.read(), ns)
    return ns["__version__"]


def sync_version(version: str) -> None:
    pyproject = CORE / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8")
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if line.startswith("version ="):
                new_lines.append(f'version = "{version}"')
            else:
                new_lines.append(line)
        pyproject.write_text("\n".join(new_lines), encoding="utf-8")

    tauri_conf = APP / "src-tauri" / "tauri.conf.json"
    if tauri_conf.exists():
        with open(tauri_conf) as f:
            conf = json.load(f)
        conf["version"] = version
        with open(tauri_conf, "w") as f:
            json.dump(conf, f, indent=2, ensure_ascii=False)

    cargo_toml = APP / "src-tauri" / "Cargo.toml"
    if cargo_toml.exists():
        content = cargo_toml.read_text(encoding="utf-8")
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if line.strip().startswith("version") and 'package' not in line.lower():
                new_lines.append(f'version = "{version}"')
            else:
                new_lines.append(line)
        cargo_toml.write_text("\n".join(new_lines), encoding="utf-8")

    package_json = APP / "package.json"
    if package_json.exists():
        with open(package_json) as f:
            pkg = json.load(f)
        pkg["version"] = version
        with open(package_json, "w") as f:
            json.dump(pkg, f, indent=2, ensure_ascii=False)

    print(f"Version synced to {version} across all config files")


def build_sidecar() -> None:
    print("Building Python sidecar EXE...")
    sidecar_dir = APP / "src-tauri" / "binaries"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(CORE)

    result = subprocess.run(
        [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--name", "math-anything-server",
            "--distpath", str(sidecar_dir),
            "--workpath", str(ROOT / ".build" / "sidecar_work"),
            "--specpath", str(ROOT / ".build"),
            "--hidden-import", "math_anything",
            "--hidden-import", "math_anything.server",
            "--hidden-import", "math_anything.tool_system",
            "--hidden-import", "math_anything.tool_registry",
            "--hidden-import", "math_anything.agent_loop",
            "--hidden-import", "math_anything.tools",
            "--hidden-import", "math_anything.validation_toolkit",
            "--hidden-import", "uvicorn.logging",
            "--hidden-import", "uvicorn.loops",
            "--hidden-import", "uvicorn.loops.auto",
            "--hidden-import", "uvicorn.protocols",
            "--hidden-import", "uvicorn.protocols.http",
            "--hidden-import", "uvicorn.protocols.http.auto",
            "--hidden-import", "uvicorn.protocols.websockets",
            "--hidden-import", "uvicorn.protocols.websockets.auto",
            "--hidden-import", "uvicorn.lifespan",
            "--hidden-import", "uvicorn.lifespan.on",
            "-c",
            str(CORE / "math_anything" / "server.py"),
        ],
        env=env,
        cwd=str(CORE),
    )

    if result.returncode != 0:
        print("ERROR: Sidecar build failed!")
        sys.exit(1)

    ext = ".exe" if sys.platform == "win32" else ""
    src = sidecar_dir / f"math-anything-server{ext}"
    target_name = f"math-anything-server-x86_64-pc-windows-gnu{ext}" if sys.platform == "win32" else f"math-anything-server-{_rust_target()}"
    dst = sidecar_dir / target_name
    if src.exists() and src != dst:
        shutil.move(str(src), str(dst))

    print(f"Sidecar built: {dst}")


def build_standalone() -> None:
    print("Building standalone EXE...")
    dist_dir = ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(CORE)

    result = subprocess.run(
        [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--name", "math-anything",
            "--distpath", str(dist_dir),
            "--workpath", str(ROOT / ".build" / "standalone_work"),
            "--specpath", str(ROOT / ".build"),
            "--hidden-import", "math_anything",
            "--hidden-import", "math_anything.server",
            "--hidden-import", "math_anything.tool_system",
            "--hidden-import", "math_anything.tool_registry",
            "--hidden-import", "math_anything.agent_loop",
            "--hidden-import", "math_anything.tools",
            "--hidden-import", "math_anything.validation_toolkit",
            "--hidden-import", "math_anything.cli",
            "-c",
            str(CORE / "math_anything" / "__main__.py"),
        ],
        env=env,
        cwd=str(CORE),
    )

    if result.returncode != 0:
        print("ERROR: Standalone build failed!")
        sys.exit(1)

    print(f"Standalone built: {dist_dir}")


def build_tauri() -> None:
    print("Building Tauri desktop app...")
    build_sidecar()

    print("Installing frontend dependencies...")
    subprocess.run(["npm", "install"], cwd=str(APP), shell=True)

    print("Building Tauri app...")
    result = subprocess.run(
        ["npm", "run", "tauri", "build"],
        cwd=str(APP),
        shell=True,
    )

    if result.returncode != 0:
        print("ERROR: Tauri build failed!")
        sys.exit(1)

    print("Tauri app built successfully!")


def _rust_target() -> str:
    if sys.platform == "win32":
        return "x86_64-pc-windows-msvc"
    elif sys.platform == "darwin":
        return "aarch64-apple-darwin"
    return "x86_64-unknown-linux-gnu"


def main():
    parser = argparse.ArgumentParser(description="Build Math-Anything desktop app")
    parser.add_argument("--target", choices=["sidecar", "standalone", "tauri", "all"], default="tauri")
    parser.add_argument("--sync-version", action="store_true", help="Sync version across all config files")
    args = parser.parse_args()

    version = get_version()
    print(f"Math-Anything v{version}")

    if args.sync_version:
        sync_version(version)

    if args.target == "sidecar":
        build_sidecar()
    elif args.target == "standalone":
        build_standalone()
    elif args.target == "tauri":
        build_tauri()
    elif args.target == "all":
        build_sidecar()
        build_standalone()
        build_tauri()


if __name__ == "__main__":
    main()
