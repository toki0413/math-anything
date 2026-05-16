"""Build standalone distributable EXE for math-anything.

Produces two executables:
  1. math-anything-server.exe  - FastAPI HTTP server (for Tauri app)
  2. math-anything.exe         - CLI tool (extract, verify, lean4, etc.)

Usage:
    python build_exe.py              # Build both
    python build_exe.py --server     # Server only
    python build_exe.py --cli        # CLI only

Requires: pip install pyinstaller
"""

import os
import sys
import shutil
import subprocess

CORE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "math-anything", "core")
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")

SERVER_ENTRY = os.path.join(CORE_DIR, "math_anything", "server.py")
CLI_ENTRY = os.path.join(CORE_DIR, "math_anything", "cli.py")

EXCLUDES = [
    "torch", "torchvision", "torchaudio",
    "tensorflow", "keras",
    "gradio", "gradio_client",
    "transformers", "diffusers",
    "accelerate", "bitsandbytes",
    "wandb", "mlflow",
    "scipy",
    "pandas",
    "matplotlib",
    "PIL",
    "cv2",
    "sklearn",
    "numba",
    "llvmlite",
    "pytest",
    "jedi",
    "IPython",
    "notebook",
    "jupyter",
    "tornado",
    "bokeh",
    "plotly",
    "altair",
    "seaborn",
    "nltk",
    "spacy",
    "gensim",
    "faiss",
    "sentence_transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "einops",
    "triton",
    "xformers",
    "flash_attn",
    "deepspeed",
    "mpi4py",
    "horovod",
    "ray",
    "dask",
    "distributed",
    "pyarrow",
    "polars",
    "duckdb",
    "sqlalchemy",
    "alembic",
    "pymongo",
    "redis",
    "celery",
    "kombu",
    "amqp",
    "billiard",
    "vine",
    "flower",
    "gunicorn",
    "uvloop",
    "httptools",
    "websockets",
    "watchfiles",
    "python_dotenv",
    "rich",
    "pygments",
    "markdown_it",
    "mdurl",
    "linkify_it",
    "uc_micro",
    "jinja2",
    "markupsafe",
    "itsdangerous",
    "werkzeug",
    "flask",
    "django",
    "fastapi_cli",
    "shellingham",
    "typer",
    "tqdm",
    "filelock",
    "fsspec",
    "s3fs",
    "boto3",
    "botocore",
    "s3transfer",
    "jmespath",
    "dateutil",
    "pytz",
    "tzdata",
    "lean_dojo",
]

HIDDEN_IMPORTS = [
    "math_anything",
    "math_anything.api",
    "math_anything.cli",
    "math_anything.server",
    "math_anything.geometry",
    "math_anything.flywheel",
    "math_anything.formal_verifier",
    "math_anything.lean4_bridge",
    "math_anything.validation_toolkit",
    "math_anything.proposition",
    "math_anything.proof_verifier",
    "math_anything.agents",
    "math_anything.schemas",
    "math_anything.schemas.math_schema",
    "math_anything.schemas.precision",
    "math_anything.schemas.extensions",
    "math_anything.core",
    "math_anything.exceptions",
    "math_anything.visualization",
    "math_anything.simplifier",
    "math_anything.multivar",
    "math_anything.eml_v2",
    "math_anything.tiered",
    "math_anything.security",
    "math_anything.emergence",
    "math_anything.repl",
    "math_anything.repl.core",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "pydantic",
    "pydantic.v1",
    "starlette",
    "starlette.routing",
    "starlette.middleware",
    "starlette.middleware.cors",
    "sympy",
    "click",
    "prompt_toolkit",
    "numpy",
]


def build_exe(name, entry_point, extra_args=None):
    if not os.path.exists(entry_point):
        print(f"Error: {entry_point} not found")
        return False

    workpath = os.path.join(OUTPUT_DIR, "_build", name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", name,
        "--distpath", OUTPUT_DIR,
        "--workpath", workpath,
        "--specpath", workpath,
        "--clean",
        "--noconfirm",
    ]

    for mod in EXCLUDES:
        cmd.extend(["--exclude-module", mod])

    for mod in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", mod])

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(entry_point)

    env = os.environ.copy()
    env["PYTHONPATH"] = CORE_DIR

    print(f"\nBuilding {name}.exe ...")
    print(f"  Entry: {entry_point}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Hidden imports: {len(HIDDEN_IMPORTS)}")
    print(f"  Excluded modules: {len(EXCLUDES)}")
    print()

    result = subprocess.run(cmd, cwd=CORE_DIR, env=env)
    if result.returncode != 0:
        print(f"Build {name} failed!")
        return False

    exe_path = os.path.join(OUTPUT_DIR, f"{name}.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"  ✓ {name}.exe ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ✗ {name}.exe not found")
        return False


def main():
    build_server = True
    build_cli = True

    if "--server" in sys.argv:
        build_cli = False
    if "--cli" in sys.argv:
        build_server = False

    print("=" * 60)
    print("  MATH-ANYTHING EXE BUILDER")
    print("=" * 60)

    results = {}

    if build_server:
        results["server"] = build_exe("math-anything-server", SERVER_ENTRY)

    if build_cli:
        results["cli"] = build_exe("math-anything", CLI_ENTRY, [
            "--console",
        ])

    build_dir = os.path.join(OUTPUT_DIR, "_build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("  BUILD SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓ SUCCESS" if ok else "✗ FAILED"
        exe_path = os.path.join(OUTPUT_DIR, f"{name}.exe")
        if ok and os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"  {status}: {name}.exe ({size_mb:.1f} MB)")
        else:
            print(f"  {status}: {name}.exe")

    if all(results.values()):
        print("\n  All builds successful!")
        print(f"  Output directory: {OUTPUT_DIR}")
    else:
        print("\n  Some builds failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
