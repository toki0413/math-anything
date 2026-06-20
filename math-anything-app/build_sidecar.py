"""Build the Python sidecar EXE for the Tauri app.

Usage:
    python build_sidecar.py

Output:
    math-anything-server.exe in src-tauri/binaries/

Requires: pip install pyinstaller
"""

import os
import sys
import shutil
import subprocess

CORE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "math-anything", "core")
)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "src-tauri", "binaries")

ENTRY_POINT = os.path.join(CORE_DIR, "math_anything", "server.py")


def main():
    if not os.path.exists(ENTRY_POINT):
        print(f"Error: server.py not found at {ENTRY_POINT}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    excludes = [
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
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "math-anything-server",
        "--distpath", OUTPUT_DIR,
        "--workpath", os.path.join(OUTPUT_DIR, "_build"),
        "--specpath", os.path.join(OUTPUT_DIR, "_build"),
        "--clean",
        "--noconfirm",
    ]

    for mod in excludes:
        cmd.extend(["--exclude-module", mod])

    cmd.extend([
        "--hidden-import", "math_anything",
        "--hidden-import", "math_anything.geometry",
        "--hidden-import", "math_anything.flywheel",
        "--hidden-import", "math_anything.formal_verifier",
        "--hidden-import", "math_anything.proposition",
        "--hidden-import", "math_anything.proof_verifier",
        "--hidden-import", "math_anything.agents",
        "--hidden-import", "math_anything.schemas",
        "--hidden-import", "math_anything.api",
        "--hidden-import", "math_anything.core",
        "--hidden-import", "math_anything.exceptions",
        "--hidden-import", "math_anything.visualization",
        "--hidden-import", "math_anything.simplifier",
        "--hidden-import", "math_anything.multivar",
        "--hidden-import", "math_anything.eml_v2",
        "--hidden-import", "math_anything.tiered",
        "--hidden-import", "math_anything.security",
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
        "--hidden-import", "fastapi",
        "--hidden-import", "pydantic",
        "--hidden-import", "pydantic.v1",
        "--hidden-import", "starlette",
        "--hidden-import", "starlette.routing",
        "--hidden-import", "starlette.middleware",
        "--hidden-import", "starlette.middleware.cors",
        ENTRY_POINT,
    ])

    env = os.environ.copy()
    env["PYTHONPATH"] = CORE_DIR

    print("Building sidecar EXE (lightweight mode)...")
    print(f"  Entry: {ENTRY_POINT}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Excluding {len(excludes)} heavy modules")
    print()

    result = subprocess.run(cmd, cwd=CORE_DIR, env=env)
    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)

    exe_name = "math-anything-server.exe"
    exe_path = os.path.join(OUTPUT_DIR, exe_name)
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\nBuild successful!")
        print(f"  EXE: {exe_path}")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print(f"\nWarning: EXE not found at {exe_path}")

    build_dir = os.path.join(OUTPUT_DIR, "_build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
