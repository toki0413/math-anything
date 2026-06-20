# Bourbaki for VS Code

A Visual Studio Code extension that brings [Bourbaki](https://github.com/math-anything/math-anything)'s mathematical-structure analysis directly into the editor.

Bourbaki reads scientific-computing input files (VASP, Quantum ESPRESSO, LAMMPS, Abaqus, OpenFOAM, GROMACS, ...) and translates them into a common mathematical-language layer: domains, conservation laws, morphisms, type-theoretic verification, and numerical solvers.

## Features

- **Language support** with syntax highlighting for:
  - VASP `INCAR` / `POSCAR`
  - Quantum ESPRESSO `.in`
  - LAMMPS `.lmp`
  - Abaqus `.inp`
- **Analyze File / Analyze Workspace** — discover the mathematical domain and conservation laws of an input file.
- **Verify Equation** — run Bourbaki's verification pipeline on a selected equation.
- **Solve Numerical** — invoke built-in numerical solvers (symplectic, variational, eigenvalue, SCF, conservation-law).
- **Diagnostics** — missing recommended parameters are surfaced as editor warnings.
- **Side-panel views** — Domains and Analysis explorers.

## Requirements

1. Python 3.10+ with `bourbaki` (or the in-repo `math_anything` package) installed.
2. VS Code 1.85 or newer.

## Installation

### From source

```bash
cd bourbaki-vscode
npm install
npm run compile
```

Then press `F5` in VS Code to launch a new Extension Development Host window.

### Settings

| Setting | Default | Description |
|--------|---------|-------------|
| `bourbaki.pythonPath` | `python` | Python interpreter that has Bourbaki installed. |
| `bourbaki.lsp.enabled` | `true` | Enable the Bourbaki language server. |
| `bourbaki.lsp.trace` | `false` | Trace LSP traffic to the output channel. |

## Commands

Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and search for **Bourbaki**:

- `Bourbaki: Analyze Mathematical Structure`
- `Bourbaki: Analyze Workspace`
- `Bourbaki: Verify Selected Equation`
- `Bourbaki: Solve Numerical Problem`
- `Bourbaki: Open Analysis Dashboard`

## Architecture

```
VS Code
   │
   ├─ TypeScript client (commands, tree views, webview)
   │
   ├─ vscode-languageclient ── stdio ──┐
   │                                    │
   └─ TextMate grammars for input files │
                                        ▼
                              lsp_server.py (Python)
                                        │
                              math_anything (Bourbaki core)
```

The extension starts `lsp_server.py` as a stdio-based JSON-RPC language server. It exposes Bourbaki capabilities through custom LSP methods (`bourbaki/analyzeFile`, `bourbaki/verifyEquation`, etc.).

## Packaging

```bash
npm install -g @vscode/vsce
npm run package
```

This produces `bourbaki-vscode-<version>.vsix`, which can be installed manually from the Extensions view.

## License

MIT — same as the Bourbaki core library.
