# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Conservation Matrix Field with 18 equation builders (Euler, NS, Schrödinger, Maxwell, Elasticity, Heat, Advection-Diffusion, MHD, Kohn-Sham, Boltzmann, Shallow Water, Wave, Dirac, Klein-Gordon, Einstein, NLS, Vlasov, Hartree-Fock)
- Rust acceleration layer (12 PyO3 functions with Rayon parallelism)
- BaseEngineExtractor ABC with 7 deep engine implementations (VASP, LAMMPS, Abaqus, Ansys, Gaussian, OpenFOAM, Quantum ESPRESSO)
- 5-layer verification pipeline (Symbolic, Type System, Logic, LLM Semantic, Lean4 Formal)
- Safe expression evaluator (AST-based safe_eval replacing unsafe eval())
- Unified exception hierarchy (27 exception classes)
- Structured logging (get_logger() replacing print())
- ErrorBoundary component for React frontend
- OpenAPI 3.0 schema and /health + /metrics endpoints

### Changed
- Split large files: algebras.py → star + vonneumann, approximations.py → dft + md + cfd + quantum + surrogate, properties.py → enums + registry + _core, geometry.py → manifold + riemannian + continuum, category_theory.py → basic + advanced, analysis.py → stability + convergence
- Engine extractors refactored to inherit from BaseEngineExtractor
- Eliminated all bare `except:` clauses (22 → 0)
- Replaced core eval() with safe_eval (4 → 0 unsafe in core)

### Deprecated
- HarnessRegistry, MathAnythingHarness, ExtractionSession (will be removed in v4.0)
- PSRN legacy API (will be removed in v4.0)

### Security
- Fixed code injection vulnerability in codegen/constraint_inference.py (bare eval → safe_eval)
- Added security annotations to PSRN eval() calls with __builtins__ restriction
