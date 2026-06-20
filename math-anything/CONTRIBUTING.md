# Contributing to Bourbaki

Thanks for your interest in contributing! This guide covers everything you need to get started.

## How to Contribute

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   # or: fix/issue-description, docs/topic
   ```
3. **Make your changes** and commit with conventional commit messages
4. **Open a Pull Request** against `main`

## Development Setup

```bash
# Clone
git clone https://github.com/toki0413/math-anything.git
cd math-anything/math-anything

# Python environment
pip install -e ".[dev,mcp]"

# Rust extension (optional — provides 12 accelerated functions)
pip install maturin
maturin develop --release
```

### Prerequisites

- Python 3.10+
- Rust toolchain (optional, for `math_anything_rs` acceleration)
- Git

## Code Style

We use **ruff** for linting and **mypy** for type checking.

```bash
# Lint
ruff check math_anything/

# Format
ruff format math_anything/

# Type check
mypy math_anything/ --ignore-missing-imports
```

### Style Rules

- Line length: 120 characters
- Target Python: 3.10+
- Use type hints where appropriate
- Follow the existing code patterns in the codebase

## Testing

```bash
# Run all unit tests with coverage
pytest tests/unit/ -v --tb=short --cov=math_anything --cov-report=term-missing

# Run integration tests
pytest tests/integration/ -v --tb=short

# Run regression tests
pytest tests/regression/ -v --tb=short

# Coverage threshold (currently 20%)
pytest tests/unit/ --cov=math_anything --cov-fail-under=20 -q
```

### Coverage Requirements

All new code should include tests. We enforce a minimum coverage threshold (currently 20%). PRs that significantly reduce coverage will be flagged.

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Usage |
|------|-------|
| `feat` | New feature or engine integration |
| `fix` | Bug fix |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or updating tests |
| `docs` | Documentation changes |
| `perf` | Performance improvement |
| `chore` | Build, CI, or tooling changes |

### Examples

```
feat(vasp): add ENCUT constraint validation
fix(lammps): handle missing timestep in NVT ensemble
refactor(core): extract shared harness logic into base class
test(ansys): add APDL parser regression tests
docs(mcp): update Claude Desktop configuration example
```

## PR Checklist

Before submitting a pull request, verify:

- [ ] All tests pass (`pytest tests/unit/ tests/integration/`)
- [ ] Lint passes (`ruff check math_anything/`)
- [ ] Type check passes (`mypy math_anything/ --ignore-missing-imports`)
- [ ] Commit messages follow Conventional Commits
- [ ] New code has tests
- [ ] Documentation updated if API changed
- [ ] No secrets, API keys, or credentials in the diff

## Reporting Issues

- **Bugs**: Use the Bug Report template. Include Python version, OS, affected engine, and full error logs.
- **Feature Requests**: Use the Feature Request template. Describe the use case and proposed solution.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
