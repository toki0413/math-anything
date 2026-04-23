# Math Anything API Reference

Complete API reference for Math Anything.

## Table of Contents

- [Core API](#core-api)
- [Exception Handling](#exception-handling)
- [Security Utilities](#security-utilities)
- [Pydantic Models](#pydantic-models)
- [Tiered Analysis](#tiered-analysis)

## Core API

### MathAnything Class

Main entry point for Math Anything functionality.

```python
from math_anything import MathAnything

ma = MathAnything()
```

#### Methods

##### `extract(engine, params)`

Extract mathematical structure from parameters.

```python
result = ma.extract("vasp", {"ENCUT": 520, "SIGMA": 0.05})
```

**Parameters:**
- `engine` (str): Computational engine name
- `params` (dict): Engine parameters

**Returns:** `ExtractionResult`

**Raises:**
- `UnsupportedEngineError`: If engine is not supported
- `ValidationError`: If parameters are invalid

##### `extract_file(engine, filepath)`

Extract from input file(s).

```python
result = ma.extract_file("vasp", "INCAR")
result = ma.extract_file("lammps", "in.file")
```

**Parameters:**
- `engine` (str): Engine name
- `filepath` (str | dict): File path or dict of file paths

**Returns:** `ExtractionResult`

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `FileAccessError`: If access is denied
- `ParseError`: If parsing fails

### Convenience Functions

#### `extract(engine, params)`

One-shot extraction without creating instance.

```python
from math_anything import extract

result = extract("lammps", {"timestep": 0.5})
```

## Exception Handling

All exceptions inherit from `MathAnythingError`.

### Exception Hierarchy

```
MathAnythingError
├── UnsupportedEngineError
├── FileNotFoundError
├── FileAccessError
├── ParseError
├── ValidationError
├── TierAnalysisError
├── SecurityError
└── ConfigurationError
```

### Common Exceptions

#### `UnsupportedEngineError`

```python
from math_anything.exceptions import UnsupportedEngineError

try:
    result = extract("unknown_engine", {})
except UnsupportedEngineError as e:
    print(f"Engine: {e.engine}")
    print(f"Available: {e.available_engines}")
```

#### `FileAccessError`

Raised when file access is denied due to security restrictions.

```python
from math_anything.exceptions import FileAccessError

try:
    result = extract_file("vasp", "../../../etc/passwd")
except FileAccessError as e:
    print(f"Path: {e.filepath}")
    print(f"Reason: {e.reason}")
```

#### `ParseError`

Raised when file parsing fails.

```python
from math_anything.exceptions import ParseError

try:
    result = extract_file("lammps", "corrupted.lmp")
except ParseError as e:
    print(f"File: {e.filepath}")
    print(f"Parser: {e.parser}")
    print(f"Line: {e.line_number}")
    print(f"Content: {e.line_content}")
```

### Error Response Format

All exceptions can be serialized to JSON:

```python
from math_anything.exceptions import MathAnythingError

try:
    # ... some operation
    pass
except MathAnythingError as e:
    error_dict = e.to_dict()
    # {
    #     "error_code": "VALIDATION_ERROR",
    #     "message": "Validation failed...",
    #     "details": {...},
    #     "type": "ValidationError"
    # }
```

## Security Utilities

### Path Security

#### `PathSecurityValidator`

Validates file paths for security.

```python
from math_anything.security import PathSecurityValidator

validator = PathSecurityValidator(
    allowed_base_dirs=["/home/user/simulations"],
    allow_absolute_paths=True,
    max_path_length=4096,
)

# Validate path
try:
    safe_path = validator.validate("simulation.lmp")
except SecurityError as e:
    print(f"Security violation: {e.violation_type}")
```

**Features:**
- Path traversal detection (`../`, `..\`)
- URL-encoded traversal detection (`%2e%2e`)
- Null byte injection detection
- Dangerous extension blocking (`.exe`, `.sh`, `.py`)
- Allowed directory restriction

#### `validate_filepath()`

Convenience function for path validation.

```python
from math_anything.security import validate_filepath

safe_path = validate_filepath("test.lmp")
```

### File Size Validation

#### `FileSizeValidator`

```python
from math_anything.security import FileSizeValidator

validator = FileSizeValidator(
    max_file_size=100 * 1024 * 1024,      # 100MB
    max_total_size=500 * 1024 * 1024,     # 500MB
)

# Single file
size = validator.validate_file("large.lmp")

# Multiple files
total = validator.validate_total_size(["file1.lmp", "file2.lmp"])
```

## Pydantic Models

All models provide automatic validation and serialization.

### Request Models

#### `ExtractionRequest`

```python
from math_anything.models import ExtractionRequest, EngineType

request = ExtractionRequest(
    engine=EngineType.VASP,
    params={"ENCUT": 520, "SIGMA": 0.05},
    validate=True,
)

# Serialization
json_str = request.json()
dict_data = request.dict()
```

#### `FileExtractionRequest`

```python
from math_anything.models import FileExtractionRequest

# Single file
request = FileExtractionRequest(
    engine="vasp",
    filepath="INCAR",
)

# Multiple files
request = FileExtractionRequest(
    engine="vasp",
    filepath={
        "INCAR": "path/to/INCAR",
        "POSCAR": "path/to/POSCAR",
    },
)
```

#### `TieredAnalysisRequest`

```python
from math_anything.models import TieredAnalysisRequest, AnalysisTierEnum

request = TieredAnalysisRequest(
    filepath="simulation.lmp",
    tier=AnalysisTierEnum.PROFESSIONAL,
    auto_tier=False,
    min_tier=2,
    max_tier=4,
)
```

### Response Models

#### `ExtractionResultModel`

```python
from math_anything.models import (
    ExtractionResultModel,
    MathematicalStructure,
    Constraint,
)

result = ExtractionResultModel(
    engine="vasp",
    success=True,
    mathematical_structure=MathematicalStructure(
        problem_type="nonlinear_eigenvalue",
        canonical_form="H[n]ψ = εψ",
    ),
    constraints=[
        Constraint(expression="ENCUT > 0", satisfied=True),
    ],
)
```

#### `ComplexityScore`

```python
from math_anything.models import ComplexityScore

score = ComplexityScore(
    total=45.0,
    system_size=20.0,
    time_scale=15.0,
    constraints=8.0,
    data_availability=2.0,
)
```

Validation:
- `total`: 0-100
- `system_size`: 0-50
- `time_scale`: 0-30
- `constraints`: 0-20
- `data_availability`: 0-10

## Tiered Analysis

### TieredAnalyzer

```python
from math_anything import TieredAnalyzer, AnalysisTier

analyzer = TieredAnalyzer()

# Auto-detect tier
result = analyzer.analyze("simulation.lmp")

# Specify tier
result = analyzer.analyze("simulation.lmp", tier=AnalysisTier.ADVANCED)

# Get recommendation only
rec = analyzer.get_recommendation("simulation.lmp")
print(f"Recommended: {rec.recommended_tier}")
print(f"Complexity: {rec.complexity_score.total}")
```

### Analysis Tiers

| Tier | Name | Description |
|------|------|-------------|
| 1 | BASIC | Quick screening |
| 2 | ENHANCED | Detailed parameters |
| 3 | PROFESSIONAL | Topology analysis |
| 4 | ADVANCED | Geometric methods |
| 5 | COMPLETE | Five-layer framework |

### Convenience Function

```python
from math_anything import tiered_analyze

result = tiered_analyze("simulation.lmp", tier=3)
```

## Type Annotations

All public APIs have complete type annotations:

```python
from typing import Dict, Any
from math_anything import extract

def process_simulation(params: Dict[str, Any]) -> ExtractionResult:
    result = extract("vasp", params)
    return result
```

## Error Handling Best Practices

### Try-Except Blocks

```python
from math_anything import extract_file
from math_anything.exceptions import (
    MathAnythingError,
    FileNotFoundError,
    ValidationError,
)

def safe_extract(engine: str, filepath: str):
    try:
        return extract_file(engine, filepath)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filepath}")
        return None
    except ValidationError as e:
        logger.error(f"Invalid parameter {e.parameter}: {e.constraint}")
        raise
    except MathAnythingError as e:
        logger.error(f"Error {e.error_code}: {e.message}")
        raise
```

### Using Error Details

```python
from math_anything.exceptions import ParseError

try:
    result = extract_file("lammps", "input.lmp")
except ParseError as e:
    if e.line_number:
        print(f"Error at line {e.line_number}: {e.line_content}")
    if e.original_error:
        print(f"Caused by: {e.original_error}")
```

## Security Best Practices

### Validate User Input

```python
from math_anything.security import validate_filepath

def user_upload(filename: str):
    # Validate path before processing
    safe_path = validate_filepath(filename)
    return extract_file("lammps", str(safe_path))
```

### Configure Allowed Directories

```python
from math_anything.security import PathSecurityValidator

# Production configuration
validator = PathSecurityValidator(
    allowed_base_dirs=["/data/simulations"],
    allow_absolute_paths=False,
)
```

### Check File Sizes

```python
from math_anything.security import validate_file_size

def process_large_file(filepath: str):
    # Will raise FileAccessError if too large
    size = validate_file_size(filepath)
    if size > 10 * 1024 * 1024:  # 10MB
        logger.warning(f"Large file: {size} bytes")
```
