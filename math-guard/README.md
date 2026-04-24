# Math Guard

Math-driven academic integrity detection engine. Uses mathematical fingerprints and structural analysis to detect anomalies in academic documents.

## Why this exists

Traditional plagiarism detection compares text similarity. Math Guard asks: does the *mathematical structure* of this document make sense?

- **Equation consistency**: are the equations dimensionally consistent?
- **Logical flow**: do the theorems follow valid deduction chains?
- **Statistical anomalies**: are the p-values or distributions suspicious?
- **Citation topology**: does the citation network look natural?

This matters because AI-generated or manipulated papers may pass text-based checks but fail mathematical consistency tests.

## What it does

- **Mathematical fingerprinting**: extract structural signatures from equations
- **Dimensional analysis**: verify physical consistency of formulas
- **Theorem chain validation**: check logical deduction structure
- **Statistical anomaly detection**: identify suspicious data patterns
- **Citation network analysis**: detect artificial citation patterns
- **Cross-engine integrity**: validate consistency across multiple analysis engines

## Six analysis engines

| Engine | Analysis | Detects |
|--------|----------|---------|
| Information Theory | Entropy, compression | Unnatural text generation |
| Graph Theory | Citation networks | Artificial citation clusters |
| Computational Complexity | Algorithm claims | Implausible complexity assertions |
| Stochastic Analysis | Statistical patterns | P-hacking, fabricated data |
| Differential Geometry | Manifold consistency | Inconsistent geometric claims |
| Formal Logic | Proof structure | Invalid deduction chains |

## Quick start

### Installation

```bash
git clone https://github.com/toki0413/math-guard.git
cd math-guard
pip install -e .
```

### CLI Usage

```bash
# Scan a single paper
math-guard scan paper.pdf --output report.json

# Batch scan a directory
math-guard scan-dir ./papers/ --recursive

# Start monitoring service
math-guard serve --port 8000
```

### Python API

```python
from math_guard import IntegrityScanner

scanner = IntegrityScanner()
result = scanner.scan_document("paper.pdf")

print(f"Overall risk: {result.risk_score}")
print(f"Anomalies: {len(result.anomalies)}")

for anomaly in result.anomalies:
    print(f"  [{anomaly.severity}] {anomaly.type}: {anomaly.description}")
```

## Modules

| Module | Purpose |
|--------|---------|
| `core` | Document extraction, fingerprinting |
| `engines` | Six mathematical analysis engines |
| `ingestion` | PDF/TeX parsing, enrichment |
| `schemas` | Data models and integrity schemas |
| `service` | Scanner, watcher, report generation |
| `tiered` | Tiered analysis based on document complexity |

## License

MIT
