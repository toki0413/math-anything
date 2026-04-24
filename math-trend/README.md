# Math Trend

Dynamical systems analysis for scientific research trends. Treat research fields as dynamical systems: identify attractors (stable research directions), detect bifurcations (paradigm shifts), and predict trajectories.

## Why this exists

Traditional bibliometrics counts papers and citations. Math Trend asks: what is the *dynamical structure* of a research field?

- **Attractors**: stable research directions that pull in resources and attention
- **Bifurcations**: moments when a field splits into new directions
- **Phase space**: the landscape of all possible research configurations
- **Lyapunov exponents**: how predictable or chaotic a field's evolution is

This matters because research strategy should depend on whether you're in a stable attractor (safe, crowded) or near a bifurcation point (risky, high reward).

## What it does

- **Field phase space analysis**: dimensionality, entropy, predictability
- **Attractor identification**: stable research directions with metrics
- **Bifurcation detection**: paradigm shifts and turning points
- **Cross-domain transfer detection**: track knowledge migration between fields
- **Journal dynamics ranking**: tier journals by dynamical importance, not just impact factor
- **Researcher position analysis**: where are you in the phase space?

## Quick start

### Installation

```bash
git clone https://github.com/toki0413/math-trend.git
cd math-trend
pip install -e .
```

### CLI Usage

```bash
# Analyze a research field
math-trend analyze "cement energy storage" --years 2019-2026

# Detect cross-domain knowledge transfer
math-trend transfer "battery technology" "cement storage" --output report.md

# Rank journals in a field
math-trend journals "materials science" --top 20
```

### Python API

#### Field Analysis

```python
from math_trend import FieldAnalyzer

analyzer = FieldAnalyzer()
result = analyzer.analyze_field("cement energy storage", year_from=2019, year_to=2026)

print(f"Phase space dimension: {result.dimension}")
print(f"Entropy (diversity): {result.entropy}")
print(f"Max Lyapunov exponent: {result.lyapunov}")
print(f"Predictability: {result.predictability}")
```

#### Attractor Identification

```python
from math_trend import AttractorDetector

detector = AttractorDetector()
attractors = detector.identify_attractors(papers, min_papers=20)

for att in attractors:
    print(f"{att.name}: {att.papers} papers, stability={att.stability}")
```

#### Cross-Domain Transfer Detection

```python
from math_trend import CrossDomainTransferDetector

detector = CrossDomainTransferDetector()
transfers = detector.detect_concept_transfer(
    source_papers=battery_papers,
    target_papers=cement_papers,
    source_domain="battery technology",
    target_domain="cement storage"
)

for t in transfers[:5]:
    print(f"{t.concept}: {t.transfer_type}, delay={t.first_occurrence-2015}y")
```

## Core concepts

### Attractors

Stable research directions. Characterized by:
- **Paper count**: how much activity
- **Stability**: how long the direction persists
- **Maturity**: TRL level
- **Prospect**: commercial potential

### Bifurcations

Paradigm shifts. Detected by:
- Sudden keyword emergence
- Citation network restructuring
- Methodology changes

### Cross-Domain Transfer

Knowledge migration between fields:
- **Method transfer**: experimental techniques
- **Theory transfer**: conceptual frameworks
- **Tool transfer**: software, instruments
- **Concept transfer**: ideas and metaphors

## Modules

| Module | Purpose |
|--------|---------|
| `core` | Field analysis, phase space metrics |
| `dynamics` | Attractor detection, bifurcation analysis |
| `data` | Data collection from OpenAlex, APIs |
| `report` | Report generation |
| `visualize` | Plotting and visualization |

## Examples

See `examples/` directory:
- `demo_cement_storage.py` - Complete analysis of cement energy storage field
- `demo_cross_domain_transfer.py` - Battery -> cement knowledge transfer

## License

MIT
