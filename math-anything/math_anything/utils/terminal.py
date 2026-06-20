"""Terminal output utilities with cross-platform encoding safety."""

import sys
from typing import Optional

# Common mathematical Unicode → ASCII fallback map
_UNICODE_FALLBACK = {
    "\u00d7": "x",  # × multiplication
    "\u0393": "Gamma",  # Γ
    "\u0394": "Delta",  # Δ
    "\u03a3": "Sum",  # Σ
    "\u03a6": "Phi",  # Φ
    "\u03a8": "Psi",  # Ψ
    "\u03a9": "Omega",  # Ω
    "\u03b1": "alpha",  # α
    "\u03b2": "beta",  # β
    "\u03b3": "gamma",  # γ
    "\u03b4": "delta",  # δ
    "\u03b5": "epsilon",  # ε
    "\u03b6": "zeta",  # ζ
    "\u03b7": "eta",  # η
    "\u03b8": "theta",  # θ
    "\u03bb": "lambda",  # λ
    "\u03bc": "mu",  # μ
    "\u03bd": "nu",  # ν
    "\u03c0": "pi",  # π
    "\u03c1": "rho",  # ρ
    "\u03c3": "sigma",  # σ
    "\u03c4": "tau",  # τ
    "\u03c6": "phi",  # φ
    "\u03c8": "psi",  # ψ
    "\u03c9": "omega",  # ω
    "\u2014": "-",  # — em dash
    "\u2022": "*",  # • bullet
    "\u207b": "-",  # ⁻ superscript minus
    "\u00b2": "^2",  # ²
    "\u00b3": "^3",  # ³
    "\u2071": "^1",  # ¹
    "\u2076": "^6",  # ⁶
    "\u2077": "^7",  # ⁷
    "\u2078": "^8",  # ⁸
    "\u2079": "^9",  # ⁹
    "\u2207": "nabla",  # ∇
    "\u2212": "-",  # − minus
    "\u221a": "sqrt",  # √
    "\u222b": "integral",  # ∫
    "\u2264": "<=",  # ≤
    "\u2265": ">=",  # ≥
    "\u22c5": ".",  # ⋅ dot
    "\u2191": "up",  # ↑
    "\u2192": "->",  # →
    "\u2193": "down",  # ↓
    "\u21d2": "=>",  # ⇒
    "\u210f": "hbar",  # ℏ
    "\u212b": "A",  # Å (Angstrom)
    "\u2500": "-",  # ─ box drawing
    "\u2501": "-",
    "\u2502": "|",
    "\u2503": "|",
    "\u2550": "=",
    "\u2551": "|",
    "\u2591": " ",  # ░ light shade
    "\u2592": " ",  # ▒ medium shade
    "\u2593": " ",  # ▓ dark shade
}


def ascii_fallback(text: str) -> str:
    """Replace common Unicode math symbols with ASCII equivalents."""
    for ch, repl in _UNICODE_FALLBACK.items():
        text = text.replace(ch, repl)
    return text


def safe_print(
    text: str,
    file: Optional[object] = None,
    end: str = "\n",
    force_ascii: bool = False,
) -> None:
    """Print text safely on Windows (GBK) terminals.

    Args:
        text: Text to print.
        file: Output stream (default sys.stdout).
        end: String appended after the last value.
        force_ascii: If True, always use ASCII fallback.
    """
    target = file or sys.stdout
    encoding = getattr(target, "encoding", None) or "utf-8"

    if force_ascii or encoding.lower() in ("gbk", "gb2312", "cp936", "ascii"):
        text = ascii_fallback(text)

    text = ascii_fallback(text) if force_ascii or encoding.lower() in ("gbk", "gb2312", "cp936", "ascii") else text
    try:
        target.write(text + end)
        target.flush()
    except UnicodeEncodeError:
        encoded = (text + end).encode(target.encoding or "utf-8", errors="replace").decode(target.encoding or "utf-8")
        target.write(encoded)
        target.flush()
