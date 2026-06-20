"""Bourbaki — Mathematical Structure Modeling for Computational Science.

Bird is the Word.  — 致敬 Freeman Dyson, "Birds and Frogs" (2009)

尼古拉·布尔巴基（Nicolas Bourbaki）是 20 世纪最有影响力的数学家集体，
他们用结构主义的视角统一了全部数学：
  - 代数结构（群、环、域）
  - 序结构（偏序、格）
  - 拓扑结构（度量、拓扑空间）

Bourbaki 继承了布尔巴基的结构主义精神：
  物理、化学、力学、材料科学——所有计算科学——都是
  同一组数学结构的不同实例化的态射和函子。

Bird is the Word — 致敬 Freeman Dyson 的 "Birds and Frogs"：

  Dyson 将数学家分为两类：
    Birds  —— 飞翔在高空，俯瞰全貌，看见不同领域之间的深层联系
    Frogs  —— 深入泥泞之中，盯住具体问题，看见极近的细节

  Bourbaki 就是 Birds 的极致——他们飞在所有数学之上，
  看见一切计算背后统一的数学结构。

  而每天在跑 VASP、LAMMPS、OpenFOAM 的科学家们，
  他们是 Frogs——他们解决具体问题，推进着计算科学的前沿。

  "Bird is the Word" 的意思是：
  你在 VASP 里设的 ENCUT=520，在数学结构上是什么意思？
  这个计算在整个态的射链上处于哪个位置？
  为什么你的 DFT 计算和同仁的 CFD 计算本质上是同一个数学结构的不同实例化？

  Bird 的视角不在具体参数的误差条里。
  Bird 的视角在数学结构本身之中。
  而那，才是真正的 Word。

Architecture (v3.0):
  bourbaki.foundation   — algorithms & formal systems
  bourbaki.structures   — mathematical structure type system
  bourbaki.morphisms    — structure-preserving transformations
  bourbaki.domains       — physics discipline instantiations
"""

__version__ = "3.0.0"
__slogan__ = "Bird is the Word"

# ── Full re-export from math_anything ──
from math_anything import *  # noqa: F401,F403

# ── Explicit re-exports for key APIs ──
# Structures
from math_anything.structures import (
    ConservationMatrixField,
    FieldConservedQuantity,
    NoetherCurrent,
    NOETHER_CORRESPONDENCE,
    HamiltonianSystem,
    NavierStokesProblem,
    SelfConsistentProblem,
    EvolutionProblem,
    EquilibriumProblem,
    CoupledSystem,
    SmoothManifold,
    AbstractMathematicalStructure,
)

# Morphisms
from math_anything.morphisms import Morphism, CompositeMorphism

# Foundation layer
from math_anything.categories import CategoryEngine, MathKnowledgeGraph, GraphQueryEngine
from math_anything.constraints import ConstraintPropagation, LearnedInvariant, BoundaryEvolution
from math_anything.type_theory.checker import TypeChecker
from math_anything.dimensional.scaling_group import BuckinghamPiEngine

# Domain layer
from math_anything.domains import DOMAIN_REGISTRY, register_domain, list_domains, get_domain

# Sub-module re-exports (namespace access)
from math_anything import structures
from math_anything import morphisms
from math_anything import conjugacy
from math_anything import constants
from math_anything import categories
from math_anything import dimensional
from math_anything import constraints
from math_anything import schemas
from math_anything import bridge
from math_anything import api
from math_anything import cli
from math_anything import types
from math_anything import exceptions
from math_anything import logging
from math_anything import plugin
from math_anything import rust_bridge
from math_anything import foundation
from math_anything import domains

BANNER = r"""
    ____                              __   _    __ 
   / __ )____  ________  ____ ______/ /__(_)  / /_
  / __  / __ \/ ___/ _ \/ __ `/ ___/ //_/ /  / __/
 / /_/ / /_/ / /  /  __/ /_/ / /__/ ,< / /  / /_  
/_____/\____/_/   \___/\__,_/\___/_/|_/_/   \__/  
                                                   
           Bird is the Word
  Mathematical Structure Modeling for Computational Science
"""
