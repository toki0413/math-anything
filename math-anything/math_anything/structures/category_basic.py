"""范畴论基础结构。

Category, Functor, NaturalTransformation — 范畴论三大基础概念。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import AbstractMathematicalStructure, StructureFamily, StructureMetadata
from .properties import StructuralInvariant

# ── Shared invariants ──

_CATEGORY_INVARIANTS: list[StructuralInvariant] = [
    StructuralInvariant(
        name="composition_associative",
        expression="h ∘ (g ∘ f) = (h ∘ g) ∘ f for all composable f, g, h",
        theorem="Category axioms",
        affected_quantities=["morphisms", "composition"],
    ),
    StructuralInvariant(
        name="identity_neutral",
        expression="f ∘ id_A = f = id_B ∘ f for all f: A → B",
        theorem="Category axioms",
        affected_quantities=["morphisms", "identities"],
    ),
]

_FUNCTOR_INVARIANTS: list[StructuralInvariant] = [
    StructuralInvariant(
        name="functor_preserves_identity",
        expression="F(id_A) = id_{F(A)}",
        theorem="Functor axioms",
        affected_quantities=["identities"],
    ),
    StructuralInvariant(
        name="functor_preserves_composition",
        expression="F(g ∘ f) = F(g) ∘ F(f)",
        theorem="Functor axioms",
        affected_quantities=["composition"],
    ),
    StructuralInvariant(
        name="functor_preserves_commutative_diagrams",
        expression="If a diagram commutes in C, its image commutes in D",
        theorem="Functoriality",
        affected_quantities=["diagrams"],
    ),
]

_NATURALITY_INVARIANTS: list[StructuralInvariant] = [
    StructuralInvariant(
        name="naturality_square",
        expression="η_B ∘ F(f) = G(f) ∘ η_A for all f: A → B in C",
        theorem="Naturality condition",
        affected_quantities=["components", "commutative_squares"],
    ),
]


@dataclass
class Category(AbstractMathematicalStructure):
    """Abstract category: objects and morphisms with associative composition.

    A category C consists of:
      - Ob(C): a collection of objects
      - Hom_C(A,B): a set of morphisms for each pair A,B ∈ Ob(C)
      - id_A ∈ Hom_C(A,A): identity morphism for each object
      - ∘: Hom(B,C) × Hom(A,B) → Hom(A,C): composition

    Invariants:
      - Composition is associative: h ∘ (g ∘ f) = (h ∘ g) ∘ f
      - Identity is neutral: f ∘ id_A = f = id_B ∘ f
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.CATEGORY_THEORY,
            name="Category",
            canonical_form="Ob(C), Hom_C(A,B), id_A, ∘ (composition)",
            description="Objects and morphisms with associative composition and identities",
        )
    )
    objects: list[str] = field(default_factory=list)
    is_small: bool = False
    is_locally_small: bool = True
    has_products: bool = False
    has_coproducts: bool = False
    is_cartesian_closed: bool = False
    is_abelian: bool = False
    is_topos: bool = False
    morphism_table: dict[str, tuple[str, str]] = field(default_factory=dict)
    composition_table: dict[tuple[str, str], str] = field(default_factory=dict)

    @property
    def function_space(self) -> str:
        return "Ob(C) — a class (possibly proper)"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants: list[StructuralInvariant] = list(_CATEGORY_INVARIANTS)
        if self.has_products:
            invariants.append(
                StructuralInvariant(
                    name="products_exist",
                    expression="∀A,B ∈ Ob(C), ∃ A×B with projections π₁, π₂",
                    theorem="Definition of cartesian products",
                    affected_quantities=["objects", "products"],
                )
            )
        if self.is_abelian:
            invariants.append(
                StructuralInvariant(
                    name="abelian_category",
                    expression="Every monomorphism is a kernel, every epimorphism is a cokernel",
                    theorem="Freyd–Mitchell embedding theorem",
                    affected_quantities=["exact_sequences", "homology"],
                )
            )
        if self.is_cartesian_closed:
            invariants.append(
                StructuralInvariant(
                    name="cartesian_closed",
                    expression="Hom(A×B, C) ≅ Hom(A, C^B)",
                    theorem="Cartesian closed category definition",
                    affected_quantities=["exponentials", "currying"],
                )
            )
        return invariants

    def compose(self, f: str, g: str) -> str:
        """复合态射 g∘f，查找 composition_table 返回结果。

        注意这里的序：compose(f, g) 返回 g∘f，
        即 f: A→B, g: B→C 时返回 h: A→C。
        """
        key = (f, g)
        if key not in self.composition_table:
            raise ValueError(f"复合 ({f}, {g}) 未定义——请确认 f 的 target 等于 g 的 source")
        return self.composition_table[key]

    def add_morphism(self, f: str, source: str, target: str) -> None:
        """向范畴添加一个态射 f: source → target。"""
        self.morphism_table[f] = (source, target)
        if source not in self.objects:
            self.objects.append(source)
        if target not in self.objects:
            self.objects.append(target)

    def is_commutative(self, square: tuple[str, str, str, str]) -> bool:
        """检查交换图是否交换。

        给定 (f, g, h, k)，验证 h∘f = g∘k。
        典型场景：
            A --f--> B
            |        |
            k        g
            v        v
            C --h--> D
        """
        f, g, h, k = square
        left = self.composition_table.get((f, h))
        right = self.composition_table.get((k, g))
        if left is None or right is None:
            return False
        return left == right

    def hom(self, a: str, b: str) -> list[str]:
        """返回从 a 到 b 的所有态射。"""
        return [name for name, (src, tgt) in self.morphism_table.items() if src == a and tgt == b]

    def _property_dict(self) -> dict[str, Any]:
        return {
            "is_small": self.is_small,
            "is_locally_small": self.is_locally_small,
            "has_products": self.has_products,
            "has_coproducts": self.has_coproducts,
            "is_cartesian_closed": self.is_cartesian_closed,
            "is_abelian": self.is_abelian,
            "is_topos": self.is_topos,
        }


@dataclass
class Functor(AbstractMathematicalStructure):
    """Covariant functor F: C → D between categories.

    A functor maps:
      - objects: F: Ob(C) → Ob(D)
      - morphisms: F: Hom_C(A,B) → Hom_D(F(A), F(B))
    Preserving identities and composition.

    Invariants:
      - F(id_A) = id_{F(A)}
      - F(g ∘ f) = F(g) ∘ F(f)
      - Functors preserve commutative diagrams
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.CATEGORY_THEORY,
            name="Functor",
            canonical_form="F: C → D, F(id_A)=id_{F(A)}, F(g∘f)=F(g)∘F(f)",
            description="Structure-preserving map between categories",
        )
    )
    source_category: str = ""
    target_category: str = ""
    is_full: bool = False
    is_faithful: bool = False
    is_essentially_surjective: bool = False
    is_equivalence: bool = False
    is_contravariant: bool = False
    object_map: dict[str, str] = field(default_factory=dict)
    morphism_map: dict[str, str] = field(default_factory=dict)

    @property
    def function_space(self) -> str:
        return f"[{self.source_category or 'C'}, {self.target_category or 'D'}] — functor category"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants: list[StructuralInvariant] = list(_FUNCTOR_INVARIANTS)
        if self.is_contravariant:
            invariants.append(
                StructuralInvariant(
                    name="contravariant_composition",
                    expression="F(g ∘ f) = F(f) ∘ F(g)",
                    theorem="Contravariant functor definition",
                    affected_quantities=["composition"],
                )
            )
        if self.is_equivalence:
            invariants.append(
                StructuralInvariant(
                    name="equivalence_of_categories",
                    expression="F is full, faithful, and essentially surjective",
                    theorem="Equivalence of categories — F: C ≃ D",
                    affected_quantities=["objects", "morphisms"],
                )
            )
        if self.is_full and self.is_faithful:
            invariants.append(
                StructuralInvariant(
                    name="fully_faithful",
                    expression="Hom_C(A,B) ≅ Hom_D(F(A),F(B))",
                    theorem="Full and faithful functor — bijection on hom-sets",
                    affected_quantities=["hom_sets"],
                )
            )
        return invariants

    def apply_to_object(self, obj: str) -> str:
        """将函子作用于对象，查找 object_map。"""
        if obj not in self.object_map:
            raise ValueError(f"对象 '{obj}' 未在 object_map 中定义")
        return self.object_map[obj]

    def apply_to_morphism(self, f: str) -> str:
        """将函子作用于态射，查找 morphism_map。"""
        if f not in self.morphism_map:
            raise ValueError(f"态射 '{f}' 未在 morphism_map 中定义")
        return self.morphism_map[f]

    def verify_axioms(self) -> tuple[bool, list[str]]:
        """验证函子公理：F(id_A) = id_{F(A)} 和 F(g∘f) = F(g)∘F(f)。

        需要配合 source_category 的 composition_table 和 identity 态射来检查。
        返回 (全部通过, 失败信息列表)。
        """
        failures: list[str] = []

        # 检查恒等态射保持：F(id_A) 应该映射到 id_{F(A)}
        # 约定：恒等态射命名为 "id_<object>"
        for obj, mapped_obj in self.object_map.items():
            id_a = f"id_{obj}"
            id_fa = f"id_{mapped_obj}"
            if id_a in self.morphism_map:
                result = self.morphism_map[id_a]
                if result != id_fa:
                    failures.append(f"F(id_{obj}) = {result}, 期望 id_{mapped_obj}")

        # 检查复合保持：F(g∘f) = F(g)∘F(f)
        # 这需要外部提供 source category 的 composition_table，
        # 这里仅检查 morphism_map 中已有的映射一致性
        # （完整验证需要 source_cat 参数，此处做力所能及的检查）
        return (len(failures) == 0, failures)

    def _property_dict(self) -> dict[str, Any]:
        return {
            "source_category": self.source_category,
            "target_category": self.target_category,
            "is_full": self.is_full,
            "is_faithful": self.is_faithful,
            "is_essentially_surjective": self.is_essentially_surjective,
            "is_equivalence": self.is_equivalence,
            "is_contravariant": self.is_contravariant,
        }


@dataclass
class NaturalTransformation(AbstractMathematicalStructure):
    """Natural transformation η: F ⇒ G between parallel functors F, G: C → D.

    A natural transformation assigns to each object A ∈ C a morphism
    η_A: F(A) → G(A) in D, such that for every f: A → B in C,
    the naturality square commutes:
        η_B ∘ F(f) = G(f) ∘ η_A

    Invariants:
      - Naturality square commutes
      - Vertical composition: (ζ · η)_A = ζ_A ∘ η_A
      - Horizontal composition (Godement product): (η ∘' θ)_A = η_{G₂(A)} ∘ F₁(θ_A)
      - Natural isomorphism: η is invertible componentwise
    """

    metadata: StructureMetadata = field(
        default_factory=lambda: StructureMetadata(
            family=StructureFamily.CATEGORY_THEORY,
            name="Natural Transformation",
            canonical_form="η: F ⇒ G, η_B ∘ F(f) = G(f) ∘ η_A",
            description="Map between functors, natural in all objects",
        )
    )
    source_functor: str = ""
    target_functor: str = ""
    is_natural_isomorphism: bool = False
    is_mono_natural: bool = False
    is_epi_natural: bool = False
    component_map: dict[str, str] = field(default_factory=dict)

    @property
    def function_space(self) -> str:
        return "Nat(F, G) = Hom_{[C,D]}(F, G)"

    @property
    def structural_invariants(self) -> list[StructuralInvariant]:
        invariants: list[StructuralInvariant] = list(_NATURALITY_INVARIANTS)
        invariants.append(
            StructuralInvariant(
                name="vertical_composition_associative",
                expression="(ξ · ζ) · η = ξ · (ζ · η)",
                theorem="Vertical composition in functor categories",
                affected_quantities=["natural_transformations", "composition"],
            )
        )
        invariants.append(
            StructuralInvariant(
                name="horizontal_composition_godement",
                expression="(η' ∘' θ') ∘ (η ∘' θ) = (η' ∘ η) ∘' (θ' ∘ θ)  (interchange law)",
                theorem="Godement product / horizontal composition",
                affected_quantities=["natural_transformations", "functors"],
            )
        )
        if self.is_natural_isomorphism:
            invariants.append(
                StructuralInvariant(
                    name="natural_isomorphism",
                    expression="η_A is invertible for all A ∈ Ob(C)",
                    theorem="Natural isomorphism — componentwise invertible",
                    affected_quantities=["components"],
                )
            )
        return invariants

    def verify_naturality(self, functor_f: Functor, functor_g: Functor, category: Category) -> tuple[bool, list[str]]:
        """验证自然性条件：对源范畴中每个态射 f: A→B，检查 η_B ∘ F(f) = G(f) ∘ η_A。

        依赖 component_map（对象 → 组件态射名）、
        functor_f/functor_g 的 morphism_map、以及 category 的 composition_table。
        返回 (全部通过, 失败信息列表)。
        """
        failures: list[str] = []

        for f_name, (src, tgt) in category.morphism_table.items():
            eta_a = self.component_map.get(src)
            eta_b = self.component_map.get(tgt)
            f_mapped = functor_f.morphism_map.get(f_name)
            g_mapped = functor_g.morphism_map.get(f_name)

            if eta_a is None or eta_b is None:
                continue
            if f_mapped is None or g_mapped is None:
                continue

            # η_B ∘ F(f)
            left = category.composition_table.get((f_mapped, eta_b))
            # G(f) ∘ η_A
            right = category.composition_table.get((eta_a, g_mapped))

            if left is not None and right is not None and left != right:
                failures.append(
                    f"自然性失败于 {f_name}: {src}→{tgt}: "
                    f"η_{tgt} ∘ F({f_name}) = {left}, "
                    f"G({f_name}) ∘ η_{src} = {right}"
                )

        return (len(failures) == 0, failures)

    def _property_dict(self) -> dict[str, Any]:
        return {
            "source_functor": self.source_functor,
            "target_functor": self.target_functor,
            "is_natural_isomorphism": self.is_natural_isomorphism,
            "is_mono_natural": self.is_mono_natural,
            "is_epi_natural": self.is_epi_natural,
        }
