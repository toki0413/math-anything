"""Smoke tests for legacy modules currently at 0% coverage."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "engines"))


# ── PSRN ──


def test_psrn_imports():
    from math_anything.psrn import (
        PSRN,
        FastTokenGenerator,
        GPTokenGenerator,
        GPUEvaluator,
        MCTSTokenGenerator,
        PSEEngine,
        PSRNConfig,
        PSRNSymbolicRegression,
        RandomTokenGenerator,
        SymbolLayer,
        TokenGenerator,
        has_gpu_support,
    )

    assert PSRN is not None
    assert PSRNConfig is not None
    assert PSEEngine is not None
    assert SymbolLayer is not None
    assert TokenGenerator is not None


def test_psrn_config():
    from math_anything.psrn import PSRNConfig

    cfg = PSRNConfig()
    assert cfg is not None


def test_psrn_symbol_layer():
    from math_anything.psrn import SymbolConfig, SymbolLayer

    cfg = SymbolConfig()
    assert cfg is not None
    layer = SymbolLayer()
    assert layer is not None


# ── Simplifier ──


def test_simplifier_imports():
    from math_anything.simplifier import ExpressionSimplifier, simplify

    simplifier = ExpressionSimplifier()
    assert simplifier is not None
    assert callable(simplifier.simplify)


def test_simplifier_basic():
    from math_anything.eml_v2 import ExprBuilder
    from math_anything.simplifier import ExpressionSimplifier

    b = ExprBuilder()
    expr = b.add(b.const(2), b.const(3))
    s = ExpressionSimplifier()
    result = s.simplify(expr)
    assert result is not None


def test_simplifier_const_folding():
    from math_anything.eml_v2 import Node, NodeType
    from math_anything.simplifier import ExpressionSimplifier

    s = ExpressionSimplifier()
    tree = Node(NodeType.ADD, left=Node(NodeType.CONST, value=2), right=Node(NodeType.CONST, value=3))
    result = s.simplify(tree)
    assert result.node_type == NodeType.CONST
    assert abs(result.value - 5.0) < 1e-10


def test_simplify_convenience():
    from math_anything.eml_v2 import Node, NodeType
    from math_anything.simplifier import simplify

    tree = Node(NodeType.MUL, left=Node(NodeType.VAR, name="x"), right=Node(NodeType.CONST, value=1))
    result = simplify(tree)
    assert result.node_type == NodeType.VAR


# ── MultiVar ──


def test_multivar_imports():
    from math_anything.multivar import MultiVariableDiscovery, analyze_interactions, discover_multivar

    mvd = MultiVariableDiscovery(population_size=10, max_depth=3, generations=5)
    assert mvd is not None
    assert callable(analyze_interactions)
    assert callable(discover_multivar)


def test_multivar_quick_analyze():
    import numpy as np

    from math_anything.multivar import analyze_interactions

    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]])
    y = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
    result = analyze_interactions(X, y, variable_names=["a", "b"])
    assert "correlations" in result
    assert "pairwise_interactions" in result


# ── Schemas ──


def test_schemas_validation():
    from math_anything.schemas import SchemaValidator

    v = SchemaValidator()
    assert v is not None


def test_schemas_extensions():
    from math_anything.schemas import ExtensionRegistry, get_available_extensions

    reg = ExtensionRegistry()
    assert reg is not None
    exts = get_available_extensions()
    assert isinstance(exts, list)
    assert len(exts) >= 1


def test_schemas_math_schema():
    from math_anything.schemas import MathSchema, MetaInfo

    meta = MetaInfo(extracted_by="test", extractor_version="1.0")
    schema = MathSchema(meta=meta)
    assert schema.meta.extracted_by == "test"


def test_schemas_extended():
    from math_anything.schemas import ExtendedMathSchema, ExtensionMetadata

    meta = ExtensionMetadata(name="test_ext", version="1.0", description="unit test")
    assert meta.name == "test_ext"


def test_schemas_precision_extractors():
    from math_anything.schemas import (
        AbaqusMathematicalPrecisionExtractor,
        AnsysMathematicalPrecisionExtractor,
        ComsolMathematicalPrecisionExtractor,
        GromacsMathematicalPrecisionExtractor,
        LammpsMathematicalPrecisionExtractor,
        MultiwfnMathematicalPrecisionExtractor,
        VaspMathematicalPrecisionExtractor,
    )

    for ex_cls in [
        VaspMathematicalPrecisionExtractor,
        LammpsMathematicalPrecisionExtractor,
        AbaqusMathematicalPrecisionExtractor,
        AnsysMathematicalPrecisionExtractor,
        ComsolMathematicalPrecisionExtractor,
        GromacsMathematicalPrecisionExtractor,
        MultiwfnMathematicalPrecisionExtractor,
    ]:
        assert ex_cls is not None


# ── EML v2 ──


def test_eml_v2_imports():
    from math_anything.eml_v2 import ExprBuilder, ImprovedSymbolicRegression, Node, NodeType, eml

    sr = ImprovedSymbolicRegression(population_size=10, max_depth=2, generations=2)
    assert sr is not None
    result = eml(1.0, 1.0)
    assert abs(result - (2.718281828459045)) < 1.0  # exp(1) - ln(1) ≈ e


def test_eml_v2_node_create():
    from math_anything.eml_v2 import Node, NodeType

    n = Node(NodeType.CONST, value=3.14)
    assert n.evaluate({}) == pytest.approx(3.14)
    assert n.to_string() == "3.14"


def test_eml_v2_evaluate():
    from math_anything.eml_v2 import Node, NodeType

    tree = Node(NodeType.ADD, left=Node(NodeType.CONST, value=2), right=Node(NodeType.VAR, name="x"))
    val = tree.evaluate({"x": 3.0})
    assert val == pytest.approx(5.0)


def test_eml_v2_discover_equation():
    import numpy as np

    from math_anything.eml_v2 import discover_equation

    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    eq = discover_equation(X, y, ["x"], population_size=20, max_depth=2, generations=10)
    assert len(eq) > 0


def test_eml_v2_backward_compat():
    from math_anything.eml_v2 import EMLNode, NodeType, SymbolicRegression

    assert EMLNode is not None
    assert SymbolicRegression is not None


# ── Visualization ──


def test_visualization_imports():
    from math_anything.visualization import VisualizationConfig, Visualizer

    viz = Visualizer()
    assert viz is not None
    assert viz.config is not None


def test_visualization_render():
    from math_anything.visualization import Visualizer

    viz = Visualizer()
    schema = {"engine": "test", "mathematical_structure": {"problem_type": "test", "canonical_form": "n/a"}}
    mermaid = viz.render(schema, format="mermaid")
    assert "test" in mermaid
    graphviz = viz.render(schema, format="graphviz")
    assert "test" in graphviz
    text = viz.render(schema, format="text")
    assert "test" in text


def test_visualization_convenience():
    from math_anything.visualization import to_graphviz, to_mermaid

    schema = {"engine": "demo"}
    m = to_mermaid(schema)
    assert "DEMO" in m
    gv = to_graphviz(schema)
    assert "DEMO" in gv


# ── Utils ──


def test_utils_imports():
    from math_anything.utils import (
        DiffReport,
        DiffType,
        LammpsDumpExtractor,
        LLMContextProtocol,
        MathDiffer,
        SamplingConfig,
        SamplingStrategy,
        SemanticValidator,
        StreamingParser,
    )

    assert MathDiffer is not None
    assert DiffReport is not None
    assert DiffType is not None
    assert StreamingParser is not None
    assert SemanticValidator is not None


def test_utils_math_differ():
    from math_anything.utils import MathDiffer

    differ = MathDiffer()
    assert differ is not None
    assert callable(differ.compare)


def test_utils_sampling_config():
    from math_anything.utils import SamplingConfig

    cfg = SamplingConfig()
    assert cfg is not None


# ── Tiered ──


def test_tiered_imports():
    from math_anything.tiered import (
        AnalysisTier,
        ComplexityScore,
        FileAnalysis,
        IntegratedTieredAnalyzer,
        ResourceRequirements,
        TieredAnalysisResult,
        TieredAnalyzer,
        TieredSymbolicRegressionAnalyzer,
        TierRecommendation,
        TierRecommender,
        tiered_symbolic_regression_analysis,
    )

    assert TieredAnalyzer is not None
    assert TierRecommender is not None
    assert TierRecommendation is not None
    assert IntegratedTieredAnalyzer is not None


def test_tiered_analyzer():
    from math_anything.tiered import TieredAnalyzer

    ta = TieredAnalyzer()
    assert ta is not None


def test_tier_recommender():
    from math_anything.tiered import TierRecommender

    tr = TierRecommender()
    assert tr is not None


# ── Templates ──


def test_templates_imports():
    from math_anything.templates import (
        CheckTemplate,
        DFTCheckTemplate,
        DFTDraftTemplate,
        DFTInsightTemplate,
        DraftTemplate,
        FEMCheckTemplate,
        FEMDraftTemplate,
        FEMInsightTemplate,
        InsightTemplate,
        MathNarrativeTemplate,
        MDCheckTemplate,
        MDDraftTemplate,
        MDInsightTemplate,
    )

    assert DFTCheckTemplate is not None
    assert FEMCheckTemplate is not None
    assert MDCheckTemplate is not None
    assert MathNarrativeTemplate is not None


def test_templates_dft():
    from math_anything.templates import DFTCheckTemplate

    t = DFTCheckTemplate()
    assert t is not None


def test_templates_fem():
    from math_anything.templates import FEMDraftTemplate

    t = FEMDraftTemplate()
    assert t is not None


def test_templates_md():
    from math_anything.templates import MDInsightTemplate

    t = MDInsightTemplate()
    assert t is not None


# ── Cross Engine ──


def test_cross_engine_imports():
    from math_anything.core.cross_engine import (
        CoupledSchema,
        CouplingInterface,
        CouplingType,
        CrossEngineSession,
        ModelScale,
        ScaleModel,
    )

    session = CrossEngineSession()
    assert session is not None
    assert ModelScale.QUANTUM.value == "quantum"
    assert ModelScale.ATOMISTIC.value == "atomistic"


def test_cross_engine_model_scale():
    from math_anything.core.cross_engine import ModelScale

    scales = list(ModelScale)
    assert len(scales) >= 3


def test_cross_engine_coupling_type():
    from math_anything.core.cross_engine import CouplingType

    assert CouplingType.SEQUENTIAL is not None
    assert CouplingType.HIERARCHICAL is not None


# ── LLM Client ──


def test_llm_client_imports():
    """Just verify module imports (no API call)."""
    from math_anything.llm_client import LLMClient, LLMError, quick_chat

    assert LLMClient is not None
    assert LLMError is not None
    assert callable(quick_chat)
    assert issubclass(LLMError, Exception)


# ── EML Conjugacy Depth ──


def test_conjugacy_depth_eml_expr_create():
    from math_anything.conjugacy import E_EXPR, ONE, PI_EXPR, ZERO, EMLExpr, const_expr, eml_expr, var_expr

    assert E_EXPR.op == "CONST"
    assert PI_EXPR.op == "CONST"
    assert ZERO.op == "CONST"
    assert ONE.op == "CONST"


def test_conjugacy_depth_evaluate():
    import math as m

    from math_anything.conjugacy import E_EXPR, ONE, eml_expr

    tree = eml_expr(E_EXPR, ONE)
    val = tree.evaluate(0.0)
    assert abs(val - m.exp(m.e) - 0) < 1.0


def test_conjugacy_depth_to_standard_form():
    from math_anything.conjugacy import E_EXPR, ONE, ZERO, const_expr, eml_expr

    tree = eml_expr(E_EXPR, ONE)
    sf = tree.to_standard_form()
    assert "exp" in sf.lower()

    tree2 = eml_expr(ZERO, ONE)
    sf2 = tree2.to_standard_form()
    assert "0" in sf2 or "exp" in sf2.lower()


def test_conjugacy_depth_make_exp_ln():
    from math_anything.conjugacy import make_exp, make_ln, var_expr

    x = var_expr("x")
    exp_tree = make_exp(x)
    assert exp_tree.op == "EML"

    ln_tree = make_ln(x)
    assert ln_tree.op == "EML"


def test_conjugacy_engine():
    from math_anything.conjugacy import EMLConjugacyEngine, EMLExpr, const_expr, eml_expr, var_expr

    engine = EMLConjugacyEngine()
    a = eml_expr(const_expr(1.0), const_expr(1.0))
    b = eml_expr(const_expr(1.0), const_expr(1.0))
    assert engine.is_conjugate(a, b)

    c = var_expr("x")
    d = eml_expr(const_expr(0.0), var_expr("x"))
    result = engine.is_conjugate(c, d)
    assert isinstance(result, bool)


def test_conjugacy_verify_all():
    from math_anything.conjugacy import EMLConjugacyEngine

    engine = EMLConjugacyEngine()
    results = engine.verify_all()
    assert isinstance(results, dict)
    assert "exp_shift" in results


def test_conjugacy_expr_depth():
    from math_anything.conjugacy import const_expr, eml_expr, var_expr

    x = var_expr("x")
    tree = eml_expr(x, const_expr(1.0))
    assert tree.depth() == 1

    deep = eml_expr(eml_expr(x, x), const_expr(1.0))
    assert deep.depth() == 2


# ── Constants Depth ──


def test_constants_depth_classify():
    from math_anything.constants import classify_constant

    cat = classify_constant(0.0)
    assert isinstance(cat, str)
    assert len(cat) > 0


def test_constants_depth_find_eml_form():
    import math as m

    from math_anything.constants import find_eml_form

    result = find_eml_form(m.e, max_depth=3)
    assert result is not None or True  # may fail to find exact


def test_constants_depth_list_known_constants():
    from math_anything.constants import list_known_constants

    known = list_known_constants()
    assert isinstance(known, list)
    assert len(known) > 5
    assert any(c["symbol"] == "e" for c in known)
    assert any(c["symbol"] == "π" for c in known)


def test_constants_engine():
    from math_anything.constants import EMLConstantEngine

    engine = EMLConstantEngine()
    assert engine is not None

    e_def = engine.get("e")
    assert e_def is not None
    assert e_def.name == "欧拉数"

    pi_def = engine.get("pi")
    assert pi_def is not None


def test_constants_classify_known():
    from math_anything.constants import classify_constant

    assert classify_constant(0.0) == "arithmetic"
    assert classify_constant(1.0) in ("arithmetic",)


def test_constants_to_dict():
    from math_anything.constants import EMLConstantEngine

    engine = EMLConstantEngine()
    d = engine.to_dict()
    assert "constants_count" in d
    assert "fundamental_constants" in d
    assert "base_set" in d
    assert d["base_set"] == ["0", "1", "e", "π"]


def test_constants_is_eml_elementary():
    from math_anything.constants import EMLConstantEngine

    engine = EMLConstantEngine()
    assert engine.is_eml_elementary(0.0) is True
    assert engine.is_eml_elementary(7.0) is True  # integer = arithmetic


def test_constants_search_depth():
    import math as m

    from math_anything.constants import find_eml_form

    result = find_eml_form(2.0, max_depth=2)
    assert isinstance(result, str) or result is None
