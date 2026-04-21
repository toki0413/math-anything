"""Tests for Math Anything code generation module.

Tests the Harness Auto-Generator, SourceCodeAnalyzer, and ConstraintInference.
"""

import os
# Add parent to path
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from math_anything.codegen import (ConstraintInference, HarnessAutoGenerator,
                                   SourceCodeAnalyzer,
                                   extract_symbolic_constraints, quick_infer)
from math_anything.codegen.source_analyzer import (ExtractedCommand,
                                                   ExtractedParameter,
                                                   ExtractionConfidence)


class TestSourceCodeAnalyzer(unittest.TestCase):
    """Test source code analysis functionality."""

    def setUp(self):
        self.analyzer = SourceCodeAnalyzer()

    def test_detect_language_cpp(self):
        """Test language detection for C++ files."""
        test_cases = [
            ("test.cpp", "cpp"),
            ("test.c", "cpp"),
            ("test.h", "cpp"),
            ("test.hpp", "cpp"),
            ("test.py", "python"),
            ("test.java", "java"),
            ("test.unknown", "unknown"),
        ]

        for filename, expected in test_cases:
            with self.subTest(filename=filename):
                path = Path(filename)
                result = self.analyzer._detect_language(path)
                self.assertEqual(result, expected)

    def test_extract_description_from_comments(self):
        """Test extraction of descriptions from code comments."""
        lines = [
            "// This is a comment",
            "// Another comment",
            "class MyClass {",
            "    void method();",
            "}",
        ]

        result = self.analyzer._extract_description(lines, 3)
        self.assertIn("comment", result.lower())

    def test_infer_param_type(self):
        """Test parameter type inference from code context."""
        test_cases = [
            ("double x = atof(arg[0]);", "float"),
            ("int n = atoi(arg[1]);", "int"),
            ("string s = arg[2];", "string"),
            ("unknown code", "unknown"),
        ]

        for code, expected in test_cases:
            with self.subTest(code=code):
                result = self.analyzer._infer_param_type(code)
                self.assertEqual(result, expected)

    def test_extraction_confidence_enum(self):
        """Test extraction confidence levels."""
        self.assertEqual(ExtractionConfidence.HIGH.value, "high")
        self.assertEqual(ExtractionConfidence.MEDIUM.value, "medium")
        self.assertEqual(ExtractionConfidence.LOW.value, "low")
        self.assertEqual(ExtractionConfidence.HEURISTIC.value, "heuristic")


class TestConstraintInference(unittest.TestCase):
    """Test constraint inference functionality."""

    def setUp(self):
        self.inferencer = ConstraintInference()

    def test_extract_variables_from_constraint(self):
        """Test variable extraction from constraint expressions."""
        test_cases = [
            ("tau_T > 0", ["tau_T"]),
            (
                "dt < dx_squared / (2*D_coeff)",
                ["dt", "dx_squared", "D_coeff"],
            ),  # Use longer names
            ("var_x + var_y > 10", ["var_x", "var_y"]),  # Use longer names
        ]

        for expr, expected_vars in test_cases:
            with self.subTest(expr=expr):
                result = self.inferencer._extract_variables(expr)
                # Check that all expected vars are in result
                for var in expected_vars:
                    self.assertIn(var, result)

    def test_quick_infer(self):
        """Test quick constraint inference convenience function."""
        contexts = [
            'if (x > 0) error->all("Must be positive")',
            "range: [0, 1]",
        ]

        result = quick_infer(contexts)
        self.assertIsInstance(result, list)

    def test_extract_symbolic_constraints(self):
        """Test symbolic constraint extraction from code."""
        code = """
        if (dt > tau_T / 10.0) error->warning("Time step too large");
        if (temperature <= 0) error->all("Must be positive");
        """

        constraints = extract_symbolic_constraints(
            code,
            parameters=["dt", "tau_T", "temperature"],
        )

        self.assertIsInstance(constraints, list)
        # Should find at least one constraint
        self.assertGreater(len(constraints), 0)

        for c in constraints:
            self.assertIsNotNone(c.expression)
            self.assertIsInstance(c.variables, list)

    def test_constraint_inference_from_code_patterns(self):
        """Test inference of constraints from common code patterns."""
        parameters = [
            {"name": "timestep"},
            {"name": "temperature"},
        ]

        command_contexts = {
            "test": [
                'if (timestep <= 0) error("Must be positive")',
                'if (temperature > 1000) warning("Very high")',
            ]
        }

        constraints = self.inferencer.infer(
            parameters=parameters,
            command_contexts=command_contexts,
        )

        self.assertIsInstance(constraints, list)


class TestHarnessAutoGenerator(unittest.TestCase):
    """Test Harness auto-generator functionality."""

    def setUp(self):
        self.generator = HarnessAutoGenerator()

    def test_to_class_name(self):
        """Test conversion of engine name to class name."""
        test_cases = [
            ("lammps", "Lammps"),
            ("my_engine", "MyEngine"),
            ("test-engine", "Test-engine"),  # Hyphen not handled
        ]

        for engine, expected in test_cases:
            with self.subTest(engine=engine):
                result = self.generator._to_class_name(engine)
                self.assertEqual(result, expected)

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        analysis = {
            "commands": [{"name": "cmd1"}, {"name": "cmd2"}],
            "parameters": [{"name": "p1"}],
            "equations": [{"form": "x = y"}],
        }

        confidence = self.generator._calculate_confidence(analysis)

        self.assertIn("command_extraction", confidence)
        self.assertIn("parameter_extraction", confidence)
        self.assertIn("equation_mapping", confidence)
        self.assertIn("overall", confidence)

        # Check ranges
        for key, value in confidence.items():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 100)

    def test_create_review_checklist(self):
        """Test review checklist generation."""
        analysis = {
            "commands": [
                {"name": "fix_nvt", "pattern": "fix nvt"},
            ],
        }
        mappings = {
            "equations": [
                {"name": "NVT Thermostat", "form": "dT/dt = ..."},
            ],
        }
        confidence = {"overall": 65.5}

        checklist = self.generator._create_review_checklist(
            analysis, mappings, confidence
        )

        self.assertIsInstance(checklist, list)
        self.assertGreater(len(checklist), 0)

        # Check that confidence is included
        confidence_items = [
            item for item in checklist if "65.5" in item or "confidence" in item.lower()
        ]
        self.assertGreater(len(confidence_items), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full workflow."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from analysis to generation."""
        # Create a temporary source file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple mock source file
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            mock_cpp = src_dir / "test_fix.cpp"
            mock_cpp.write_text("""
/* Test fix for NVT
   Governing equation: dT/dt = (T_target - T) / tau
   Constraint: tau > 0
*/
class FixNVT : public Fix {
public:
    FixNVT(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
        double tau = atof(arg[3]);  // Thermostat damping
        if (tau <= 0) error->all("Must be positive");
    }
};
""")

            # Run analysis
            analyzer = SourceCodeAnalyzer()
            analysis = analyzer.analyze(str(src_dir), file_patterns=["*.cpp"])

            # Verify analysis results
            self.assertIn("commands", analysis)
            self.assertIn("parameters", analysis)
            self.assertIn("coverage_metrics", analysis)
            self.assertIn("warnings", analysis)

            # Check coverage metrics
            coverage = analysis["coverage_metrics"]
            self.assertIn("estimated_command_coverage", coverage)
            self.assertIn("reliability", coverage)

            # Run inference
            inferencer = ConstraintInference()
            constraints = inferencer.infer(
                parameters=analysis["parameters"],
                command_contexts=analysis["command_contexts"],
            )

            self.assertIsInstance(constraints, list)


class TestEncodingConsistency(unittest.TestCase):
    """Test that file operations use consistent UTF-8 encoding."""

    def test_file_operations_use_utf8(self):
        """Test that analyzer uses UTF-8 encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.cpp"
            # Write file with UTF-8 content
            test_file.write_text(
                "// Test: 中文注释\nclass TestClass {};", encoding="utf-8"
            )

            # Analyze should handle UTF-8
            analyzer = SourceCodeAnalyzer()
            result = analyzer.analyze(tmpdir, file_patterns=["*.cpp"])

            # Should not raise encoding errors
            self.assertIn("commands", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
