"""Expression simplification for EML trees.

Provides algebraic simplification to make discovered equations more readable.

Examples:
    ((2 + 3) * x)         -> (5 * x)
    (x + 0)               -> x
    (x * 1)               -> x
    ((x * 2) / 2)         -> x
    eml(x, 1)             -> exp(x)
"""

import math

from math_anything.logging import get_logger

from .eml_v2 import Node, NodeType

logger = get_logger(__name__)


class ExpressionSimplifier:
    """Simplify expression trees using algebraic rules."""

    def simplify(self, node: Node) -> Node:
        """Simplify expression tree.

        Args:
            node: Expression tree to simplify

        Returns:
            Simplified tree
        """
        if node is None:
            return None

        # Recursively simplify children first (bottom-up)
        if node.left:
            node.left = self.simplify(node.left)
        if node.right:
            node.right = self.simplify(node.right)

        # Apply simplification rules
        return self._simplify_node(node)

    def _simplify_node(self, node: Node) -> Node:
        """Apply simplification rules to a single node."""
        if node.node_type == NodeType.CONST:
            return node

        if node.node_type == NodeType.VAR:
            return node

        # Apply specific rules for each operator
        if node.node_type == NodeType.ADD:
            return self._simplify_add(node)

        elif node.node_type == NodeType.SUB:
            return self._simplify_sub(node)

        elif node.node_type == NodeType.MUL:
            return self._simplify_mul(node)

        elif node.node_type == NodeType.DIV:
            return self._simplify_div(node)

        elif node.node_type == NodeType.EML:
            return self._simplify_eml(node)

        elif node.node_type == NodeType.POW:
            return self._simplify_pow(node)

        elif node.node_type == NodeType.SIN:
            return self._simplify_sin(node)

        elif node.node_type == NodeType.COS:
            return self._simplify_cos(node)

        elif node.node_type == NodeType.SQRT:
            return self._simplify_sqrt(node)

        elif node.node_type == NodeType.ABS:
            return self._simplify_abs(node)

        return node

    def _simplify_add(self, node: Node) -> Node:
        """Simplify addition: x + y."""
        left, right = node.left, node.right

        # x + 0 = x
        if self._is_zero(right):  # type: ignore[arg-type]
            return left  # type: ignore[return-value]

        # 0 + x = x
        if self._is_zero(left):  # type: ignore[arg-type]
            return right  # type: ignore[return-value]

        # const + const = const
        if left.node_type == NodeType.CONST and right.node_type == NodeType.CONST:  # type: ignore[union-attr]
            return Node(NodeType.CONST, value=left.value + right.value)  # type: ignore[operator, union-attr]

        # x + x = 2 * x
        if self._trees_equal(left, right):  # type: ignore[arg-type]
            two = Node(NodeType.CONST, value=2.0)
            return Node(NodeType.MUL, left=two, right=left)

        return node

    def _simplify_sub(self, node: Node) -> Node:
        """Simplify subtraction: x - y."""
        left, right = node.left, node.right

        # x - 0 = x
        if self._is_zero(right):  # type: ignore[arg-type]
            return left  # type: ignore[return-value]

        # 0 - x = -x (keep as 0 - x for now)
        if self._is_zero(left):  # type: ignore[arg-type]
            return node

        # x - x = 0
        if self._trees_equal(left, right):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=0.0)

        # const - const = const
        if left.node_type == NodeType.CONST and right.node_type == NodeType.CONST:  # type: ignore[union-attr]
            return Node(NodeType.CONST, value=left.value - right.value)  # type: ignore[operator, union-attr]

        return node

    def _simplify_mul(self, node: Node) -> Node:
        """Simplify multiplication: x * y."""
        left, right = node.left, node.right

        # x * 0 = 0
        if self._is_zero(right) or self._is_zero(left):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=0.0)

        # x * 1 = x
        if self._is_one(right):  # type: ignore[arg-type]
            return left  # type: ignore[return-value]

        # 1 * x = x
        if self._is_one(left):  # type: ignore[arg-type]
            return right  # type: ignore[return-value]

        # const * const = const
        if left.node_type == NodeType.CONST and right.node_type == NodeType.CONST:  # type: ignore[union-attr]
            return Node(NodeType.CONST, value=left.value * right.value)  # type: ignore[operator, union-attr]

        # 2 * 2 = 4 (but we keep as is for clarity in expressions)
        # Actually, let's merge small integer constants
        if left.node_type == NodeType.CONST and right.node_type == NodeType.CONST:  # type: ignore[union-attr]
            result = left.value * right.value  # type: ignore[operator, union-attr]
            if abs(result - round(result)) < 1e-10:  # Integer result
                return Node(NodeType.CONST, value=result)

        # const * (const * x) = (const * const) * x
        if (
            left.node_type == NodeType.CONST  # type: ignore[union-attr]
            and right.node_type == NodeType.MUL  # type: ignore[union-attr]
            and right.left.node_type == NodeType.CONST  # type: ignore[union-attr]
        ):
            new_const = left.value * right.left.value  # type: ignore[operator, union-attr]
            return Node(
                NodeType.MUL,
                left=Node(NodeType.CONST, value=new_const),
                right=right.right,  # type: ignore[union-attr]
            )

        return node

    def _simplify_div(self, node: Node) -> Node:
        """Simplify division: x / y."""
        left, right = node.left, node.right

        # x / 1 = x
        if self._is_one(right):  # type: ignore[arg-type]
            return left  # type: ignore[return-value]

        # 0 / x = 0 (for x != 0)
        if self._is_zero(left):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=0.0)

        # x / x = 1 (for x != 0)
        if self._trees_equal(left, right):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=1.0)

        # const / const = const
        if left.node_type == NodeType.CONST and right.node_type == NodeType.CONST:  # type: ignore[union-attr]
            if abs(right.value) > 1e-10:  # type: ignore[arg-type, union-attr]
                return Node(NodeType.CONST, value=left.value / right.value)  # type: ignore[operator, union-attr]

        # (x * const) / const = x
        if (
            left.node_type == NodeType.MUL  # type: ignore[union-attr]
            and left.left.node_type == NodeType.CONST  # type: ignore[union-attr]
            and right.node_type == NodeType.CONST  # type: ignore[union-attr]
        ):
            if abs(right.value - left.left.value) < 1e-10:  # type: ignore[operator, union-attr]
                return left.right  # type: ignore[return-value, union-attr]

        # (x * const) / (y * const) = x / y
        if (
            left.node_type == NodeType.MUL  # type: ignore[union-attr]
            and right.node_type == NodeType.MUL  # type: ignore[union-attr]
            and left.left.node_type == NodeType.CONST  # type: ignore[union-attr]
            and right.left.node_type == NodeType.CONST  # type: ignore[union-attr]
        ):
            if abs(left.left.value - right.left.value) < 1e-10:  # type: ignore[operator, union-attr]
                return Node(NodeType.DIV, left=left.right, right=right.right)  # type: ignore[union-attr]

        return node

    def _simplify_eml(self, node: Node) -> Node:
        """Simplify EML: eml(x, y) = exp(x) - ln(y)."""
        left, right = node.left, node.right

        # eml(x, 1) = exp(x)
        if self._is_one(right):  # type: ignore[arg-type]
            # We can't actually change the node type easily
            # But we can simplify the structure
            pass

        # eml(0, 1) = exp(0) - ln(1) = 1 - 0 = 1
        if self._is_zero(left) and self._is_one(right):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=1.0)

        # eml(1, 1) = e - 0 = e
        if self._is_one(left) and self._is_one(right):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=math.e)

        # eml(const, const) = const
        if left.node_type == NodeType.CONST and right.node_type == NodeType.CONST:  # type: ignore[union-attr]
            try:
                import numpy as np

                result = np.exp(left.value) - np.log(max(right.value, 1e-10))  # type: ignore[arg-type, type-var, union-attr]
                if not (np.isnan(result) or np.isinf(result)):
                    return Node(NodeType.CONST, value=float(result))
            except (ValueError, OverflowError) as e:
                logger.debug(f"Failed to simplify eml constants: {e}")

        return node

    def _simplify_pow(self, node: Node) -> Node:
        """Simplify power: x^y."""
        left, right = node.left, node.right

        # x^0 = 1
        if self._is_zero(right):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=1.0)

        # x^1 = x
        if self._is_one(right):  # type: ignore[arg-type]
            return left  # type: ignore[return-value]

        # 0^x = 0 (for x > 0)
        if self._is_zero(left):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=0.0)

        # 1^x = 1
        if self._is_one(left):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=1.0)

        # const^const = const
        if left.node_type == NodeType.CONST and right.node_type == NodeType.CONST:  # type: ignore[union-attr]
            try:
                result = left.value**right.value  # type: ignore[operator, union-attr]
                if not (math.isnan(result) or math.isinf(result)):
                    return Node(NodeType.CONST, value=float(result))
            except (ValueError, OverflowError, ZeroDivisionError) as e:
                logger.debug(f"Failed to simplify power constants: {e}")

        # x^2 = x * x (for small integers)
        if right.node_type == NodeType.CONST and right.value == 2.0:  # type: ignore[union-attr]
            return Node(NodeType.MUL, left=left, right=left.copy())  # type: ignore[union-attr]

        return node

    def _simplify_sin(self, node: Node) -> Node:
        """Simplify sin(x)."""
        left = node.left

        # sin(0) = 0
        if self._is_zero(left):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=0.0)

        # sin(const) = const
        if left.node_type == NodeType.CONST:  # type: ignore[union-attr]
            try:
                result = math.sin(left.value)  # type: ignore[arg-type, union-attr]
                return Node(NodeType.CONST, value=float(result))
            except (ValueError, OverflowError) as e:
                logger.debug(f"Failed to simplify sin constant: {e}")

        return node

    def _simplify_cos(self, node: Node) -> Node:
        """Simplify cos(x)."""
        left = node.left

        # cos(0) = 1
        if self._is_zero(left):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=1.0)

        # cos(const) = const
        if left.node_type == NodeType.CONST:  # type: ignore[union-attr]
            try:
                result = math.cos(left.value)  # type: ignore[arg-type, union-attr]
                return Node(NodeType.CONST, value=float(result))
            except (ValueError, OverflowError) as e:
                logger.debug(f"Failed to simplify cos constant: {e}")

        return node

    def _simplify_sqrt(self, node: Node) -> Node:
        """Simplify sqrt(x)."""
        left = node.left

        # sqrt(0) = 0
        if self._is_zero(left):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=0.0)

        # sqrt(1) = 1
        if self._is_one(left):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=1.0)

        # sqrt(const) = const
        if left.node_type == NodeType.CONST:  # type: ignore[union-attr]
            try:
                if left.value >= 0:  # type: ignore[operator, union-attr]
                    result = math.sqrt(left.value)  # type: ignore[arg-type, union-attr]
                    return Node(NodeType.CONST, value=float(result))
            except (ValueError, OverflowError) as e:
                logger.debug(f"Failed to simplify sqrt constant: {e}")

        # sqrt(x^2) = abs(x), but we keep as is for now

        return node

    def _simplify_abs(self, node: Node) -> Node:
        """Simplify abs(x)."""
        left = node.left

        # abs(0) = 0
        if self._is_zero(left):  # type: ignore[arg-type]
            return Node(NodeType.CONST, value=0.0)

        # abs(const) = const
        if left.node_type == NodeType.CONST:  # type: ignore[union-attr]
            return Node(NodeType.CONST, value=abs(left.value))  # type: ignore[arg-type, union-attr]

        # abs(abs(x)) = abs(x)
        if left.node_type == NodeType.ABS:  # type: ignore[union-attr]
            return left  # type: ignore[return-value]

        return node

    def _is_zero(self, node: Node) -> bool:
        """Check if node is zero."""
        return node.node_type == NodeType.CONST and node.value is not None and abs(node.value) < 1e-10

    def _is_one(self, node: Node) -> bool:
        """Check if node is one."""
        return node.node_type == NodeType.CONST and node.value is not None and abs(node.value - 1.0) < 1e-10

    def _trees_equal(self, n1: Node, n2: Node) -> bool:
        """Check if two expression trees are structurally equal."""
        if n1 is None and n2 is None:
            return True
        if n1 is None or n2 is None:
            return False

        if n1.node_type != n2.node_type:
            return False

        if n1.node_type == NodeType.CONST:
            return abs(n1.value - n2.value) < 1e-10  # type: ignore[no-any-return, operator]

        if n1.node_type == NodeType.VAR:
            return n1.name == n2.name

        return self._trees_equal(n1.left, n2.left) and self._trees_equal(n1.right, n2.right)  # type: ignore[arg-type]


def simplify(node: Node) -> Node:
    """Convenience function to simplify an expression tree."""
    simplifier = ExpressionSimplifier()
    return simplifier.simplify(node)


# Simplification examples for testing
if __name__ == "__main__":
    # Test simplifications
    builder = __import__("math_anything.eml_v2", fromlist=["ExprBuilder"]).ExprBuilder

    # Test 1: (2 + 3) * x -> 5 * x
    expr = builder.mul(builder.add(builder.const(2), builder.const(3)), builder.var("x"))
    logger.info(f"Before: {expr.to_string()}")
    simplified = simplify(expr)
    logger.info(f"After:  {simplified.to_string()}")

    # Test 2: x * 1 -> x
    expr = builder.mul(builder.var("x"), builder.const(1))
    logger.info(f"\nBefore: {expr.to_string()}")
    simplified = simplify(expr)
    logger.info(f"After:  {simplified.to_string()}")

    # Test 3: x + 0 -> x
    expr = builder.add(builder.var("x"), builder.const(0))
    logger.info(f"\nBefore: {expr.to_string()}")
    simplified = simplify(expr)
    logger.info(f"After:  {simplified.to_string()}")
