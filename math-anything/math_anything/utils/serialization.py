"""Secure serialization utilities.

Provides safe pickle alternatives with class whitelisting and HMAC
integrity verification to prevent arbitrary code execution.
"""

import hashlib
import hmac
import io
import pickle
from typing import Any


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that restricts allowed classes to prevent arbitrary code execution."""

    ALLOWED_CLASSES = {
        # builtins
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bool"),
        ("builtins", "bytes"),
        ("builtins", "NoneType"),
        # numpy
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy", "float64"),
        ("numpy", "int64"),
        # collections
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "Counter"),
        # dataclasses used in streaming_parser checkpoints
        ("math_anything.utils.streaming_parser", "Checkpoint"),
        ("math_anything.utils.streaming_parser", "TrajectoryStats"),
        ("math_anything.utils.streaming_parser", "SamplingConfig"),
        ("math_anything.utils.streaming_parser", "SamplingStrategy"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED_CLASSES:
            raise pickle.UnpicklingError(
                f"Forbidden class: {module}.{name}. If this class is needed, add it to SafeUnpickler.ALLOWED_CLASSES."
            )
        return super().find_class(module, name)


def safe_loads(data: bytes) -> Any:
    """Safely deserialize pickle data with class whitelist."""
    return SafeUnpickler(io.BytesIO(data)).load()


def safe_load(file_obj) -> Any:
    """Safely deserialize pickle data from a file object with class whitelist."""
    return SafeUnpickler(file_obj).load()


def signed_dumps(obj: Any, secret_key: str = "") -> bytes:
    """Serialize with HMAC signature for integrity verification."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if secret_key:
        signature = hmac.new(secret_key.encode(), data, hashlib.sha256).digest()
        return signature + data
    return data


def signed_loads(data: bytes, secret_key: str = "") -> Any:
    """Deserialize with HMAC signature verification."""
    if secret_key:
        stored_sig = data[:32]
        payload = data[32:]
        expected_sig = hmac.new(secret_key.encode(), payload, hashlib.sha256).digest()
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise ValueError("Data integrity check failed: HMAC signature mismatch")
        return safe_loads(payload)
    return safe_loads(data)
