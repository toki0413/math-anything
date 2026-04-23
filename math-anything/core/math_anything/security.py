"""Security utilities for Math Anything.

Provides file path validation and access control to prevent path traversal
and other security issues.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Set, Union

from .exceptions import FileAccessError, SecurityError


class PathSecurityValidator:
    """Validates file paths for security.

    Prevents path traversal attacks and restricts access to allowed directories.
    """

    # Dangerous patterns that could indicate attacks
    DANGEROUS_PATTERNS = [
        r"\.\./",  # Unix parent directory
        r"\.\.\\",  # Windows parent directory
        r"\.\.$",  # Current directory reference
        r"%2e%2e",  # URL-encoded parent
        r"\x00",  # Null byte
        r"~",  # Home directory expansion
    ]

    # File extensions that are typically dangerous
    DANGEROUS_EXTENSIONS = {
        ".exe",
        ".dll",
        ".bat",
        ".cmd",
        ".sh",
        ".py",
        ".rb",
        ".pl",
    }

    def __init__(
        self,
        allowed_base_dirs: Optional[List[Union[str, Path]]] = None,
        allow_absolute_paths: bool = False,
        max_path_length: int = 4096,
    ):
        """Initialize path security validator.

        Args:
            allowed_base_dirs: List of directories that files can be accessed from.
                If None, only relative paths from current directory are allowed.
            allow_absolute_paths: Whether to allow absolute paths.
            max_path_length: Maximum allowed path length.
        """
        self.allowed_base_dirs = (
            [Path(d).resolve() for d in allowed_base_dirs]
            if allowed_base_dirs
            else None
        )
        self.allow_absolute_paths = allow_absolute_paths
        self.max_path_length = max_path_length

    def validate(self, filepath: Union[str, Path]) -> Path:
        """Validate a file path for security.

        Args:
            filepath: Path to validate.

        Returns:
            Resolved Path object if valid.

        Raises:
            FileAccessError: If path is invalid or unsafe.
            SecurityError: If security violation detected.
        """
        path_str = str(filepath)

        # Check path length
        if len(path_str) > self.max_path_length:
            raise FileAccessError(
                path_str,
                f"Path exceeds maximum length of {self.max_path_length} characters",
            )

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, path_str, re.IGNORECASE):
                raise SecurityError(
                    "PATH_TRAVERSAL",
                    f"Path contains dangerous pattern: {pattern}",
                )

        # Convert to Path object
        path = Path(filepath)

        # Check for dangerous extensions
        if path.suffix.lower() in self.DANGEROUS_EXTENSIONS:
            raise SecurityError(
                "DANGEROUS_EXTENSION",
                f"File extension '{path.suffix}' is not allowed",
            )

        # Check if original path is absolute
        original_is_absolute = path.is_absolute()

        # Resolve to absolute path
        try:
            resolved = path.resolve()
        except (OSError, ValueError) as e:
            raise FileAccessError(path_str, f"Invalid path: {e}")

        # Validate absolute paths (only if original was absolute)
        if original_is_absolute:
            if not self.allow_absolute_paths and self.allowed_base_dirs is None:
                raise FileAccessError(
                    path_str,
                    "Absolute paths are not allowed. Use relative paths or configure allowed directories.",
                )

            # Check if within allowed directories
            if self.allowed_base_dirs:
                in_allowed_dir = any(
                    self._is_subpath(resolved, allowed_dir)
                    for allowed_dir in self.allowed_base_dirs
                )
                if not in_allowed_dir:
                    raise FileAccessError(
                        path_str,
                        f"Path must be within allowed directories: {[str(d) for d in self.allowed_base_dirs]}",
                    )

        return resolved

    def _is_subpath(self, path: Path, potential_parent: Path) -> bool:
        """Check if path is a subpath of potential_parent."""
        try:
            path.relative_to(potential_parent)
            return True
        except ValueError:
            return False

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename for safe usage.

        Removes or replaces dangerous characters.

        Args:
            filename: Original filename.

        Returns:
            Sanitized filename.
        """
        # Remove null bytes
        sanitized = filename.replace("\x00", "")

        # Replace path separators
        sanitized = sanitized.replace("/", "_").replace("\\", "_")

        # Remove control characters
        sanitized = "".join(c for c in sanitized if ord(c) >= 32)

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")

        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[: 255 - len(ext)] + ext

        # Ensure not empty
        if not sanitized:
            sanitized = "unnamed"

        return sanitized


class FileSizeValidator:
    """Validates file sizes."""

    def __init__(
        self,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB default
        max_total_size: int = 500 * 1024 * 1024,  # 500MB default
    ):
        """Initialize file size validator.

        Args:
            max_file_size: Maximum size for individual files in bytes.
            max_total_size: Maximum total size for multiple files in bytes.
        """
        self.max_file_size = max_file_size
        self.max_total_size = max_total_size

    def validate_file(self, filepath: Union[str, Path]) -> int:
        """Validate a single file's size.

        Args:
            filepath: Path to file.

        Returns:
            File size in bytes.

        Raises:
            FileAccessError: If file is too large.
        """
        path = Path(filepath)

        if not path.exists():
            from .exceptions import FileNotFoundError

            raise FileNotFoundError(str(filepath))

        size = path.stat().st_size

        if size > self.max_file_size:
            raise FileAccessError(
                str(filepath),
                f"File size ({size} bytes) exceeds maximum allowed ({self.max_file_size} bytes)",
            )

        return size

    def validate_total_size(self, filepaths: List[Union[str, Path]]) -> int:
        """Validate total size of multiple files.

        Args:
            filepaths: List of file paths.

        Returns:
            Total size in bytes.

        Raises:
            FileAccessError: If total size is too large.
        """
        total_size = 0

        for filepath in filepaths:
            total_size += self.validate_file(filepath)

        if total_size > self.max_total_size:
            raise FileAccessError(
                str(filepaths[0]) if filepaths else "",
                f"Total file size ({total_size} bytes) exceeds maximum allowed ({self.max_total_size} bytes)",
            )

        return total_size


# Default validators with secure defaults
default_path_validator = PathSecurityValidator(
    allow_absolute_paths=False,
)

default_size_validator = FileSizeValidator()


def validate_filepath(
    filepath: Union[str, Path],
    validator: Optional[PathSecurityValidator] = None,
) -> Path:
    """Convenience function to validate a file path.

    Args:
        filepath: Path to validate.
        validator: Custom validator. Uses default if None.

    Returns:
        Resolved Path object.
    """
    v = validator or default_path_validator
    return v.validate(filepath)


def validate_file_size(
    filepath: Union[str, Path],
    validator: Optional[FileSizeValidator] = None,
) -> int:
    """Convenience function to validate file size.

    Args:
        filepath: Path to file.
        validator: Custom validator. Uses default if None.

    Returns:
        File size in bytes.
    """
    v = validator or default_size_validator
    return v.validate_file(filepath)
