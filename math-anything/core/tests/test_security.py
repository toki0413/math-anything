"""Tests for security utilities."""

import os
import tempfile
from pathlib import Path

import pytest
from math_anything.exceptions import FileAccessError, FileNotFoundError, SecurityError
from math_anything.security import (
    FileSizeValidator,
    PathSecurityValidator,
    default_path_validator,
    default_size_validator,
    validate_file_size,
    validate_filepath,
)


class TestPathSecurityValidator:
    """Test path security validator."""

    def test_valid_relative_path(self):
        """Test valid relative path."""
        validator = PathSecurityValidator()
        result = validator.validate("test.lmp")
        assert isinstance(result, Path)

    def test_path_traversal_unix(self):
        """Test Unix path traversal detection."""
        validator = PathSecurityValidator()
        with pytest.raises(SecurityError) as exc_info:
            validator.validate("../etc/passwd")
        assert "PATH_TRAVERSAL" in str(exc_info.value)

    def test_path_traversal_windows(self):
        """Test Windows path traversal detection."""
        validator = PathSecurityValidator()
        with pytest.raises(SecurityError) as exc_info:
            validator.validate("..\\windows\\system32\\config")
        assert "PATH_TRAVERSAL" in str(exc_info.value)

    def test_url_encoded_traversal(self):
        """Test URL-encoded path traversal."""
        validator = PathSecurityValidator()
        with pytest.raises(SecurityError) as exc_info:
            validator.validate("%2e%2e%2f%2e%2e%2fetc%2fpasswd")
        assert "PATH_TRAVERSAL" in str(exc_info.value)

    def test_dangerous_extension_exe(self):
        """Test dangerous .exe extension."""
        validator = PathSecurityValidator()
        with pytest.raises(SecurityError) as exc_info:
            validator.validate("malicious.exe")
        assert "DANGEROUS_EXTENSION" in str(exc_info.value)

    def test_dangerous_extension_script(self):
        """Test dangerous script extensions."""
        validator = PathSecurityValidator()
        with pytest.raises(SecurityError):
            validator.validate("run.sh")
        with pytest.raises(SecurityError):
            validator.validate("script.py")

    def test_absolute_path_denied(self):
        """Test absolute path denial."""
        validator = PathSecurityValidator(allow_absolute_paths=False)
        # Create a temporary directory and get its absolute path
        with tempfile.TemporaryDirectory() as tmpdir:
            # Construct an absolute path that doesn't exist
            abs_path = os.path.join(tmpdir, "nonexistent", "file.lmp")
            # Ensure this is recognized as absolute
            assert os.path.isabs(abs_path)
            with pytest.raises(FileAccessError) as exc_info:
                validator.validate(abs_path)
            assert "Absolute paths are not allowed" in str(exc_info.value)

    def test_absolute_path_allowed(self):
        """Test absolute path with allowed directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = PathSecurityValidator(
                allowed_base_dirs=[tmpdir],
                allow_absolute_paths=True,
            )
            test_file = Path(tmpdir) / "test.lmp"
            test_file.touch()
            result = validator.validate(str(test_file))
            assert result.is_absolute()

    def test_path_outside_allowed_dirs(self):
        """Test path outside allowed directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = PathSecurityValidator(allowed_base_dirs=[tmpdir])
            # Create a different temp dir to simulate "outside" path
            with tempfile.TemporaryDirectory() as other_tmpdir:
                outside_path = os.path.join(other_tmpdir, "file.lmp")
                with pytest.raises(FileAccessError) as exc_info:
                    validator.validate(outside_path)
                assert "within allowed directories" in str(exc_info.value)

    def test_path_too_long(self):
        """Test path length validation."""
        validator = PathSecurityValidator(max_path_length=10)
        with pytest.raises(FileAccessError) as exc_info:
            validator.validate("a" * 100)
        assert "exceeds maximum length" in str(exc_info.value)

    def test_null_byte_injection(self):
        """Test null byte injection detection."""
        validator = PathSecurityValidator()
        with pytest.raises(SecurityError):
            validator.validate("file\x00.txt")

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        validator = PathSecurityValidator()
        assert validator.sanitize_filename("test.lmp") == "test.lmp"
        # Path separators are replaced
        assert validator.sanitize_filename("dir/test.lmp") == "dir_test.lmp"
        assert validator.sanitize_filename("dir\\test.lmp") == "dir_test.lmp"
        # Null bytes are removed
        assert validator.sanitize_filename("file\x00.txt") == "file.txt"
        # Leading/trailing dots and spaces are stripped (both ends)
        assert validator.sanitize_filename("  .hidden.  ") == "hidden"

    def test_sanitize_empty_filename(self):
        """Test sanitization of empty filename."""
        validator = PathSecurityValidator()
        assert validator.sanitize_filename("") == "unnamed"
        assert validator.sanitize_filename("   ") == "unnamed"

    def test_sanitize_long_filename(self):
        """Test sanitization of very long filename."""
        validator = PathSecurityValidator()
        long_name = "a" * 300 + ".txt"
        sanitized = validator.sanitize_filename(long_name)
        assert len(sanitized) <= 255


class TestFileSizeValidator:
    """Test file size validator."""

    def test_valid_file_size(self):
        """Test valid file size."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            validator = FileSizeValidator(max_file_size=1024)
            size = validator.validate_file(temp_path)
            assert size == len(b"test content")
        finally:
            os.unlink(temp_path)

    def test_file_too_large(self):
        """Test file size exceeding limit."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 100)
            temp_path = f.name

        try:
            validator = FileSizeValidator(max_file_size=50)
            with pytest.raises(FileAccessError) as exc_info:
                validator.validate_file(temp_path)
            assert "exceeds maximum allowed" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test validation of non-existent file."""
        validator = FileSizeValidator()
        with pytest.raises(FileNotFoundError):
            validator.validate_file("/nonexistent/path/file.txt")

    def test_total_size_validation(self):
        """Test total size validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(3):
                path = Path(tmpdir) / f"file{i}.txt"
                path.write_text("x" * 200)  # 200 bytes each
                files.append(str(path))

            # 3 files × 200 = 600 bytes, limit is 500
            validator = FileSizeValidator(max_file_size=1000, max_total_size=500)
            with pytest.raises(FileAccessError) as exc_info:
                validator.validate_total_size(files)
            assert "Total file size" in str(exc_info.value)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_validate_filepath_default(self):
        """Test default filepath validation."""
        result = validate_filepath("test.lmp")
        assert isinstance(result, Path)

    def test_validate_filepath_custom_validator(self):
        """Test filepath validation with custom validator."""
        validator = PathSecurityValidator(allow_absolute_paths=True)
        result = validate_filepath("test.lmp", validator=validator)
        assert isinstance(result, Path)

    def test_validate_file_size_default(self):
        """Test default file size validation."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            size = validate_file_size(temp_path)
            assert size == 4
        finally:
            os.unlink(temp_path)


class TestDefaultValidators:
    """Test default validator instances."""

    def test_default_path_validator(self):
        """Test default path validator."""
        assert default_path_validator.allow_absolute_paths is False
        assert default_path_validator.allowed_base_dirs is None

    def test_default_size_validator(self):
        """Test default size validator."""
        assert default_size_validator.max_file_size == 100 * 1024 * 1024  # 100MB
        assert default_size_validator.max_total_size == 500 * 1024 * 1024  # 500MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
