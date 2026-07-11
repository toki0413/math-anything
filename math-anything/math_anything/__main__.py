"""Allow ``python -m math_anything`` as a package-level CLI invocation."""

import sys

from math_anything.cli import main

if __name__ == "__main__":
    sys.exit(main())
