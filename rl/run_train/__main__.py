"""`python -m rl.run_train {start,status,stop,export,final-test}` (DESIGN.md 7)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
