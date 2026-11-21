"""Allow ``python -m app`` to behave like ``wan22-s2v``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
