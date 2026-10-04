"""`python -m aitextjury` — start the local API server, or use the CLI.

Server:  python -m aitextjury            (default http://127.0.0.1:8000)
CLI:     python -m aitextjury.cli <command>
"""
from .main import run

if __name__ == "__main__":  # pragma: no cover
    run()
