"""`python -m zeroaibench` — start the local API server, or use the CLI.

Server:  python -m zeroaibench            (default http://127.0.0.1:8000)
CLI:     python -m zeroaibench.cli <command>
"""
from .main import run

if __name__ == "__main__":  # pragma: no cover
    run()
