"""`python -m picklikeme <command> ...` - same entry point as the installed
`picklikeme` console script.

Exists so every subcommand (`analyze`, `annotate`, `build-manifest`, ...) is
reachable without depending on the `picklikeme` console script being on PATH -
which requires both an editable install (`pip install -e .`) *and* the right
virtualenv's Scripts/bin directory being active. `python -m picklikeme` only
needs the interpreter that runs it to have the package importable, which is
true of whatever interpreter is currently in use.
"""

from .ingest.cli import main

if __name__ == "__main__":
    main()
