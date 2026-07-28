"""How `picklikeme` is meant to be launched, and that it actually launches.

Three equivalent forms exist (see docs/analyzer.md "Entry points"):

  picklikeme <command> ...                 - the installed console script
  python -m picklikeme <command> ...        - always works, no PATH needed
  python -m picklikeme.analyzer ...          - older alias, `analyze` only

Only the middle form is guaranteed to work regardless of whether a console
script is on PATH or which of this project's two virtualenvs is active, which
is why every command this tool prints for the user to run next
(`picklikeme.config.cli_prefix`) uses it, naming `sys.executable` explicitly.

These tests exercise real subprocesses rather than just argument parsing,
because the regression this guards against was a documented command failing
outright when actually run - a parser-only test would have missed it.
"""

import json
import socket
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    """An ephemeral port nothing is listening on yet.

    `--port 0` cannot be used here: `serve()` computes the port as
    `args.port or DEFAULT_PORT`, and 0 is falsy, so it would silently fall
    back to the default port instead of asking the OS for a free one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ModuleEntryPointTests(unittest.TestCase):
    """`python -m picklikeme` must work from a bare interpreter invocation -
    no console script, no PATH - which is exactly what a package with no
    `__main__.py` cannot do."""

    def test_the_top_level_package_has_a_main_module(self):
        main_py = PROJECT_ROOT / "src" / "picklikeme" / "__main__.py"
        self.assertTrue(main_py.is_file(), "python -m picklikeme has no entry point")

    def test_module_invocation_lists_every_subcommand(self):
        result = subprocess.run(
            [sys.executable, "-m", "picklikeme", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("analyze", "annotate", "build-manifest", "review"):
            self.assertIn(command, result.stdout)

    def test_module_invocation_reaches_the_annotate_subcommand(self):
        result = subprocess.run(
            [sys.executable, "-m", "picklikeme", "annotate", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)
        self.assertIn("--port", result.stdout)

    def test_module_invocation_still_works_from_outside_the_project_directory(self):
        """A `python -m` command must not silently depend on cwd - the whole
        point is that it works from wherever the user happens to be."""
        with tempfile.TemporaryDirectory() as elsewhere:
            result = subprocess.run(
                [sys.executable, "-m", "picklikeme", "annotate", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=elsewhere,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--output", result.stdout)


class ConsoleScriptRegistrationTests(unittest.TestCase):
    """The installed console script is a thin wrapper over the same function
    `python -m picklikeme` calls, not a second implementation - guard that
    pyproject.toml still points it there."""

    def test_pyproject_registers_the_console_script(self):
        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('picklikeme = "picklikeme.ingest.cli:main"', text)

    def test_the_registered_target_exists_and_is_callable(self):
        from picklikeme.ingest.cli import main

        self.assertTrue(callable(main))

    def test_the_module_entry_point_delegates_to_the_same_function(self):
        """`python -m picklikeme` and the console script must run identical
        code, or a fix to one silently misses the other."""
        import picklikeme.__main__ as top_level_main
        from picklikeme.ingest.cli import main as ingest_main

        self.assertIs(top_level_main.main, ingest_main)


class PrintedInvocationTests(unittest.TestCase):
    """`cli_prefix()` is what every "run this next" message uses instead of a
    bare `picklikeme` - it must name a real interpreter and produce a command
    that `annotate`'s own parser actually accepts."""

    def test_cli_prefix_names_the_running_interpreter_explicitly(self):
        from picklikeme.config import cli_prefix

        prefix = cli_prefix()
        self.assertIn(sys.executable, prefix)
        self.assertIn("-m picklikeme", prefix)
        # Never the bare console-script form: that is exactly what silently
        # stops working when PATH doesn't include the active venv's Scripts.
        self.assertNotIn('"picklikeme"', prefix)

    def test_the_printed_annotate_command_parses_as_valid_arguments(self):
        from picklikeme.analyzer.cli import build_annotate_parser
        from picklikeme.config import cli_prefix

        printed = f'{cli_prefix()} annotate --output "some/dir"'
        # Drop the interpreter/module part the way a shell would after
        # dispatch, keeping only what argparse sees.
        tail = printed.split(" annotate ", 1)[1]
        args = shlex.split(tail)
        parsed = build_annotate_parser().parse_args(args)
        self.assertEqual(parsed.output, "some/dir")


class AnnotateServerLaunchTests(unittest.TestCase):
    """The end-to-end regression: launch the annotation server exactly the
    way `picklikeme analyze` tells a user to, and confirm it actually serves -
    not just that argument parsing succeeds."""

    def _report_dir(self, tmp: Path) -> Path:
        report_dir = tmp / "out"
        report_dir.mkdir(parents=True)
        (report_dir / "report.html").write_text("<html>report</html>", encoding="utf-8")
        return report_dir

    def _wait_for_health(self, port: int, proc: subprocess.Popen, timeout: float = 15.0):
        url = f"http://127.0.0.1:{port}/api/health"
        deadline = time.time() + timeout
        last_error = None
        while time.time() < deadline:
            if proc.poll() is not None:
                self.fail(f"annotate server exited early ({proc.returncode}):\n{proc.stdout.read()}")
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    return json.load(response)
            except (urllib.error.URLError, ConnectionError) as exc:
                last_error = exc
                time.sleep(0.2)
        self.fail(f"server never answered {url}: {last_error}")

    def _stop(self, proc: subprocess.Popen) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()

    def test_python_dash_m_picklikeme_annotate_launches_a_working_server(self):
        """The exact command `python -m picklikeme annotate --output ...`
        must start a server that answers /api/health - this is what "the
        annotation workflow cannot be launched" means in practice."""
        # `ignore_cleanup_errors`: on Windows, terminate() is a hard kill, not
        # a Ctrl+C - the child never runs its `finally: store.close()`, so the
        # OS can be slightly behind on releasing the sqlite file handle when
        # this block exits. That is a teardown-timing detail, not something
        # this test is about; the server having answered is what matters.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            report_dir = self._report_dir(root)
            port = _free_port()

            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "picklikeme", "annotate",
                    "--output", str(report_dir),
                    "--annotations-db", str(root / "kb.db"),
                    "--port", str(port),
                    "--no-browser",
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                body = self._wait_for_health(port, proc)
                self.assertTrue(body["ok"])
            finally:
                self._stop(proc)

    def test_the_exact_command_analyze_prints_launches_a_working_server(self):
        """Reproduces the reported bug literally: run `picklikeme analyze`,
        take the follow-up line it prints verbatim, execute it in a real
        shell (as a user copy-pasting it would), and confirm the server it
        starts actually answers - not a reconstruction of what the command
        *should* be."""
        from test_annotations import build_fn_dataset

        # `ignore_cleanup_errors`: on Windows, terminate() is a hard kill, not
        # a Ctrl+C - the child never runs its `finally: store.close()`, so the
        # OS can be slightly behind on releasing the sqlite file handle when
        # this block exits. That is a teardown-timing detail, not something
        # this test is about; the server having answered is what matters.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            ranking, selected, rejected = build_fn_dataset(root)

            analyze = subprocess.run(
                [
                    sys.executable, "-m", "picklikeme", "analyze",
                    "--ranking", str(ranking),
                    "--selected", str(selected),
                    "--rejected", str(rejected),
                    "--output", str(root / "out"),
                    "--annotations-db", str(root / "kb.db"),
                    "--no-charts", "--no-contact-sheets", "--quiet", "--no-serve",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(analyze.returncode, 0, analyze.stdout + analyze.stderr)
            line = next(
                (l for l in analyze.stdout.splitlines() if "annotate --output" in l), None
            )
            self.assertIsNotNone(line, "analyze did not print the annotate follow-up command")

            port = _free_port()
            proc = subprocess.Popen(
                line.strip() + f' --port {port} --no-browser',
                shell=True,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                body = self._wait_for_health(port, proc)
                self.assertTrue(body["ok"])
            finally:
                self._stop(proc)


if __name__ == "__main__":
    unittest.main()
