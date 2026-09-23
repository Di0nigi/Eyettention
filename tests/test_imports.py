import ast
import sys
import unittest
from pathlib import Path


EYETTENTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EYETTENTION_ROOT.parent
EXCLUDED_DIRS = {"__pycache__", ".git", "tests"}
EXPECTED_IMPORT_FAILURES = {
    ("Eyettention/app.py", "import gradio as gr"),
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class EyettentionImportTests(unittest.TestCase):
    def test_import_lines(self):
        failures = []

        for file_path in _python_files(EYETTENTION_ROOT):
            source = file_path.read_text()
            tree = ast.parse(source, filename=str(file_path))

            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    continue

                import_line = ast.get_source_segment(source, node)
                try:
                    exec(
                        compile(import_line, str(file_path), "exec"),
                        _import_globals(file_path),
                    )
                except Exception as exc:
                    relative_path = str(file_path.relative_to(PROJECT_ROOT))
                    if _is_expected_failure(relative_path, import_line):
                        continue
                    failures.append(
                        f"{relative_path}:{node.lineno}\n"
                        f"{import_line}\n"
                        f"{type(exc).__name__}: {exc}"
                    )

        if failures:
            self.fail("Failed import line(s):\n\n" + "\n\n".join(failures))


def _python_files(root):
    for file_path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in file_path.parts):
            continue
        yield file_path


def _import_globals(file_path):
    module_path = file_path.relative_to(PROJECT_ROOT).with_suffix("")
    module_parts = module_path.parts

    if module_parts[-1] == "__init__":
        module_name = ".".join(module_parts[:-1])
        package = module_name
    else:
        module_name = ".".join(module_parts)
        package = ".".join(module_parts[:-1])

    return {
        "__name__": module_name,
        "__package__": package,
    }


def _is_expected_failure(relative_path, import_line):
    return any(
        relative_path == expected_path and import_line.startswith(expected_import)
        for expected_path, expected_import in EXPECTED_IMPORT_FAILURES
    )


if __name__ == "__main__":
    unittest.main()
