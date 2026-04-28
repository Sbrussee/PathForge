from pathlib import Path
import ast


def test_optional_backend_direct_imports_are_confined_to_adapters_and_optional_guards():
    src_root = Path("src/pathbench")
    allowed_prefixes = {
        Path("src/pathbench/adapters"),
        Path("src/pathbench/utils/optional"),
    }
    offenders = []

    for path in src_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        direct_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                direct_import = any(alias.name in {"torchmil", "torchmetrics", "torchsurv"} for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                direct_import = node.module in {"torchmil", "torchmetrics", "torchsurv"} or (
                    node.module is not None
                    and any(node.module.startswith(f"{name}.") for name in ("torchmil", "torchmetrics", "torchsurv"))
                )
            if direct_import:
                break
        if direct_import is False:
            continue
        if not any(path.is_relative_to(prefix) for prefix in allowed_prefixes):
            offenders.append(str(path))

    assert offenders == []


def test_torchmil_backend_is_not_registered_by_import_side_effect():
    backend_path = Path("src/pathbench/adapters/torchmil/backend.py")
    text = backend_path.read_text(encoding="utf-8")

    assert '@MODELS.register("torchmil")' not in text
    assert "def register_torchmil_backend" in text


def test_training_layer_uses_adapter_collate_not_torchmil_package():
    lightning_path = Path("src/pathbench/training/lightning.py")
    tree = ast.parse(lightning_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "pathbench.adapters.torchmil.collate" in imported_modules
    assert all(not module.startswith("torchmil") for module in imported_modules)
