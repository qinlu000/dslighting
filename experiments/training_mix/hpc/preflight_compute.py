"""Compute-node gate for Agent Lightning, CUDA, and Bubblewrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence


def _run_json(command: Sequence[str], program: str, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [*command, "-c", program, *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def _fingerprint(command: Sequence[str], *, trainer: bool) -> dict[str, Any]:
    packages = ["agentlightning"]
    if trainer:
        packages.extend(["torch", "verl", "vllm", "flash-attn"])
    else:
        packages.append("dslighting")
    program = """
import importlib
import importlib.metadata as metadata
import json
import sys
names = json.loads(sys.argv[1])
versions = {}
module_names = {
    "agentlightning": "agentlightning",
    "dslighting": "dslighting",
    "torch": "torch",
    "verl": "verl",
    "vllm": "vllm",
    "flash-attn": "flash_attn",
}
import_errors = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
    try:
        importlib.import_module(module_names[name])
    except Exception as exc:
        import_errors[name] = f"{type(exc).__name__}: {exc}"
payload = {"python": sys.version.split()[0], "packages": versions, "import_errors": import_errors}
if "torch" in names and versions["torch"] is not None:
    import torch
    payload["cuda_available"] = torch.cuda.is_available()
    payload["cuda_devices"] = torch.cuda.device_count()
    payload["torch_cuda"] = torch.version.cuda
print(json.dumps(payload))
"""
    payload = _run_json(command, program, json.dumps(packages))
    payload["command"] = list(command)
    payload["passed"] = bool(
        not any(value is None for value in payload["packages"].values())
        and not payload["import_errors"]
    )
    if trainer:
        payload["passed"] = bool(
            payload["passed"]
            and payload.get("cuda_available")
            and payload.get("cuda_devices", 0) > 0
        )
    return payload


def _bubblewrap_smoke(controller_python: Path, sandbox_python: Path) -> dict[str, Any]:
    program = """
import asyncio
import json
import sys
import tempfile
from dslighting.services.sandbox_backends.backends.base import SandboxBackendConfig
from dslighting.services.sandbox_backends.backends.local import LocalSandboxBackend

async def main():
    with tempfile.TemporaryDirectory(prefix="dslighting-bwrap-preflight-") as workspace:
        backend = LocalSandboxBackend(config=SandboxBackendConfig(
            timeout=30,
            isolation="bubblewrap",
            environment_policy="allowlist",
            network_policy="disabled",
            python_executable=sys.argv[1],
        ))
        try:
            result = await backend.execute(
                "import json, platform; print(json.dumps({'python': platform.python_version()}))",
                workspace,
            )
        finally:
            await backend.shutdown()
    print(json.dumps({
        "passed": result.success,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "metadata": result.metadata,
    }))

asyncio.run(main())
"""
    return _run_json([str(controller_python)], program, str(sandbox_python))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle(bundle_root: Path, *, verify_hashes: bool) -> dict[str, Any]:
    manifest_path = bundle_root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_manifest = bundle_root / "manifests" / "train_full.jsonl"
    failures: list[str] = []
    if _sha256(train_manifest) != manifest["manifest_sha256"]:
        failures.append("training manifest sha256 mismatch")
    if verify_hashes:
        for entry in manifest["entries"]:
            path = bundle_root / entry["path"]
            if not path.is_file():
                failures.append(f"missing: {entry['path']}")
            elif path.stat().st_size != entry["bytes"]:
                failures.append(f"size mismatch: {entry['path']}")
            elif _sha256(path) != entry["sha256"]:
                failures.append(f"sha256 mismatch: {entry['path']}")
            if len(failures) >= 20:
                break
    return {
        "passed": not failures,
        "root": str(bundle_root),
        "tasks": manifest["tasks"],
        "files": manifest["files"],
        "bytes": manifest["bytes"],
        "hashes_verified": verify_hashes,
        "failures": failures,
    }


def _model(model_root: Path) -> dict[str, Any]:
    model_root = model_root.expanduser().resolve()
    failures: list[str] = []
    config_path = model_root / "config.json"
    tokenizer_path = model_root / "tokenizer.json"
    for path in (config_path, tokenizer_path):
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty: {path.name}")

    weight_files: set[str] = set()
    index_path = model_root / "model.safetensors.index.json"
    single_weight = model_root / "model.safetensors"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files = {str(value) for value in index["weight_map"].values()}
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid model.safetensors.index.json: {exc}")
    elif single_weight.is_file():
        weight_files = {single_weight.name}
    else:
        failures.append("missing safetensors weights or index")
    for name in sorted(weight_files):
        path = model_root / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty model shard: {name}")

    model_type = None
    if config_path.is_file():
        try:
            model_type = json.loads(config_path.read_text(encoding="utf-8")).get(
                "model_type"
            )
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid config.json: {exc}")
    return {
        "passed": not failures,
        "root": str(model_root),
        "model_type": model_type,
        "weight_files": sorted(weight_files),
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--controller-python", type=Path, required=True)
    parser.add_argument("--trainer-python", type=Path, required=True)
    parser.add_argument("--trainer-image", type=Path)
    parser.add_argument("--container-bind", type=Path)
    parser.add_argument("--dare-python", type=Path, required=True)
    parser.add_argument("--general-python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verify-bundle-hashes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = {
        "controller_python": args.controller_python,
        "dare_python": args.dare_python,
        "general_python": args.general_python,
    }
    missing = [name for name, path in paths.items() if not path.expanduser().is_file()]
    if not args.trainer_python.expanduser().is_file():
        missing.append("trainer_python")
    if args.trainer_image and not args.trainer_image.expanduser().is_file():
        missing.append("trainer_image")
    if args.container_bind and not args.container_bind.expanduser().is_dir():
        missing.append("container_bind")
    if not args.model.expanduser().is_dir():
        missing.append("model")
    if missing:
        print(json.dumps({"status": "failed", "missing": missing}, indent=2))
        return 1

    try:
        trainer_command = [str(args.trainer_python)]
        if args.trainer_image:
            trainer_command = ["singularity", "exec", "--nv"]
            if args.container_bind:
                bind = str(args.container_bind.expanduser().resolve())
                trainer_command.extend(["--bind", f"{bind}:{bind}"])
            trainer_command.extend(
                [str(args.trainer_image.expanduser().resolve()), str(args.trainer_python)]
            )
        report = {
            "bundle": _bundle(
                args.bundle_root.expanduser().resolve(),
                verify_hashes=args.verify_bundle_hashes,
            ),
            "controller": _fingerprint([str(args.controller_python)], trainer=False),
            "trainer": _fingerprint(trainer_command, trainer=True),
            "model": _model(args.model),
            "bubblewrap": {
                "dare": _bubblewrap_smoke(args.controller_python, args.dare_python),
                "general": _bubblewrap_smoke(args.controller_python, args.general_python),
            },
        }
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1

    passed = all(
        (
            report["bundle"]["passed"],
            report["controller"]["passed"],
            report["trainer"]["passed"],
            report["model"]["passed"],
            report["bubblewrap"]["dare"]["passed"],
            report["bubblewrap"]["general"]["passed"],
        )
    )
    report["status"] = "passed" if passed else "failed"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
