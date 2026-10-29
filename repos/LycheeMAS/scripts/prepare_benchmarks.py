"""Prepare benchmark data and runtime resources for LycheeMAS benchmarks.

This command downloads project-owned benchmark data. It does not copy from other
users' workspaces. It also prepares benchmark-owned runtime resources such as
HumanEval/GAIA Docker images. Use --source modelscope or --source huggingface to
force one provider; the default auto mode uses each benchmark's preferred
provider order.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

DOCKER_IMAGE_SPECS = {
    "python_sandbox": {
        "image": "lychee-python-sandbox:local",
        "dockerfile": "docker/python-sandbox.Dockerfile",
        "deps": [],
    },
    "human_eval": {
        "image": "lychee-human-eval:local",
        "dockerfile": "docker/human_eval.Dockerfile",
        "deps": ["python_sandbox"],
    },
    "gaia": {
        "image": "lychee-gaia:local",
        "dockerfile": "docker/gaia.Dockerfile",
        "deps": ["python_sandbox"],
    },
}

DOCKER_BUILD_ORDER = ("python_sandbox", "human_eval", "gaia")


def _parse_prepare_targets(raw: list[str] | None, preparers: dict, *, all_full: bool, full_targets) -> list[str]:
    if all_full:
        return list(full_targets)
    if not raw:
        return sorted(preparers)
    targets: list[str] = []
    for item in raw:
        targets.extend(part.strip() for part in item.split(",") if part.strip())
    return targets


def _print_names(names) -> None:
    print("\n".join(sorted(names)))


def _print_structure(rows, aliases: dict[str, str]) -> None:
    for row in rows:
        print(f"Benchmark source: {row['benchmark_source']}")
        print(f"  full prepare target: {row['full_prepare_target']}")
        print("  prepare target -> runnable task(s) -> kind/scorer(s)")
        for mapping in row.get("mappings", []):
            prepare_target = mapping["prepare_target"]
            actual = aliases.get(prepare_target)
            alias_suffix = f" (alias -> {actual})" if actual else ""
            tasks = ", ".join(mapping.get("runnable_tasks") or ["-"])
            kinds = ", ".join(mapping.get("kinds") or ["-"])
            note = mapping.get("note")
            note_suffix = f" [{note}]" if note else ""
            print(f"    {prepare_target}{alias_suffix} -> {tasks} -> {kinds}{note_suffix}")
        print()


def _format_list(values) -> str:
    return ", ".join(str(value) for value in values if value) or "-"


def _print_download_sources(catalog: dict) -> None:
    for name, entry in catalog.items():
        print(f"Benchmark source: {entry.get('benchmark_source', name)} ({name})")
        for provider in ("modelscope", "huggingface"):
            section = entry.get(provider, {})
            env_name = section.get("env") or "-"
            defaults = _format_list(section.get("default_ids") or [])
            print(f"  {provider}: env={env_name}; default_ids={defaults}")
        others = entry.get("other_defaults") or []
        if others:
            print("  other defaults:")
            for item in others:
                print(f"    {item.get('provider', '-')}: {item.get('id', '-')} ({item.get('purpose', '-')})")
        else:
            print("  other defaults: -")
        fallbacks = entry.get("fallback_files") or []
        if fallbacks:
            print("  direct-file fallback:")
            for item in fallbacks:
                files = _format_list(item.get("files") or item.get("patterns") or [])
                strict = item.get("strict", "-")
                print(f"    {item.get('provider', '-')}: {files}; strict={strict}; {item.get('purpose', '-')}")
        else:
            print("  direct-file fallback: -")
        print()


def _docker_image_exists(image: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SystemExit("Docker CLI not found. Install Docker or use --docker-images never.") from exc
    return result.returncode == 0


def _docker_targets_for_prepare_targets(targets: list[str], aliases: dict[str, str]) -> set[str]:
    docker_targets: set[str] = set()
    for target in targets:
        canonical = aliases.get(target, target)
        names = {target, canonical}
        if "human_eval" in names:
            docker_targets.add("human_eval")
        if any(name == "gaia" or name.startswith("gaia_") for name in names):
            docker_targets.add("gaia")
    return docker_targets


def _expand_docker_targets(targets: set[str]) -> list[str]:
    expanded = set(targets)
    pending = list(targets)
    while pending:
        target = pending.pop()
        for dep in DOCKER_IMAGE_SPECS[target]["deps"]:
            if dep not in expanded:
                expanded.add(dep)
                pending.append(dep)
    return [name for name in DOCKER_BUILD_ORDER if name in expanded]


def _build_docker_images(
    targets: set[str],
    *,
    apt_mirror: str | None,
    pip_index_url: str,
    no_cache: bool,
    pull_base: bool,
    force: bool,
) -> None:
    for name in _expand_docker_targets(targets):
        spec = DOCKER_IMAGE_SPECS[name]
        image = spec["image"]
        dockerfile = spec["dockerfile"]
        if not force and _docker_image_exists(image):
            print(f"[prepare:docker] {image} already exists", flush=True)
            continue

        cmd = ["docker", "build", "--network=host"]
        if no_cache:
            cmd.append("--no-cache")
        if pull_base and name == "python_sandbox":
            cmd.append("--pull")
        if name in {"python_sandbox", "gaia"}:
            cmd.extend(["--build-arg", f"PIP_INDEX_URL={pip_index_url}"])
        if name == "python_sandbox" and apt_mirror:
            cmd.extend(["--build-arg", f"APT_MIRROR={apt_mirror}"])
        cmd.extend(["-f", dockerfile, "-t", image, "."])

        print(f"[prepare:docker] building {image} from {dockerfile}", flush=True)
        try:
            subprocess.run(cmd, cwd=_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"[prepare:docker] failed to build {image}") from exc

    for name in _expand_docker_targets(targets):
        image = DOCKER_IMAGE_SPECS[name]["image"]
        if not _docker_image_exists(image):
            raise SystemExit(f"[prepare:docker] image missing after build: {image}")
        print(f"[prepare:docker] {image} ready", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LycheeMAS benchmark data and runtime resources")
    parser.add_argument(
        "--tasks",
        nargs="*",
        help=(
            "Prepare target names, comma-separated or space-separated. "
            "Use --list-prepare-targets to inspect accepted names."
        ),
    )
    parser.add_argument(
        "--all-full-benchmarks",
        action="store_true",
        help="Prepare one full/canonical prepare target for every benchmark source.",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Set LYCHEE_BENCHMARK_RAW_ROOT for upstream/raw benchmark files",
    )
    parser.add_argument(
        "--prepared-root",
        default=None,
        help="Set LYCHEE_BENCHMARK_PREPARED_ROOT for loader-ready benchmark files",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "huggingface"),
        default="auto",
        help=(
            "Download provider. auto uses benchmark-specific order: confirmed "
            "ModelScope mirrors first, otherwise HuggingFace."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if target exists")
    parser.add_argument(
        "--docker-images",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Prepare benchmark Docker images. auto builds images required by selected targets "
            "(human_eval/gaia); always builds all official benchmark images; never skips Docker."
        ),
    )
    parser.add_argument(
        "--force-docker-images",
        action="store_true",
        help="Rebuild selected benchmark Docker images even if the local image already exists.",
    )
    parser.add_argument(
        "--docker-apt-mirror",
        default=os.environ.get("LYCHEE_DOCKER_APT_MIRROR"),
        help="Ubuntu apt mirror used while building benchmark Docker images.",
    )
    parser.add_argument(
        "--docker-pip-index-url",
        default=os.environ.get("LYCHEE_DOCKER_PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple"),
        help="Python package index used while building benchmark Docker images.",
    )
    parser.add_argument(
        "--docker-no-cache",
        action="store_true",
        help="Build benchmark Docker images without Docker layer cache.",
    )
    parser.add_argument(
        "--docker-pull-base",
        action="store_true",
        help="Ask Docker to pull a fresh ubuntu:22.04 base when building the sandbox image.",
    )
    parser.add_argument(
        "--list-benchmark-sources",
        action="store_true",
        help="List canonical benchmark sources and exit",
    )
    parser.add_argument(
        "--list-prepare-targets",
        action="store_true",
        help="List every name accepted by --tasks, including source names and aliases, and exit",
    )
    parser.add_argument(
        "--list-runnable-tasks",
        action="store_true",
        help="List runnable benchmark tasks accepted by run_mas.py --task and exit",
    )
    parser.add_argument(
        "--list-benchmark-structure",
        action="store_true",
        help="Show benchmark source with prepare target -> runnable task(s) -> kind/scorer mapping and exit",
    )
    parser.add_argument(
        "--list-download-sources",
        action="store_true",
        help="Show ModelScope/HuggingFace/other source candidates and direct-file fallback rules",
    )
    args = parser.parse_args()

    if args.raw_root:
        os.environ["LYCHEE_BENCHMARK_RAW_ROOT"] = args.raw_root
    if args.prepared_root:
        os.environ["LYCHEE_BENCHMARK_PREPARED_ROOT"] = args.prepared_root
    os.environ["LYCHEE_DATA_SOURCE"] = args.source
    if args.all_full_benchmarks:
        os.environ.setdefault("LYCHEE_MAST_DOWNLOAD_FULL", "1")

    from lychee_mas.eval.benchmarks import (
        BENCHMARK_STRUCTURE,
        FULL_PREPARE_TARGETS,
        LOADERS,
        PREPARE_ALIASES,
        PREPARERS,
        prepare,
    )
    from lychee_mas.eval.benchmarks.common import prepared_root, raw_root
    from lychee_mas.eval.benchmarks.source_catalog import source_catalog

    if args.list_benchmark_sources:
        _print_names(row["benchmark_source"] for row in BENCHMARK_STRUCTURE)
        return
    if args.list_prepare_targets:
        _print_names(PREPARERS)
        return
    if args.list_runnable_tasks:
        _print_names(LOADERS)
        return
    if args.list_benchmark_structure:
        _print_structure(BENCHMARK_STRUCTURE, PREPARE_ALIASES)
        return
    if args.list_download_sources:
        _print_download_sources(source_catalog())
        return

    targets = _parse_prepare_targets(
        args.tasks, PREPARERS, all_full=args.all_full_benchmarks, full_targets=FULL_PREPARE_TARGETS
    )

    print(f"[prepare] raw_root={raw_root()}")
    print(f"[prepare] prepared_root={prepared_root()}")
    print(f"[prepare] source={args.source}")
    if args.all_full_benchmarks:
        print(f"[prepare] all_full_benchmarks={','.join(FULL_PREPARE_TARGETS)}")
    for target in targets:
        print(f"[prepare] {target} ...", flush=True)
        try:
            location = prepare(target, force=args.force, source=args.source)
        except Exception as exc:
            raise SystemExit(f"[prepare] failed for {target}: {exc}") from exc
        print(f"[prepare] {target} ready at {location}", flush=True)

    if args.docker_images == "never":
        docker_targets: set[str] = set()
    elif args.docker_images == "always":
        docker_targets = {"human_eval", "gaia"}
    else:
        docker_targets = _docker_targets_for_prepare_targets(targets, PREPARE_ALIASES)
    if docker_targets:
        selected = ",".join(_expand_docker_targets(docker_targets))
        print(f"[prepare:docker] selected={selected}", flush=True)
        _build_docker_images(
            docker_targets,
            apt_mirror=args.docker_apt_mirror,
            pip_index_url=args.docker_pip_index_url,
            no_cache=args.docker_no_cache,
            pull_base=args.docker_pull_base,
            force=args.force_docker_images,
        )


if __name__ == "__main__":
    main()
