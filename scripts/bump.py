"""Repository maintenance tasks for bumping application and Parquet schema versions."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_REPO_URL: Final[str] = "https://github.com/matthdsm/ccnv"

_PEP_TO_SEMVER_TAG: Final[dict[str, str]] = {
    "a": "alpha",
    "b": "beta",
    "rc": "rc",
    "dev": "dev",
}

_APP_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:(?P<pep_tag>a|b|rc|dev)(?P<pep_num>\d*)|-(?P<sem_tag>[0-9A-Za-z.-]+))?$"
)
_STRICT_SEMVER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)
_GFM_ALPHA_BANNER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^> \[!NOTE\]\n"
    r"> `ccnv` is in active pre-1\.0 alpha development \(`[^`]+`\)\. "
    r"CLI interfaces, data models, and schemas may evolve before the initial stable release\.\n+"
)
_ZENSICAL_ALPHA_BANNER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^!!! note\n\n"
    r"    `ccnv` is in active pre-1\.0 alpha development \(`[^`]+`\)\. "
    r"CLI interfaces, data models, and schemas may evolve before the initial stable release\.\n+"
)


@dataclass(frozen=True, slots=True)
class ParsedAppVersion:
    """Normalized PEP 440 / SemVer application version components."""

    raw: str
    major: int
    minor: int
    patch: int
    cargo_version: str
    is_dev: bool


def _parse_app_version(value: str) -> ParsedAppVersion:
    cleaned = value.strip().removeprefix("v")
    match = _APP_VERSION_RE.match(cleaned)
    if match is None:
        raise ValueError(
            f"invalid application version {value!r}; expected X.Y.Z (optional a/b/rc/dev suffix)"
        )
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    pep_tag = match.group("pep_tag")
    pep_num = match.group("pep_num") or ""
    sem_tag = match.group("sem_tag")

    if sem_tag is not None:
        if any(not part for part in sem_tag.split(".")):
            raise ValueError(f"invalid pre-release identifier in {value!r}")
        is_dev = sem_tag == "dev" or sem_tag.startswith("dev.")
        cargo_ver = f"{major}.{minor}.{patch}-{sem_tag}"
    elif pep_tag is not None:
        mapped = _PEP_TO_SEMVER_TAG[pep_tag]
        is_dev = pep_tag == "dev"
        suffix = f"{mapped}.{int(pep_num)}" if pep_num else mapped
        cargo_ver = f"{major}.{minor}.{patch}-{suffix}"
    else:
        is_dev = False
        cargo_ver = f"{major}.{minor}.{patch}"

    return ParsedAppVersion(
        raw=cleaned,
        major=major,
        minor=minor,
        patch=patch,
        cargo_version=cargo_ver,
        is_dev=is_dev,
    )


def _semver_tuple(version: str) -> tuple[int, int, int]:
    match = _STRICT_SEMVER_RE.match(version)
    if match is None:
        raise ValueError(
            f"invalid schema SemVer {version!r}; expected strict MAJOR.MINOR.PATCH"
        )
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _write_text_atomic(path: Path, content: str) -> None:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _commit_planned_writes(writes: dict[Path, str]) -> list[Path]:
    changed: list[Path] = []
    for path, new_text in writes.items():
        old_text = path.read_text(encoding="utf-8")
        if new_text != old_text:
            _write_text_atomic(path, new_text)
            changed.append(path)
    return changed


def _replace_section_key(
    toml_text: str, section_header: str, key: str, new_val: str, label: str
) -> str:
    sec_re = re.compile(rf"(?ms)(^\[{re.escape(section_header)}\]\n)(.*?)(?=^\[|\Z)")
    sec_match = sec_re.search(toml_text)
    if sec_match is None:
        raise ValueError(f"missing [{section_header}] section in {label}")
    body = sec_match.group(2)
    key_re = re.compile(rf'(?m)^({re.escape(key)}\s*=\s*)"[^"]*"')
    updated_body, count = key_re.subn(rf'\g<1>"{new_val}"', body, count=1)
    if count != 1:
        raise ValueError(f"missing {key} under [{section_header}] in {label}")
    return (
        toml_text[: sec_match.start(2)] + updated_body + toml_text[sec_match.end(2) :]
    )


def _resolve_repo_url(repo_root: Path) -> str:
    zensical_path = repo_root / "zensical.toml"
    if zensical_path.is_file():
        cfg = tomllib.loads(zensical_path.read_text(encoding="utf-8"))
        proj = cfg.get("project")
        if isinstance(proj, dict):
            url = proj.get("repo_url")
            if isinstance(url, str) and url.strip():
                return url.strip().rstrip("/")
    return DEFAULT_REPO_URL


def _update_alpha_banner(text: str, parsed: ParsedAppVersion, *, zensical: bool) -> str:
    if parsed.major >= 1:
        pattern = _ZENSICAL_ALPHA_BANNER_RE if zensical else _GFM_ALPHA_BANNER_RE
        updated = pattern.sub("", text)
        return re.sub(r"\n{3,}", "\n\n", updated)
    return re.sub(
        r"(active pre-1\.0 alpha development \(`)[^`]+(`\))",
        rf"\g<1>{parsed.raw}\g<2>",
        text,
    )


def _finalize_changelog(
    changelog_text: str,
    parsed: ParsedAppVersion,
    *,
    release_date: str,
    repo_url: str,
    allow_empty: bool,
) -> str:
    text = _update_alpha_banner(changelog_text, parsed, zensical=False)

    ref_link_re = re.compile(r"(?m)^\[[^\]]+\]:\s+https?://\S+\s*$")
    first_ref = ref_link_re.search(text)
    main_text = (
        text[: first_ref.start()].rstrip() if first_ref is not None else text.rstrip()
    )

    unrel_re = re.compile(r"(?m)^## \[Unreleased\]\s*$")
    unrel_match = unrel_re.search(main_text)

    if parsed.is_dev:
        if unrel_match is None:
            first_rel = re.search(r"(?m)^## \[", main_text)
            if first_rel is not None:
                main_text = (
                    main_text[: first_rel.start()].rstrip()
                    + "\n\n## [Unreleased]\n\n"
                    + main_text[first_rel.start() :]
                )
            else:
                main_text = main_text + "\n\n## [Unreleased]"
    else:
        existing_rel = re.search(
            rf"(?m)^## \[{re.escape(parsed.raw)}\](?:\s|$)", main_text
        )
        if existing_rel is not None:
            raise ValueError(
                f"CHANGELOG.md already contains release section ## [{parsed.raw}]"
            )
        if unrel_match is None:
            raise ValueError("CHANGELOG.md is missing '## [Unreleased]' section")

        after_unrel = main_text[unrel_match.end() :]
        next_sec = re.search(r"(?m)^## \[", after_unrel)
        if next_sec is not None:
            unrel_body = after_unrel[: next_sec.start()].strip()
            older_releases = after_unrel[next_sec.start() :].strip()
        else:
            unrel_body = after_unrel.strip()
            older_releases = ""

        if not unrel_body and not allow_empty:
            raise ValueError(
                "CHANGELOG.md [Unreleased] section has no entries to finalize; "
                "add release notes or pass --allow-empty"
            )

        prefix = main_text[: unrel_match.start()].rstrip()
        parts = [prefix, "## [Unreleased]", f"## [{parsed.raw}] - {release_date}"]
        if unrel_body:
            parts.append(unrel_body)
        if older_releases:
            parts.append(older_releases)
        main_text = "\n\n".join(parts)

    releases = [
        m.group(1)
        for m in re.finditer(r"(?m)^## \[(?!Unreleased\])([^\]]+)\]", main_text)
    ]
    if releases:
        link_lines = [f"[Unreleased]: {repo_url}/compare/v{releases[0]}...HEAD"]
        for idx, rel_ver in enumerate(releases):
            if idx + 1 < len(releases):
                prev_ver = releases[idx + 1]
                link_lines.append(
                    f"[{rel_ver}]: {repo_url}/compare/v{prev_ver}...v{rel_ver}"
                )
            else:
                link_lines.append(f"[{rel_ver}]: {repo_url}/releases/tag/v{rel_ver}")
        return main_text.rstrip() + "\n\n" + "\n".join(link_lines) + "\n"

    return main_text.rstrip() + "\n"


def bump_version(
    repo_root: Path,
    new_version: str,
    *,
    release_date: str | None = None,
    allow_empty: bool = False,
) -> tuple[str, list[Path]]:
    """Bump application version across Python, Rust, docs, and finalize CHANGELOG.md."""
    parsed = _parse_app_version(new_version)
    resolved_date = (
        dt.date.today().isoformat() if release_date is None else release_date.strip()
    )
    try:
        dt.date.fromisoformat(resolved_date)
    except ValueError as err:
        raise ValueError(
            f"invalid release date {resolved_date!r}; expected YYYY-MM-DD"
        ) from err

    pyproject_path = repo_root / "pyproject.toml"
    init_path = repo_root / "src" / "ccnv" / "__init__.py"
    cargo_toml_path = repo_root / "Cargo.toml"
    cargo_lock_path = repo_root / "Cargo.lock"
    changelog_path = repo_root / "CHANGELOG.md"
    readme_path = repo_root / "README.md"
    docs_index_path = repo_root / "docs" / "index.md"
    galaxy_macros_path = repo_root / "galaxy" / "macros.xml"

    for req in (pyproject_path, init_path, cargo_toml_path, changelog_path):
        if not req.is_file():
            raise FileNotFoundError(f"required repository file not found: {req}")

    writes: dict[Path, str] = {}
    writes[pyproject_path] = _replace_section_key(
        pyproject_path.read_text(encoding="utf-8"),
        "project",
        "version",
        parsed.raw,
        "pyproject.toml",
    )

    init_text = init_path.read_text(encoding="utf-8")
    new_init, init_count = re.subn(
        r'(?m)^(__version__\s*=\s*)"[^"]*"',
        rf'\g<1>"{parsed.raw}"',
        init_text,
        count=1,
    )
    if init_count != 1:
        raise ValueError("missing __version__ assignment in src/ccnv/__init__.py")
    writes[init_path] = new_init

    writes[cargo_toml_path] = _replace_section_key(
        cargo_toml_path.read_text(encoding="utf-8"),
        "package",
        "version",
        parsed.cargo_version,
        "Cargo.toml",
    )

    if cargo_lock_path.is_file():
        lock_text = cargo_lock_path.read_text(encoding="utf-8")
        new_lock, lock_count = re.subn(
            r'(name = "ccnv-core"\r?\nversion = )"[^"]*"',
            rf'\g<1>"{parsed.cargo_version}"',
            lock_text,
            count=1,
        )
        if lock_count != 1:
            raise ValueError("missing ccnv-core package entry in Cargo.lock")
        writes[cargo_lock_path] = new_lock

    if readme_path.is_file():
        writes[readme_path] = _update_alpha_banner(
            readme_path.read_text(encoding="utf-8"), parsed, zensical=False
        )
    if docs_index_path.is_file():
        writes[docs_index_path] = _update_alpha_banner(
            docs_index_path.read_text(encoding="utf-8"), parsed, zensical=True
        )
    if galaxy_macros_path.is_file():
        macros_text = galaxy_macros_path.read_text(encoding="utf-8")
        new_macros, macros_count = re.subn(
            r'(<token\s+name="@TOOL_VERSION@">)[^<]*(</token>)',
            rf"\g<1>{parsed.raw}\g<2>",
            macros_text,
            count=1,
        )
        if macros_count != 1:
            raise ValueError("missing @TOOL_VERSION@ token in galaxy/macros.xml")
        writes[galaxy_macros_path] = new_macros

    repo_url = _resolve_repo_url(repo_root)
    writes[changelog_path] = _finalize_changelog(
        changelog_path.read_text(encoding="utf-8"),
        parsed,
        release_date=resolved_date,
        repo_url=repo_url,
        allow_empty=allow_empty,
    )

    return parsed.raw, _commit_planned_writes(writes)


def _rewrite_schema_prose(
    text: str,
    old_ver: str,
    new_ver: str,
    set_brace_str: str,
) -> str:
    old_esc = re.escape(old_ver)
    replacements: tuple[tuple[str, str], ...] = (
        (rf"\bSchema v{old_esc}\b", f"Schema v{new_ver}"),
        (rf"\bSchema {old_esc}\b", f"Schema {new_ver}"),
        (rf"\bSchema `{old_esc}`", f"Schema `{new_ver}`"),
        (rf"(?i)(schema version `){old_esc}(`)", rf"\g<1>{new_ver}\g<2>"),
        (rf"\bschema {old_esc}\b", f"schema {new_ver}"),
        (rf"\bschema \(`{old_esc}`\)", f"schema (`{new_ver}`)"),
        (rf"\bSAMPLE_SCHEMA {old_esc}\b", f"SAMPLE_SCHEMA {new_ver}"),
        (rf"\bREFERENCE_SCHEMA {old_esc}\b", f"REFERENCE_SCHEMA {new_ver}"),
        (
            rf"Apache Arrow/Parquet \(`{old_esc}`\)",
            f"Apache Arrow/Parquet (`{new_ver}`)",
        ),
        (rf"Arrow `{old_esc}` schemas", f"Arrow `{new_ver}` schemas"),
        (rf'SCHEMA_VERSION = "{old_esc}"', f'SCHEMA_VERSION = "{new_ver}"'),
        (
            r"SUPPORTED_SCHEMA_VERSIONS = \{[^}]+\}",
            f"SUPPORTED_SCHEMA_VERSIONS = {set_brace_str}",
        ),
    )
    updated = text
    for pattern, repl in replacements:
        updated = re.sub(pattern, repl, updated)
    return updated


def bump_schema(
    repo_root: Path,
    target: str,
    *,
    keep_previous: bool = False,
) -> tuple[str, list[Path]]:
    """Bump Parquet SCHEMA_VERSION across models, docstrings, docs, and contract tests."""
    metadata_path = repo_root / "src" / "ccnv" / "models" / "metadata.py"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata module not found: {metadata_path}")

    meta_text = metadata_path.read_text(encoding="utf-8")
    ver_match = re.search(
        r'(?m)^SCHEMA_VERSION:\s*Final\[str\]\s*=\s*"(?P<ver>[^"]+)"', meta_text
    )
    sup_match = re.search(
        r"(?m)^SUPPORTED_SCHEMA_VERSIONS:\s*Final\[frozenset\[str\]\]\s*=\s*"
        r"frozenset\(\s*\{(?P<items>[^}]*)\}\s*\)",
        meta_text,
    )
    if ver_match is None or sup_match is None:
        raise ValueError(
            "failed to locate SCHEMA_VERSION / SUPPORTED_SCHEMA_VERSIONS in metadata.py"
        )

    old_ver = ver_match.group("ver")
    maj, min_, pat = _semver_tuple(old_ver)
    cleaned_target = target.strip()
    if cleaned_target == "major":
        new_ver = f"{maj + 1}.0.0"
    elif cleaned_target == "minor":
        new_ver = f"{maj}.{min_ + 1}.0"
    elif cleaned_target == "patch":
        new_ver = f"{maj}.{min_}.{pat + 1}"
    else:
        explicit = cleaned_target.removeprefix("v")
        _semver_tuple(explicit)
        new_ver = explicit

    if new_ver == old_ver:
        raise ValueError(f"Parquet schema version is already {old_ver}")

    existing_supported = {
        m.group(1) for m in re.finditer(r'"([^"]+)"', sup_match.group("items"))
    }
    supported_list = (
        sorted(existing_supported | {new_ver}, key=_semver_tuple)
        if keep_previous
        else [new_ver]
    )
    items_literal = ", ".join(f'"{v}"' for v in supported_list)
    set_brace_str = "{" + items_literal + "}"
    frozenset_str = f"frozenset({set_brace_str})"

    writes: dict[Path, str] = {}
    new_meta = re.sub(
        r'(?m)^(SCHEMA_VERSION:\s*Final\[str\]\s*=\s*)"[^"]+"',
        rf'\g<1>"{new_ver}"',
        meta_text,
        count=1,
    )
    new_meta = re.sub(
        r"(?m)^(SUPPORTED_SCHEMA_VERSIONS:\s*Final\[frozenset\[str\]\]\s*=\s*)"
        r"frozenset\(\s*\{[^}]*\}\s*\)",
        rf"\g<1>{frozenset_str}",
        new_meta,
        count=1,
    )
    writes[metadata_path] = _rewrite_schema_prose(
        new_meta, old_ver, new_ver, set_brace_str
    )

    scan_paths: list[Path] = []
    for sub in ("src", "tests", "docs"):
        base = repo_root / sub
        if base.is_dir():
            scan_paths.extend(sorted(base.rglob("*.py")))
            scan_paths.extend(sorted(base.rglob("*.md")))
    for root_doc in ("AGENTS.md", "README.md"):
        p = repo_root / root_doc
        if p.is_file():
            scan_paths.append(p)

    old_esc = re.escape(old_ver)
    self_test_path = repo_root / "tests" / "test_release_tasks.py"
    for path in scan_paths:
        if path in {metadata_path, self_test_path}:
            continue
        if any(part in {"superpowers", "designs"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        updated = _rewrite_schema_prose(text, old_ver, new_ver, set_brace_str)
        if path.parent.name == "tests" or "tests" in path.parts:
            updated = re.sub(
                rf'(SCHEMA_VERSION,\s*)"{old_esc}"',
                rf'\g<1>"{new_ver}"',
                updated,
            )
            updated = re.sub(
                r"(SUPPORTED_SCHEMA_VERSIONS,\s*)frozenset\(\s*\{[^}]*\}\s*\)",
                rf"\g<1>{frozenset_str}",
                updated,
            )
            updated = re.sub(
                rf'("schema_version"\]\s*,\s*)"{old_esc}"',
                rf'\g<1>"{new_ver}"',
                updated,
            )
            updated = re.sub(
                rf'("schema_version"\s*:\s*)"{old_esc}"',
                rf'\g<1>"{new_ver}"',
                updated,
            )
        if updated != text:
            writes[path] = updated

    changelog_path = repo_root / "CHANGELOG.md"
    if changelog_path.is_file():
        cl_text = changelog_path.read_text(encoding="utf-8")
        unrel_match = re.search(r"(?m)^## \[Unreleased\]\s*$", cl_text)
        if unrel_match is not None:
            after_start = unrel_match.end()
            next_stop = re.search(
                r"(?m)^(?:## \[|\[[^\]]+\]:\s+https?://)", cl_text[after_start:]
            )
            unrel_end = (
                after_start + next_stop.start()
                if next_stop is not None
                else len(cl_text)
            )
            updated_unrel = _rewrite_schema_prose(
                cl_text[after_start:unrel_end], old_ver, new_ver, set_brace_str
            )
            new_cl = cl_text[:after_start] + updated_unrel + cl_text[unrel_end:]
            if new_cl != cl_text:
                writes[changelog_path] = new_cl

    return new_ver, _commit_planned_writes(writes)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for bump-version and bump-schema subcommands."""
    parser = argparse.ArgumentParser(
        prog="bump.py",
        description="Bump ccnv application version or Parquet schema version.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root directory (defaults to parent of scripts/).",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    ver_p = sub.add_parser(
        "version",
        help="Bump application version across the repository and finalize CHANGELOG.md.",
    )
    ver_p.add_argument(
        "version",
        help="Target application version (e.g. 0.1.0, 0.2.0dev, 1.0.0).",
    )
    ver_p.add_argument(
        "--date",
        default=None,
        help="Release date in YYYY-MM-DD format (defaults to today).",
    )
    ver_p.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow finalizing CHANGELOG.md even when [Unreleased] has no entries.",
    )

    sch_p = sub.add_parser(
        "schema",
        help="Bump Parquet SCHEMA_VERSION across models, docs, and contract tests.",
    )
    sch_p.add_argument(
        "target",
        help="Target SemVer (e.g. 3.0.0) or increment level: major | minor | patch.",
    )
    sch_p.add_argument(
        "--keep-previous",
        action="store_true",
        help="Retain existing SUPPORTED_SCHEMA_VERSIONS alongside the new version.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning POSIX exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = Path(args.root).resolve()
    try:
        if args.subcommand == "version":
            normalized_ver, changed = bump_version(
                repo_root,
                args.version,
                release_date=args.date,
                allow_empty=bool(args.allow_empty),
            )
            rel_list = ", ".join(p.relative_to(repo_root).as_posix() for p in changed)
            sys.stdout.write(
                f"Bumped application version to {normalized_ver} "
                f"({len(changed)} files updated: {rel_list})\n"
            )
            return 0

        new_schema_ver, changed = bump_schema(
            repo_root,
            args.target,
            keep_previous=bool(args.keep_previous),
        )
        rel_list = ", ".join(p.relative_to(repo_root).as_posix() for p in changed)
        sys.stdout.write(
            f"Bumped Parquet schema version to {new_schema_ver} "
            f"({len(changed)} files updated: {rel_list})\n"
        )
        return 0
    except (FileNotFoundError, ValueError, OSError) as err:
        sys.stderr.write(f"error: {err}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
