import ast
import itertools
import re
import tempfile
import tomllib
import unittest
from pathlib import Path

_SKIPPED_TIER_TOKENS: frozenset[str] = frozenset(
    {"Typer commands", "lazy aliases", "Rust PyO3: noodles + bigtools"}
)
_LAZY_ROOT_ALIASES: dict[str, str] = {
    "run_extract": "extract.run",
    "run_reference": "reference.run",
    "run_reference_qc": "ref_qc.run_reference_qc",
    "run_predict": "predict.run",
}


def _is_overload(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _is_property(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(dec, ast.Name) and dec.id == "property" for dec in fn.decorator_list
    )


def _param_tokens(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, *, is_method: bool = False
) -> tuple[str, ...]:
    posonly = list(fn.args.posonlyargs)
    regular = list(fn.args.args)
    if is_method and regular and regular[0].arg in {"self", "cls"}:
        regular = regular[1:]
    elif is_method and posonly and posonly[0].arg in {"self", "cls"}:
        posonly = posonly[1:]
    parts = [arg.arg for arg in posonly]
    if posonly:
        parts.append("/")
    parts.extend(arg.arg for arg in regular)
    if fn.args.vararg is not None:
        parts.append(f"*{fn.args.vararg.arg}")
    elif fn.args.kwonlyargs:
        parts.append("*")
    parts.extend(arg.arg for arg in fn.args.kwonlyargs)
    if fn.args.kwarg is not None:
        parts.append(f"**{fn.args.kwarg.arg}")
    return tuple(parts)


def _find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if quote is not None:
            if ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(f"Unbalanced parentheses in signature span: {text!r}")


def _parse_doc_call_params(span: str) -> tuple[str, tuple[str, ...] | None]:
    stripped = span.strip()
    if "(" not in stripped:
        return stripped, None
    open_idx = stripped.index("(")
    close_idx = _find_matching_paren(stripped, open_idx)
    raw_name = stripped[:open_idx].strip()
    inner = stripped[open_idx + 1 : close_idx]
    parsed = ast.parse(f"def _sig({inner}):\n    pass\n")
    fn = parsed.body[0]
    if not isinstance(fn, ast.FunctionDef):
        raise TypeError(f"Expected FunctionDef parsing {span!r}")
    return raw_name, _param_tokens(fn, is_method=False)


class TestDocsMarkdown(unittest.TestCase):
    repo_root: Path
    zensical_md_files: list[Path]
    github_md_files: list[Path]
    tracked_md_files: list[Path]

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parent.parent
        docs_dir = cls.repo_root / "docs"
        excluded_docs = {"superpowers", "designs"}
        cls.zensical_md_files = sorted(
            p
            for p in docs_dir.rglob("*.md")
            if not any(part in excluded_docs for part in p.relative_to(docs_dir).parts)
        )
        root_md = [
            cls.repo_root / "AGENTS.md",
            cls.repo_root / "CHANGELOG.md",
            cls.repo_root / "LICENSE.md",
            cls.repo_root / "README.md",
        ]
        sub_md = sorted(
            p
            for base in (cls.repo_root / "src", cls.repo_root / "tests")
            for p in base.rglob("*.md")
            if "__pycache__" not in p.parts
        )
        cls.github_md_files = [p for p in root_md if p.exists()] + sub_md
        cls.tracked_md_files = cls.zensical_md_files + cls.github_md_files

    @staticmethod
    def _mask_fenced_code(content: str) -> str:
        """Replace fenced code blocks with line-preserving newlines."""
        return re.sub(
            r"(?m)^([ \t]*`{3,})[^\n]*\n.*?\n[ \t]*\1[ \t]*$",
            lambda m: "\n" * m.group(0).count("\n"),
            content,
            flags=re.DOTALL,
        )

    @classmethod
    def _mask_non_math_code(cls, content: str) -> str:
        """Mask fenced code blocks and non-math inline code spans while preserving $`...`$ and line offsets."""
        masked = cls._mask_fenced_code(content)
        return re.sub(
            r"(?<!\$)(`+)([^`\n]+?)\1(?!\$)",
            lambda m: " " * len(m.group(0)),
            masked,
        )

    @staticmethod
    def _iter_inline_math(masked_content: str) -> list[tuple[int, str, str]]:
        """Return (lineno, raw_span, inner_math) for $`...`$ and $...$ inline math spans outside block math."""
        no_block = re.sub(
            r"\$\$.*?\$\$",
            lambda m: "\n" * m.group(0).count("\n"),
            masked_content,
            flags=re.DOTALL,
        )
        spans: list[tuple[int, str, str]] = []
        pattern = re.compile(r"\$`([^`\n]+)`\$|(?<!\\)\$(?!\$|`)([^\n$]+?)(?<!\\|`)\$")
        for m in pattern.finditer(no_block):
            lineno = no_block[: m.start()].count("\n") + 1
            inner = m.group(1) if m.group(1) is not None else m.group(2)
            spans.append((lineno, m.group(0), inner))
        return spans

    def test_markdown_tables_and_cell_pipes(self) -> None:
        """Verify all tables have consistent column counts and no unescaped bare pipes in math cells."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            lines = file_path.read_text(encoding="utf-8").splitlines()
            in_table = False
            expected_cols: int | None = None
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("|") and stripped.endswith("|"):
                    cells = [c.strip() for c in re.split(r"(?<!\\)\|", stripped)[1:-1]]
                    if not in_table:
                        in_table = True
                        expected_cols = len(cells)
                    elif len(cells) != expected_cols:
                        errors.append(
                            f"{file_path.relative_to(self.repo_root)}:{idx} "
                            f"column mismatch: expected {expected_cols}, got {len(cells)}"
                        )
                    for math in re.findall(r"\$`?([^`$\n]+)`?\$", stripped):
                        if re.search(r"(?<!\\)\|", math):
                            errors.append(
                                f"{file_path.relative_to(self.repo_root)}:{idx} "
                                f"unescaped pipe in math: ${math}$"
                            )
                else:
                    in_table = False
                    expected_cols = None
        self.assertEqual(
            errors, [], "Markdown table issues found:\n" + "\n".join(errors)
        )

    def test_mermaid_no_empty_lines(self) -> None:
        """Verify Mermaid code blocks contain no blank lines."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            lines = file_path.read_text(encoding="utf-8").splitlines()
            in_mermaid = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```mermaid"):
                    in_mermaid = True
                elif in_mermaid and stripped.startswith("```"):
                    in_mermaid = False
                elif in_mermaid and stripped == "":
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{idx} empty line in mermaid block"
                    )
        self.assertEqual(
            errors, [], "Mermaid syntax issues found:\n" + "\n".join(errors)
        )

    def test_shared_latex_syntax_and_balance(self) -> None:
        """Verify TeX macro validity, brace/delimiter balance, and display block placement across all files."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            rel = file_path.relative_to(self.repo_root)
            content = file_path.read_text(encoding="utf-8")
            stripped = self._mask_non_math_code(content)

            for m in re.finditer(r"\\operatorname\*?\{", stripped):
                lineno = stripped[: m.start()].count("\n") + 1
                errors.append(
                    f"{rel}:{lineno} disallowed \\operatorname macro; use \\mathrm or \\text"
                )
            for m in re.finditer(r"(?m)^[ \t]*\$\$(.+?)\$\$[ \t]*$", stripped):
                lineno = stripped[: m.start()].count("\n") + 1
                errors.append(
                    f"{rel}:{lineno} single-line display math block; place $$ on separate lines"
                )
            for m in re.finditer(r"(?m)^[ \t]+\$\$[ \t]*$", stripped):
                lineno = stripped[: m.start()].count("\n") + 1
                errors.append(
                    f"{rel}:{lineno} indented $$ delimiter; place $$ at column 0"
                )

            all_exprs: list[tuple[int, str]] = []
            for m in re.finditer(r"\$\$(.*?)\$\$", stripped, flags=re.DOTALL):
                lineno = stripped[: m.start()].count("\n") + 1
                all_exprs.append((lineno, m.group(1)))
            for m in re.finditer(r"\\\[(.*?)\\\]", stripped, flags=re.DOTALL):
                lineno = stripped[: m.start()].count("\n") + 1
                all_exprs.append((lineno, m.group(1)))
            for m in re.finditer(r"\\\((.*?)\\\)", stripped):
                lineno = stripped[: m.start()].count("\n") + 1
                all_exprs.append((lineno, m.group(1)))
            for lineno, raw, inner in self._iter_inline_math(stripped):
                if inner.startswith(" ") or inner.endswith(" "):
                    errors.append(f"{rel}:{lineno} space-padded inline math: '{raw}'")
                all_exprs.append((lineno, inner))

            for lineno, expr in all_exprs:
                for m in re.finditer(r"\^(\*|\\top|\w+-\d+)", expr):
                    errors.append(
                        f"{rel}:{lineno} unbraced superscript '^{m.group(1)}'; wrap in {{...}}"
                    )
                if "%" in expr:
                    errors.append(
                        f"{rel}:{lineno} '%' in math triggers TeX comment hazard; move outside math"
                    )
                if re.search(r"(?<!\\)[<>]", expr):
                    errors.append(
                        f"{rel}:{lineno} bare '<' or '>' in math; use \\lt or \\gt"
                    )
                clean_braces = re.sub(r"\\[{}]", "", expr)
                depth = 0
                for ch in clean_braces:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                    if depth < 0:
                        break
                if depth != 0:
                    errors.append(f"{rel}:{lineno} unbalanced TeX braces in: {expr!r}")
                n_left = len(re.findall(r"\\left(?![a-zA-Z])", expr))
                n_right = len(re.findall(r"\\right(?![a-zA-Z])", expr))
                if n_left != n_right:
                    errors.append(
                        f"{rel}:{lineno} mismatched \\left ({n_left}) vs \\right ({n_right})"
                    )
                if len(re.findall(r"\\lbrace(?![a-zA-Z])", expr)) != len(
                    re.findall(r"\\rbrace(?![a-zA-Z])", expr)
                ):
                    errors.append(f"{rel}:{lineno} mismatched \\lbrace vs \\rbrace")
                if re.findall(r"\\begin\{([^}]+)\}", expr) != re.findall(
                    r"\\end\{([^}]+)\}", expr
                ):
                    errors.append(f"{rel}:{lineno} mismatched \\begin / \\end")

        self.assertEqual(
            errors, [], "Shared LaTeX syntax/balance issues:\n" + "\n".join(errors)
        )

    def test_zensical_docs_rendering(self) -> None:
        """Enforce Zensical (pymdownx.arithmatex + admonition) compatibility on all docs/**/*.md files."""
        errors: list[str] = []
        smart_dollar_re = re.compile(
            r"(?<!\\)\$(?!\s)(?:\\.|[^\\$\x02\x03\n])+?(?<!\s)\$"
        )
        for file_path in self.zensical_md_files:
            rel = file_path.relative_to(self.repo_root)
            content = file_path.read_text(encoding="utf-8")
            no_fences = self._mask_fenced_code(content)

            for m in re.finditer(
                r"(?m)^[ \t]*>[ \t]*\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]",
                no_fences,
            ):
                lineno = no_fences[: m.start()].count("\n") + 1
                errors.append(
                    f"{rel}:{lineno} GitHub alert syntax in docs/; use Zensical '!!! note' admonition"
                )

            masked = self._mask_non_math_code(content)
            for m in re.finditer(r"\$`[^`\n]+`\$", masked):
                lineno = masked[: m.start()].count("\n") + 1
                errors.append(
                    f"{rel}:{lineno} GitHub $`...`$ syntax breaks Zensical pymdownx.arithmatex; use $...$"
                )

            for m in re.finditer(r"\$\$(.*?)\$\$", masked, flags=re.DOTALL):
                inner = m.group(1)
                lineno = masked[: m.start()].count("\n") + 1
                if r"\begin{cases}" in inner and r"\\\\" in inner:
                    errors.append(
                        f"{rel}:{lineno} quad-backslash \\\\\\\\ in \\begin{{cases}} renders blank row in Zensical; use \\\\"
                    )

            code_sentinel = re.sub(
                r"(`+)([^`\n]+?)\1",
                lambda m: "\x02" + ("c" * (len(m.group(0)) - 2)) + "\x03",
                no_fences,
            )
            no_blocks = re.sub(
                r"\$\$.*?\$\$",
                lambda m: "\n" * m.group(0).count("\n"),
                code_sentinel,
                flags=re.DOTALL,
            )
            after_inline = smart_dollar_re.sub(
                lambda m: " " * len(m.group(0)), no_blocks
            )
            for m in re.finditer(r"(?<!\\)\$", after_inline):
                lineno = after_inline[: m.start()].count("\n") + 1
                errors.append(
                    f"{rel}:{lineno} unparsed stray '$' under Zensical pymdownx.arithmatex"
                )

        self.assertEqual(
            errors, [], "Zensical docs/ rendering issues found:\n" + "\n".join(errors)
        )

    def test_github_markdown_rendering(self) -> None:
        """Enforce GitHub Flavored Markdown (cmark-gfm) compatibility on all non-docs/ Markdown files."""
        errors: list[str] = []
        for file_path in self.github_md_files:
            rel = file_path.relative_to(self.repo_root)
            content = file_path.read_text(encoding="utf-8")
            no_fences = self._mask_fenced_code(content)

            for m in re.finditer(r"(?m)^[ \t]*(?:!!!|\?\?\?)[ \t]+\w+", no_fences):
                lineno = no_fences[: m.start()].count("\n") + 1
                errors.append(
                    f"{rel}:{lineno} Zensical admonition in GitHub Markdown; use '> [!NOTE]' alert"
                )

            masked = self._mask_non_math_code(content)

            for m in re.finditer(r"\$\$(.*?)\$\$", masked, flags=re.DOTALL):
                inner = m.group(1)
                lineno = masked[: m.start()].count("\n") + 1
                if r"\{" in inner or r"\}" in inner:
                    errors.append(
                        f"{rel}:{lineno} \\{{ or \\}} in GFM display math; use \\lbrace or \\rbrace"
                    )
                if r"\;" in inner:
                    errors.append(
                        f"{rel}:{lineno} \\; in GFM display math; use \\quad or comma"
                    )
                if "^{*}" in inner:
                    errors.append(
                        f"{rel}:{lineno} ^{{*}} in GFM display math; use ^{{\\ast}}"
                    )
                if r"\begin{cases}" in inner and re.search(r"(?<!\\)\\\\(?!\\)", inner):
                    errors.append(
                        f"{rel}:{lineno} single \\\\ in GFM \\begin{{cases}}; use \\\\\\\\"
                    )

            no_block = re.sub(
                r"\$\$.*?\$\$",
                lambda m: "\n" * m.group(0).count("\n"),
                masked,
                flags=re.DOTALL,
            )
            no_backtick_math = re.sub(
                r"\$`[^`\n]+`\$",
                lambda m: " " * len(m.group(0)),
                no_block,
            )
            for m in re.finditer(
                r"(.?)(?<!\\)\$(?!\$|`)([^\n$]+?)(?<!\\|`)\$(.?)", no_backtick_math
            ):
                lineno = no_backtick_math[: m.start()].count("\n") + 1
                pre, inner, post = m.group(1), m.group(2), m.group(3)
                if pre in ("-", "–", "(") or post in ("-", "–", ")"):
                    adj = pre if pre in ("-", "–", "(") else post
                    errors.append(
                        f"{rel}:{lineno} bare $...$ adjacent to '{adj}' ({m.group(0)!r}); use $`...`$ in GitHub Markdown"
                    )
                elif any(ch in inner for ch in ("{", "}", "_", "*")):
                    errors.append(
                        f"{rel}:{lineno} bare ${inner}$ contains braces/underscores/asterisks; use $`...`$ in GitHub Markdown"
                    )

        self.assertEqual(
            errors, [], "GitHub Markdown rendering issues found:\n" + "\n".join(errors)
        )

    def test_links_and_urls(self) -> None:
        """Verify no local file:/// URLs, no bracketed pseudo-links, and all relative links resolve."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            rel = file_path.relative_to(self.repo_root)
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            for idx, line in enumerate(lines, 1):
                if re.search(r"(?<!`)(?:\[[^\]]*\]\()?(file:///[^\s\)`]+)", line):
                    errors.append(f"{rel}:{idx} contains local file:/// URL")
                line_no_double_ticks = re.sub(r"``.+?``", "", line)
                if re.search(r"`\[src/ccnv/[^\]]+\]`", line_no_double_ticks):
                    errors.append(
                        f"{rel}:{idx} contains unlinked pseudo-link `[src/ccnv/...]`; use a relative Markdown link"
                    )

            no_fences = self._mask_fenced_code(content)
            no_code = re.sub(r"`[^`\n]*`", "", no_fences)
            for _text, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", no_code):
                if link.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                target = link.split("#")[0]
                if not target:
                    continue
                target_path = (file_path.parent / target).resolve()
                if not target_path.exists():
                    errors.append(f"{rel}: broken link '{link}' target not found")

        self.assertEqual(
            errors, [], "Markdown link/URL issues found:\n" + "\n".join(errors)
        )

    def test_unbraced_superscript_scoped_to_math_only(self) -> None:
        """Verify prose tokens like x^* do not fail while math spans $x^*$ and \\(x^*\\) are flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prose_md = Path(tmpdir) / "prose.md"
            prose_md.write_text(
                "Plain prose token x^* and tag v1-2.\n", encoding="utf-8"
            )
            math_dollar_md = Path(tmpdir) / "dollar.md"
            math_dollar_md.write_text("Inline math $x^*$ fails.\n", encoding="utf-8")
            math_paren_md = Path(tmpdir) / "paren.md"
            math_paren_md.write_text(
                "Arithmatex \\(x^\\top\\) fails.\n", encoding="utf-8"
            )

            orig_files = self.tracked_md_files
            orig_root = self.repo_root
            try:
                self.repo_root = Path(tmpdir)
                self.tracked_md_files = [prose_md]
                self.test_shared_latex_syntax_and_balance()

                self.tracked_md_files = [math_dollar_md]
                with self.assertRaises(AssertionError):
                    self.test_shared_latex_syntax_and_balance()

                self.tracked_md_files = [math_paren_md]
                with self.assertRaises(AssertionError):
                    self.test_shared_latex_syntax_and_balance()
            finally:
                self.tracked_md_files = orig_files
                self.repo_root = orig_root

    def test_no_stale_v1_schema_or_solver_claims_in_published_docs(self) -> None:
        """Verify no obsolete v1 schema names, removed functions, or legacy solver claims exist."""
        stale_tokens = (
            "list<uint16>",
            "nearest_ref_indices",
            "align_reference_to_weighted_sample",
            "pysam.AlignmentFile",
            "MATERNAL_SOMATIC",
            "write_calls_vcf",
        )
        for path in self.tracked_md_files:
            if path.name == "CHANGELOG.md":
                continue
            text = path.read_text(encoding="utf-8")
            for token in stale_tokens:
                self.assertNotIn(
                    token,
                    text,
                    f"Stale v1 token {token!r} found in {path.relative_to(self.repo_root)}",
                )

    def test_api_doc_bidirectional_symbol_and_signature_sync(self) -> None:
        """Verify docs/api.md stays bidirectionally synchronized with src/ccnv symbols and signatures."""
        api_path = self.repo_root / "docs" / "api.md"
        self.assertTrue(api_path.exists(), "docs/api.md must exist")
        api_text = api_path.read_text(encoding="utf-8")

        zensical_cfg = tomllib.loads(
            (self.repo_root / "zensical.toml").read_text(encoding="utf-8")
        )
        nav_entries = (
            zensical_cfg.get("project", {}).get("nav") or zensical_cfg.get("nav") or []
        )
        nav_targets = [
            target
            for item in nav_entries
            if isinstance(item, dict)
            for target in item.values()
        ]
        self.assertIn(
            "api.md", nav_targets, "api.md must be registered in zensical.toml nav"
        )

        prose_without_fences = re.sub(r"```.*?```", "", api_text, flags=re.DOTALL)
        for cited_path in re.findall(
            r"`((?:src|crates)/[A-Za-z0-9_./-]+)`", prose_without_fences
        ):
            self.assertTrue(
                (self.repo_root / cited_path).exists(),
                f"Cited source path in docs/api.md does not exist: {cited_path}",
            )

        section_matches = list(
            re.finditer(r"^### `([^`]+)` \((.+?)\)\s*$", api_text, flags=re.MULTILINE)
        )
        sections_by_path: dict[str, str] = {}
        for match in section_matches:
            start = match.end()
            next_heading = re.search(r"^#{2,3} ", api_text[start:], flags=re.MULTILINE)
            end = start + next_heading.start() if next_heading else len(api_text)
            body = api_text[start:end]
            for path_str in re.findall(r"`([^`]+)`", match.group(2)):
                if path_str.endswith((".py", ".pyi")):
                    sections_by_path[path_str] = body

        source_paths = sorted((self.repo_root / "src" / "ccnv").rglob("*.py"))
        core_stub = self.repo_root / "src" / "ccnv" / "_core.pyi"
        if core_stub.exists():
            source_paths.append(core_stub)

        for src_path in source_paths:
            rel = src_path.relative_to(self.repo_root).as_posix()
            tree = ast.parse(
                src_path.read_text(encoding="utf-8"), filename=str(src_path)
            )
            ast_symbols: set[str] = set()
            required_symbols: set[str] = set()
            required_call_params: dict[str, tuple[str, ...]] = {}
            all_call_params: dict[str, tuple[str, ...]] = {}

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_overload(node):
                        continue
                    params = _param_tokens(node, is_method=False)
                    ast_symbols.add(node.name)
                    all_call_params[node.name] = params
                    if not node.name.startswith("_") or node.name == "__getattr__":
                        required_symbols.add(node.name)
                        required_call_params[node.name] = params
                elif isinstance(node, ast.ClassDef):
                    ast_symbols.add(node.name)
                    if not node.name.startswith("_"):
                        required_symbols.add(node.name)
                    for member in node.body:
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if _is_overload(member):
                                continue
                            qual_member = f"{node.name}.{member.name}"
                            ast_symbols.add(member.name)
                            ast_symbols.add(qual_member)
                            params = _param_tokens(member, is_method=True)
                            if not _is_property(member):
                                all_call_params[member.name] = params
                                all_call_params[qual_member] = params
                            if not node.name.startswith("_") and (
                                not member.name.startswith("_")
                                or member.name == "__getitem__"
                            ):
                                required_symbols.add(member.name)
                                if not _is_property(member):
                                    required_call_params[qual_member] = params
                elif src_path == core_stub:
                    target_id: str | None = None
                    if isinstance(node, ast.AnnAssign) and isinstance(
                        node.target, ast.Name
                    ):
                        target_id = node.target.id
                    elif (
                        isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                    ):
                        target_id = node.targets[0].id
                    if target_id is not None and target_id.isupper():
                        ast_symbols.add(target_id)
                        required_symbols.add(target_id)

            if rel == "src/ccnv/__init__.py":
                for alias in _LAZY_ROOT_ALIASES:
                    ast_symbols.add(alias)
                    required_symbols.add(alias)

            if not required_symbols:
                self.assertIn(
                    f"`{rel}`",
                    prose_without_fences,
                    f"Re-export module must be referenced in docs/api.md: {rel}",
                )
                continue

            self.assertIn(
                rel,
                sections_by_path,
                f"Module must have a dedicated ### section in docs/api.md: {rel}",
            )
            section_body = sections_by_path[rel]
            section_no_fences = re.sub(r"```.*?```", "", section_body, flags=re.DOTALL)

            section_tokens: set[str] = set()
            for span in re.findall(r"`([^`]+)`", section_no_fences):
                section_tokens.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", span))

            missing_symbols = sorted(required_symbols - section_tokens)
            self.assertEqual(
                missing_symbols,
                [],
                f"Undocumented required symbols in {rel}: {missing_symbols}",
            )

            documented_params: dict[str, tuple[str, ...]] = {}
            for line in section_no_fences.splitlines():
                internal_match = re.match(
                    r"^\s*-\s+\*\*Internal\b.*?\*\*\s+—\s+(.*)$", line
                )
                if internal_match:
                    for helper in re.findall(
                        r"`(_[A-Za-z0-9_]+)`", internal_match.group(1)
                    ):
                        self.assertIn(
                            helper,
                            ast_symbols,
                            f"Stale internal helper `{helper}` documented in {rel}",
                        )
                    continue

                bullet_match = re.match(r"^\s*-\s+\*\*`([^`]+)`\*\*", line)
                if bullet_match:
                    doc_name, parsed_params = _parse_doc_call_params(
                        bullet_match.group(1)
                    )
                    short_name = doc_name.split(".")[-1]
                    self.assertTrue(
                        doc_name in ast_symbols or short_name in ast_symbols,
                        f"Stale or mislocated symbol `{doc_name}` documented under {rel}",
                    )
                    if parsed_params is not None:
                        expected_params = all_call_params.get(
                            doc_name
                        ) or all_call_params.get(short_name)
                        self.assertIsNotNone(
                            expected_params,
                            f"Callable signature `{bullet_match.group(1)}` does not match a function/method in {rel}",
                        )
                        self.assertEqual(
                            parsed_params,
                            expected_params,
                            f"Parameter signature drift for `{doc_name}` in {rel}",
                        )
                        documented_params[doc_name] = parsed_params
                        documented_params[short_name] = parsed_params

            for callable_name, expected_params in required_call_params.items():
                short_name = callable_name.split(".")[-1]
                doc_sig = documented_params.get(callable_name) or documented_params.get(
                    short_name
                )
                self.assertEqual(
                    doc_sig,
                    expected_params,
                    f"Missing or mismatched explicit parameter signature for `{callable_name}` in {rel}",
                )

    def test_api_doc_mermaid_dags_match_codebase_call_graph(self) -> None:
        """Verify Mermaid tier and pipeline DAGs in docs/api.md match actual codebase modules and calls."""
        api_text = (self.repo_root / "docs" / "api.md").read_text(encoding="utf-8")
        mermaid_blocks = re.findall(
            r"```mermaid\nflowchart\b[^\n]*\n(.*?)```", api_text, flags=re.DOTALL
        )
        self.assertGreaterEqual(
            len(mermaid_blocks),
            5,
            "docs/api.md must contain all 5 architecture and pipeline Mermaid DAGs",
        )

        src_root = self.repo_root / "src" / "ccnv"
        existing_modules: set[str] = set()
        for path in src_root.rglob("*.py"):
            rel_mod = (
                path.relative_to(self.repo_root / "src").with_suffix("").as_posix()
            )
            dot_mod = rel_mod.replace("/", ".")
            existing_modules.add(dot_mod.removesuffix(".__init__"))
            existing_modules.add(dot_mod)
        existing_modules.add("ccnv._core")

        tier_block = mermaid_blocks[0]
        for label in re.findall(r'\["([^"]+)"\]', tier_block):
            head = label.split("(", 1)[0].strip()
            if head.startswith("ccnv.") and head not in existing_modules:
                self.fail(f"Tier DAG references missing package/module {head!r}")
            paren_match = re.search(r"\(([^)]+)\)", label)
            if not paren_match:
                continue
            paren_text = paren_match.group(1).strip()
            if paren_text in _SKIPPED_TIER_TOKENS:
                continue
            base_pkg = head if head.startswith("ccnv.") else "ccnv"
            for mod_name in (item.strip() for item in paren_text.split(",")):
                target = f"{base_pkg}.{mod_name}"
                self.assertIn(
                    target,
                    existing_modules,
                    f"Tier DAG references non-existent module {target!r}",
                )

        calls_by_qual: dict[str, set[str]] = {}
        short_to_quals: dict[str, set[str]] = {}
        source_paths = sorted(src_root.rglob("*.py")) + [src_root / "_core.pyi"]
        for src_path in source_paths:
            mod = (
                src_path.relative_to(self.repo_root / "src")
                .with_suffix("")
                .as_posix()
                .replace("/", ".")
                .removesuffix(".__init__")
            )
            tree = ast.parse(
                src_path.read_text(encoding="utf-8"), filename=str(src_path)
            )
            alias_map: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for imported in node.names:
                        if imported.asname:
                            alias_map[imported.asname] = (
                                f"{node.module}.{imported.name}"
                            )

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = f"{mod}.{node.name}"
                    for key in (
                        node.name,
                        f"{src_path.stem}.{node.name}",
                        qual,
                        qual.removeprefix("ccnv."),
                    ):
                        short_to_quals.setdefault(key, set()).add(qual)
                    called: set[str] = set()
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call):
                            if isinstance(sub.func, ast.Name):
                                name = sub.func.id
                                called.add(
                                    _LAZY_ROOT_ALIASES.get(
                                        name, alias_map.get(name, name)
                                    )
                                )
                            elif isinstance(sub.func, ast.Attribute):
                                called.add(sub.func.attr)
                    calls_by_qual[qual] = called
                elif isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            qual = f"{mod}.{node.name}.{member.name}"
                            for key in (
                                member.name,
                                f"{node.name}.{member.name}",
                                qual,
                                qual.removeprefix("ccnv."),
                            ):
                                short_to_quals.setdefault(key, set()).add(qual)
                            called = set()
                            for sub in ast.walk(member):
                                if isinstance(sub, ast.Call):
                                    if isinstance(sub.func, ast.Name):
                                        name = sub.func.id
                                        called.add(
                                            _LAZY_ROOT_ALIASES.get(
                                                name, alias_map.get(name, name)
                                            )
                                        )
                                    elif isinstance(sub.func, ast.Attribute):
                                        called.add(sub.func.attr)
                            calls_by_qual[qual] = called

        def is_reachable(src_label: str, dst_label: str) -> bool:
            srcs: set[str] = set()
            for part in (p.strip() for p in src_label.split("/")):
                srcs.update(short_to_quals.get(part, set()))
            dsts: set[str] = set()
            for part in (p.strip() for p in dst_label.split("/")):
                dsts.update(short_to_quals.get(part, set()))
            if not srcs or not dsts:
                return False
            visited = set(srcs)
            stack = list(srcs)
            while stack:
                current = stack.pop()
                if current in dsts:
                    return True
                for callee_tok in calls_by_qual.get(current, ()):
                    for next_qual in short_to_quals.get(callee_tok, ()):
                        if next_qual not in visited:
                            visited.add(next_qual)
                            stack.append(next_qual)
            return False

        for dag_idx, dag_block in enumerate(mermaid_blocks[1:], start=2):
            node_labels: dict[str, str] = {}
            for raw_line in dag_block.splitlines():
                stripped = raw_line.strip()
                if stripped.startswith("subgraph "):
                    continue
                for node_id, node_label in re.findall(
                    r'([A-Za-z0-9_]+)\["([^"]+)"\]', stripped
                ):
                    node_labels[node_id] = node_label
                    for part in (p.strip() for p in node_label.split("/")):
                        self.assertIn(
                            part,
                            short_to_quals,
                            f"DAG 1.{dag_idx} node {node_id}[{node_label!r}] unresolved in codebase AST",
                        )

            for raw_line in dag_block.splitlines():
                stripped = raw_line.strip()
                if "-->" not in stripped:
                    continue
                chain_ids: list[str] = []
                for segment in stripped.split("-->"):
                    id_match = re.match(r"^\s*([A-Za-z0-9_]+)", segment)
                    self.assertIsNotNone(
                        id_match,
                        f"Malformed Mermaid edge in DAG 1.{dag_idx}: {stripped}",
                    )
                    if id_match is not None:
                        chain_ids.append(id_match.group(1))
                for src_id, dst_id in itertools.pairwise(chain_ids):
                    src_label = node_labels[src_id]
                    dst_label = node_labels[dst_id]
                    self.assertTrue(
                        is_reachable(src_label, dst_label),
                        f"DAG 1.{dag_idx} edge {src_label!r} --> {dst_label!r} has no path in codebase AST",
                    )
