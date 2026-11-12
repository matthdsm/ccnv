import re
import tempfile
import unittest
from pathlib import Path


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
