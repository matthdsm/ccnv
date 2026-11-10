import re
import unittest
from pathlib import Path


class TestDocsMarkdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.tracked_md_files = [
            cls.repo_root / "AGENTS.md",
            cls.repo_root / "CHANGELOG.md",
            cls.repo_root / "LICENSE.md",
            cls.repo_root / "README.md",
            cls.repo_root / "docs" / "benchmark.md",
            cls.repo_root / "docs" / "cli.md",
            cls.repo_root / "docs" / "designs" / "cffdna_fragmentomics_prenatal_cnv.md",
            cls.repo_root / "docs" / "methodology.md",
            cls.repo_root / "docs" / "pipeline.md",
            cls.repo_root / "src" / "README.md",
            cls.repo_root / "src" / "ccnv" / "README.md",
            cls.repo_root / "src" / "ccnv" / "features" / "README.md",
            cls.repo_root / "src" / "ccnv" / "io" / "README.md",
            cls.repo_root / "src" / "ccnv" / "models" / "README.md",
            cls.repo_root / "src" / "ccnv" / "qc" / "README.md",
            cls.repo_root / "src" / "ccnv" / "report" / "README.md",
            cls.repo_root / "src" / "ccnv" / "stats" / "README.md",
            cls.repo_root / "tests" / "fixtures" / "benchmark" / "README.md",
            cls.repo_root / "tests" / "fixtures" / "extract" / "README.md",
        ]

    def test_markdown_tables_and_cell_pipes(self) -> None:
        """Verify all tables have consistent column counts and no unescaped bare pipes in math cells."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            if not file_path.exists():
                continue
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
                    for math in re.findall(r"\$(.*?)\$", stripped):
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
            if not file_path.exists():
                continue
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

    @staticmethod
    def _mask_non_math_code(content: str) -> str:
        """Mask fenced code blocks and non-math inline code spans while preserving $`...`$ inline math and line offsets."""
        masked = re.sub(
            r"```.*?```",
            lambda m: "\n" * m.group(0).count("\n"),
            content,
            flags=re.DOTALL,
        )
        return re.sub(
            r"(?<!\$)`[^`\n]*`(?!\$)",
            lambda m: " " * len(m.group(0)),
            masked,
        )

    @staticmethod
    def _iter_inline_math(masked_content: str) -> list[tuple[int, str, str]]:
        """Return (lineno, raw_span, inner_math) for both $`...`$ and $...$ inline math spans."""
        spans: list[tuple[int, str, str]] = []
        pattern = re.compile(r"\$`([^`\n]+)`\$|(?<!\\)\$(?!\$|`)(.+?)(?<!\\|`)\$")
        for m in pattern.finditer(masked_content):
            lineno = masked_content[: m.start()].count("\n") + 1
            inner = m.group(1) if m.group(1) is not None else m.group(2)
            spans.append((lineno, m.group(0), inner))
        return spans

    def test_no_disallowed_latex_macros(self) -> None:
        r"""Verify no disallowed LaTeX macros like \operatorname exist in markdown files."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8")
            stripped_content = self._mask_non_math_code(content)
            for m in re.finditer(r"\\operatorname\*?\{", stripped_content):
                lineno = stripped_content[: m.start()].count("\n") + 1
                errors.append(
                    f"{file_path.relative_to(self.repo_root)}:{lineno} "
                    r"disallowed \operatorname macro found; use \mathrm or \text instead"
                )
            for m in re.finditer(r"\^(\*|\\top|\w+-\d+)", stripped_content):
                lineno = stripped_content[: m.start()].count("\n") + 1
                errors.append(
                    f"{file_path.relative_to(self.repo_root)}:{lineno} "
                    f"unbraced superscript '^{m.group(1)}' found; wrap in {{...}}"
                )
        self.assertEqual(
            errors, [], "Disallowed LaTeX macros found:\n" + "\n".join(errors)
        )

    def test_inline_math_no_padding(self) -> None:
        """Verify inline math delimiters do not have leading or trailing space padding outside code spans."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8")
            stripped_content = self._mask_non_math_code(content)
            for lineno, raw, inner in self._iter_inline_math(stripped_content):
                if inner.startswith(" ") or inner.endswith(" "):
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{lineno} "
                        f"space-padded inline math: '{raw}'"
                    )
        self.assertEqual(errors, [], "Math spacing issues found:\n" + "\n".join(errors))

    def test_math_delimiters_and_syntax(self) -> None:
        """Verify GitHub GFM + MathJax compatibility across all display and inline math blocks."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8")
            stripped_content = self._mask_non_math_code(content)

            # Check for single-line display math blocks ($$...$$ on same line)
            for m in re.finditer(r"(?m)^[ \t]*\$\$(.+?)\$\$[ \t]*$", stripped_content):
                lineno = stripped_content[: m.start()].count("\n") + 1
                errors.append(
                    f"{file_path.relative_to(self.repo_root)}:{lineno} "
                    "single-line display math block found; place $$ on separate lines"
                )

            # Check for indented $$ delimiters (cmark-gfm fails to parse indented $$ inside lists)
            for m in re.finditer(r"(?m)^[ \t]+\$\$[ \t]*$", stripped_content):
                lineno = stripped_content[: m.start()].count("\n") + 1
                errors.append(
                    f"{file_path.relative_to(self.repo_root)}:{lineno} "
                    "indented $$ delimiter found; place $$ at column 0 so GitHub recognizes display math"
                )

            # Check block math for %, bare < or >, \{, \}, \;, ^{*}, and single \\ in cases
            for m in re.finditer(r"\$\$(.*?)\$\$", stripped_content, flags=re.DOTALL):
                inner = m.group(1)
                lineno = stripped_content[: m.start()].count("\n") + 1
                if "%" in inner:
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{lineno} "
                        r"percent symbol '%' in display math triggers TeX comment hazard; move outside math"
                    )
                if re.search(r"(?<!\\)[<>]", inner):
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{lineno} "
                        r"bare '<' or '>' in display math triggers HTML parsing conflicts; use \lt or \gt"
                    )
                if r"\{" in inner or r"\}" in inner:
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{lineno} "
                        r"escaped brace \{ or \} in display math is stripped by GFM; use \lbrace or \rbrace"
                    )
                if r"\;" in inner:
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{lineno} "
                        r"\; in display math has backslash stripped by GFM; use \quad or comma"
                    )
                if "^{*}" in inner:
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{lineno} "
                        r"^{*} in display math triggers GFM emphasis; use ^{\ast}"
                    )
                if r"\begin{cases}" in inner and re.search(r"(?<!\\)\\\\(?!\\)", inner):
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{lineno} "
                        r"single \\ in \begin{cases} is stripped by GFM; use \\\\ for row breaks"
                    )

            # Check bare $...$ adjacent to hyphens (cmark-gfm rejects -$ or $-)
            for m in re.finditer(r"-\$(?!`|(?:\$))|(?<!\\|`)\$-", stripped_content):
                lineno = stripped_content[: m.start()].count("\n") + 1
                errors.append(
                    f"{file_path.relative_to(self.repo_root)}:{lineno} "
                    "bare $ delimiter adjacent to hyphen is rejected by GFM; use $`...`$ syntax"
                )

            # Check inline math for %, bare < or >
            for lineno, _raw, inner in self._iter_inline_math(stripped_content):
                if "%" in inner:
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{lineno} "
                        r"percent symbol '%' in inline math triggers TeX comment hazard; move outside math"
                    )
                if re.search(r"(?<!\\)[<>]", inner):
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{lineno} "
                        r"bare '<' or '>' in inline math triggers HTML parsing conflicts; use \lt or \gt"
                    )
        self.assertEqual(
            errors, [], "Math delimiter and syntax issues found:\n" + "\n".join(errors)
        )

    def test_no_local_filesystem_urls(self) -> None:
        """Verify no local file:/// URLs exist in tracked markdown files outside inline code/docs."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            if not file_path.exists():
                continue
            lines = file_path.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines, 1):
                # Only flag file:/// if not in a backticked example or comment
                if re.search(r"(?<!`)(?:\[[^\]]*\]\()?(file:///[^\s\)`]+)", line):
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}:{idx} contains local file:/// URL"
                    )
        self.assertEqual(
            errors, [], "Local filesystem URLs found:\n" + "\n".join(errors)
        )

    def test_relative_links_resolve(self) -> None:
        """Verify internal relative markdown links resolve to existing files."""
        errors: list[str] = []
        for file_path in self.tracked_md_files:
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8")
            for _text, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content):
                if link.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                target = link.split("#")[0]
                if not target:
                    continue
                target_path = (file_path.parent / target).resolve()
                if not target_path.exists():
                    errors.append(
                        f"{file_path.relative_to(self.repo_root)}: broken link '{link}' target not found"
                    )
        self.assertEqual(
            errors, [], "Broken markdown links found:\n" + "\n".join(errors)
        )
