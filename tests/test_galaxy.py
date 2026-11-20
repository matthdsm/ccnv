"""Contract and end-to-end staging tests for the Galaxy tool suite under galaxy/."""

from __future__ import annotations

import copy
import gzip
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pysam
import typer.core
import typer.main
import yaml
from helpers import write_synthetic_sample, write_test_bam

import ccnv
from ccnv.cli import app
from ccnv.io.samplesheet import load_samplesheet

_GALAXY_DIR = Path(__file__).resolve().parent.parent / "galaxy"
_TOOL_SUBCOMMANDS: dict[str, str] = {
    "ccnv_extract.xml": "extract",
    "ccnv_reference.xml": "reference",
    "ccnv_ref_qc.xml": "ref-qc",
    "ccnv_predict.xml": "predict",
}
_CONTIGS: tuple[tuple[str, int], ...] = (
    ("chr1", 4000),
    ("chr2", 4000),
    ("chrX", 2000),
    ("chrY", 2000),
)
_BIN_SIZE = 500


def _load_macros() -> tuple[dict[str, str], dict[str, ET.Element]]:
    """Parse token definitions and XML macros from galaxy/macros.xml."""
    macros_root = ET.parse(_GALAXY_DIR / "macros.xml").getroot()
    tokens: dict[str, str] = {}
    xml_macros: dict[str, ET.Element] = {}
    for child in macros_root:
        if child.tag == "token":
            name = child.attrib.get("name")
            if name:
                tokens[name] = child.text or ""
        elif child.tag == "xml":
            name = child.attrib.get("name")
            if name:
                xml_macros[name] = child
    return tokens, xml_macros


def _expand_macros_in_tree(
    node: ET.Element, xml_macros: dict[str, ET.Element]
) -> list[str]:
    """Expand <expand macro='...'> elements in place and return referenced names."""
    seen: list[str] = []
    idx = 0
    children = list(node)
    while idx < len(children):
        child = children[idx]
        if child.tag == "expand":
            macro_name = child.attrib["macro"]
            seen.append(macro_name)
            if macro_name not in xml_macros:
                idx += 1
                continue
            replacement = [copy.deepcopy(el) for el in list(xml_macros[macro_name])]
            children[idx : idx + 1] = replacement
            node[:] = children
            continue
        seen.extend(_expand_macros_in_tree(child, xml_macros))
        idx += 1
    return seen


def _resolve_tokens(text: str, tokens: dict[str, str]) -> str:
    """Resolve @TOKEN@ placeholders iteratively using declared macro tokens."""
    resolved = text
    for _ in range(5):
        changed = False
        for token_name, token_val in tokens.items():
            if token_name in resolved:
                resolved = resolved.replace(token_name, token_val)
                changed = True
        if not changed:
            break
    return resolved


def _write_paired_bam(path: Path, contigs: tuple[tuple[str, int], ...]) -> None:
    """Write a minimal coordinate-sorted, indexed paired-end BAM fixture."""
    seq = "ACGT" * 25
    qual = pysam.qualitystring_to_array("I" * len(seq))
    reads: list[pysam.AlignedSegment] = []
    for ref_id, (_, length) in enumerate(contigs):
        for idx, pos in enumerate(range(0, length - 200, 125)):
            mate_pos = pos + 50
            tlen = 150
            for flag, start, mpos, isize in (
                (0x1 | 0x2 | 0x40, pos, mate_pos, tlen),
                (0x1 | 0x2 | 0x80, mate_pos, pos, -tlen),
            ):
                rec = pysam.AlignedSegment()
                rec.query_name = f"pair_{ref_id}_{idx}"
                rec.query_sequence = seq
                rec.flag = flag
                rec.reference_id = ref_id
                rec.reference_start = start
                rec.mapping_quality = 60
                rec.cigar = ((0, len(seq)),)
                rec.next_reference_id = ref_id
                rec.next_reference_start = mpos
                rec.template_length = isize
                rec.query_qualities = qual
                reads.append(rec)
    write_test_bam(path, reads, contigs=contigs, index=True)


def _run_ccnv(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Execute the `ccnv` CLI inside `cwd` for one staged Galaxy job directory."""
    return subprocess.run(
        [sys.executable, "-m", "ccnv.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _cli_long_options(subcommand: str) -> set[str]:
    """Return every registered CLI long flag for a Typer subcommand."""
    root_cmd = typer.main.get_command(app)
    if not isinstance(root_cmd, typer.core.TyperGroup):
        raise TypeError("Expected TyperGroup from ccnv.cli.app")
    cmd = root_cmd.commands[subcommand]
    flags: set[str] = set()
    for param in cmd.params:
        for opt in (*param.opts, *param.secondary_opts):
            if opt.startswith("--"):
                flags.add(opt)
    return flags


def _collect_section_fields(inputs_el: ET.Element) -> dict[str, set[str]]:
    """Collect Galaxy section names mapped to their child parameter names."""
    section_fields: dict[str, set[str]] = {}
    for child in inputs_el:
        if child.tag == "section":
            sec_name = child.attrib["name"]
            section_fields[sec_name] = {
                _param_galaxy_name(sub) for sub in child.iter("param")
            }
    return section_fields


def _param_galaxy_name(param_el: ET.Element) -> str:
    """Resolve Galaxy's parameter identifier from name or argument attribute."""
    if "name" in param_el.attrib:
        return param_el.attrib["name"]
    arg = param_el.attrib["argument"]
    return arg.lstrip("-").replace("-", "_")


class TestGalaxyToolSuite(unittest.TestCase):
    """Validate Galaxy ToolShed metadata, tool XMLs, CLI parity, and .dat staging."""

    def test_shed_yaml_and_macros_valid(self) -> None:
        shed_path = _GALAXY_DIR / ".shed.yml"
        self.assertTrue(shed_path.is_file(), "galaxy/.shed.yml must exist")
        shed = yaml.safe_load(shed_path.read_text(encoding="utf-8"))
        self.assertIsInstance(shed, dict)
        for key in (
            "name",
            "owner",
            "description",
            "homepage_url",
            "remote_repository_url",
            "type",
            "categories",
            "auto_tool_repositories",
            "suite",
        ):
            self.assertIn(key, shed, f"Missing {key!r} in galaxy/.shed.yml")
        self.assertEqual(shed["name"], "ccnv")
        self.assertEqual(shed["suite"]["name"], "suite_ccnv")
        self.assertEqual(shed["suite"]["type"], "repository_suite_definition")

        tokens, xml_macros = _load_macros()
        self.assertEqual(tokens["@TOOL_VERSION@"], ccnv.__version__)
        self.assertIn("@STAGE_OPTIONAL_FASTA@", tokens)
        self.assertIn("@STAGE_COVARIATE_TRACKS@", tokens)
        for required_macro in (
            "biotools",
            "requirements",
            "version_command",
            "stdio",
            "loglevel_param",
            "optional_fasta_macro",
            "covariate_tracks_macro",
            "citations",
        ):
            self.assertIn(
                required_macro, xml_macros, f"Missing macro {required_macro!r}"
            )

    def test_all_tool_xmls_parse_expand_and_resolve_tokens(self) -> None:
        tokens, xml_macros = _load_macros()
        for filename in _TOOL_SUBCOMMANDS:
            xml_path = _GALAXY_DIR / filename
            self.assertTrue(xml_path.is_file(), f"Missing {filename}")
            root = ET.parse(xml_path).getroot()
            self.assertEqual(root.tag, "tool")
            for attr in ("id", "name", "version", "profile"):
                self.assertIn(attr, root.attrib, f"Missing {attr!r} in {filename}")
                resolved_attr = _resolve_tokens(root.attrib[attr], tokens)
                self.assertNotRegex(
                    resolved_attr,
                    r"@[A-Z0-9_]+@",
                    f"Unresolved token in {filename} attribute {attr}",
                )

            used_macros = _expand_macros_in_tree(root, xml_macros)
            for m in used_macros:
                self.assertIn(m, xml_macros, f"Undefined macro {m!r} in {filename}")

            for tag in (
                "description",
                "requirements",
                "stdio",
                "version_command",
                "command",
                "inputs",
                "outputs",
                "tests",
                "help",
                "citations",
            ):
                self.assertIsNotNone(
                    root.find(tag),
                    f"Missing <{tag}> section after expansion in {filename}",
                )

            cmd_el = root.find("command")
            self.assertIsNotNone(cmd_el)
            if cmd_el is not None:
                expanded_cmd = _resolve_tokens(cmd_el.text or "", tokens)
                self.assertNotRegex(
                    expanded_cmd,
                    r"@[A-Z0-9_]+@",
                    f"Unresolved @TOKEN@ in <command> of {filename}",
                )
                self.assertIn("ccnv ", expanded_cmd)

            inputs_el = root.find("inputs")
            outputs_el = root.find("outputs")
            if inputs_el is not None and outputs_el is not None:
                section_fields = _collect_section_fields(inputs_el)
                for filter_el in outputs_el.findall(".//filter"):
                    expr = filter_el.text or ""
                    for sec_name, key_name in re.findall(
                        r"([A-Za-z0-9_]+)\['([A-Za-z0-9_]+)'\]", expr
                    ):
                        self.assertIn(
                            sec_name,
                            section_fields,
                            f"Filter references missing section {sec_name!r} in {filename}",
                        )
                        self.assertIn(
                            key_name,
                            section_fields[sec_name],
                            f"Filter references missing param {sec_name}[{key_name!r}] in {filename}",
                        )

    def test_bidirectional_cli_flag_parity(self) -> None:
        tokens, xml_macros = _load_macros()
        for filename, subcmd in _TOOL_SUBCOMMANDS.items():
            cli_flags = _cli_long_options(subcmd)
            root = ET.parse(_GALAXY_DIR / filename).getroot()
            _expand_macros_in_tree(root, xml_macros)

            cmd_el = root.find("command")
            inputs_el = root.find("inputs")
            if cmd_el is None or inputs_el is None:
                self.fail(f"Expected command and inputs elements in {filename}")
            expanded_cmd = _resolve_tokens(cmd_el.text or "", tokens)

            xml_flags_in_cmd: set[str] = set()
            for tok in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", expanded_cmd):
                xml_flags_in_cmd.add(tok)

            xml_bool_flags: set[str] = set()
            for param_el in inputs_el.iter("param"):
                for attr in ("truevalue", "falsevalue"):
                    val = param_el.attrib.get(attr, "").strip()
                    if val.startswith("--"):
                        xml_bool_flags.add(val)

            wrapped_flags = xml_flags_in_cmd | xml_bool_flags
            missing = sorted(cli_flags - wrapped_flags)
            extra = sorted(wrapped_flags - cli_flags)
            self.assertEqual(
                missing,
                [],
                f"CLI flags for `ccnv {subcmd}` missing in {filename}: {missing}",
            )
            self.assertEqual(
                extra,
                [],
                f"Unknown CLI flags found in {filename} for `ccnv {subcmd}`: {extra}",
            )

    def test_galaxy_dat_staging_and_declared_outputs_end_to_end(self) -> None:
        extract_root = ET.parse(_GALAXY_DIR / "ccnv_extract.xml").getroot()
        ref_root = ET.parse(_GALAXY_DIR / "ccnv_reference.xml").getroot()
        ref_qc_root = ET.parse(_GALAXY_DIR / "ccnv_ref_qc.xml").getroot()
        predict_root = ET.parse(_GALAXY_DIR / "ccnv_predict.xml").getroot()

        extract_outputs = [
            d.attrib["from_work_dir"] for d in extract_root.findall("./outputs/data")
        ]
        ref_outputs = [
            d.attrib["from_work_dir"] for d in ref_root.findall("./outputs/data")
        ]
        ref_qc_outputs = [
            d.attrib["from_work_dir"] for d in ref_qc_root.findall("./outputs/data")
        ]
        predict_outputs = [
            d.attrib["from_work_dir"] for d in predict_root.findall("./outputs/data")
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            galaxy_store = root / "galaxy_datasets"
            galaxy_store.mkdir()

            source_bam = galaxy_store / "dataset_bam.bam"
            _write_paired_bam(source_bam, _CONTIGS)
            dat_bam = galaxy_store / "dataset_001.dat"
            dat_bai = galaxy_store / "dataset_002.dat"
            shutil.copy2(source_bam, dat_bam)
            shutil.copy2(Path(f"{source_bam}.bai"), dat_bai)

            extract_work = root / "work_extract"
            extract_work.mkdir()
            (extract_work / "input.bam").symlink_to(dat_bam)
            (extract_work / "input.bam.bai").symlink_to(dat_bai)
            proc_ext = _run_ccnv(
                extract_work,
                "extract",
                "input.bam",
                "--prefix",
                "out_sample",
                "--bin-size",
                str(_BIN_SIZE),
                "--threads",
                "1",
            )
            self.assertEqual(proc_ext.returncode, 0, msg=proc_ext.stderr)
            for out_rel in extract_outputs:
                self.assertTrue(
                    (extract_work / out_rel).is_file(),
                    f"Missing extract output {out_rel}",
                )

            rng = np.random.default_rng(7)
            cohort_dats: list[Path] = []
            for idx in range(10):
                dat_sample = galaxy_store / f"cohort_{idx}.dat"
                counts: list[int] = []
                contigs: list[str] = []
                starts: list[int] = []
                ends: list[int] = []
                is_male = idx >= 5
                for cname, clen in _CONTIGS:
                    n_bins = clen // _BIN_SIZE
                    base = (
                        100
                        if cname in {"chr1", "chr2"}
                        else (
                            50
                            if (cname == "chrX" and is_male)
                            else (100 if cname == "chrX" else (50 if is_male else 0))
                        )
                    )
                    vals = np.maximum(
                        1 if base > 0 else 0, base + rng.integers(-6, 7, size=n_bins)
                    )
                    for b in range(n_bins):
                        contigs.append(cname)
                        starts.append(b * _BIN_SIZE)
                        ends.append((b + 1) * _BIN_SIZE)
                        counts.append(int(vals[b]))
                write_synthetic_sample(
                    dat_sample,
                    counts,
                    contigs=contigs,
                    starts=starts,
                    ends=ends,
                    contig_lengths=dict(_CONTIGS),
                    bin_size=_BIN_SIZE,
                )
                cohort_dats.append(dat_sample)

            ref_work = root / "work_reference"
            (ref_work / "samples").mkdir(parents=True)
            entries_collection: list[dict[str, object]] = []
            entries_annotated: list[dict[str, object]] = []
            for idx, dat_path in enumerate(cohort_dats):
                staged = ref_work / "samples" / f"sample_{idx}.ccnv"
                staged.symlink_to(dat_path)
                rel_s = f"samples/sample_{idx}.ccnv"
                entries_collection.append({"filepath": rel_s})
                entries_annotated.append(
                    {
                        "filepath": rel_s,
                        "sex": "F" if idx < 5 else "M",
                        "ff": None if idx < 5 else 0.12,
                    }
                )

            annotated_json = ref_work / "samplesheet_annotated.json"
            annotated_json.write_text(json.dumps(entries_annotated), encoding="utf-8")
            loaded_ann = load_samplesheet(annotated_json)
            self.assertEqual(len(loaded_ann), 10)
            self.assertEqual(loaded_ann[0].sex, "F")
            self.assertIsNone(loaded_ann[0].ff)
            self.assertEqual(loaded_ann[5].sex, "M")
            self.assertAlmostEqual(loaded_ann[5].ff or 0.0, 0.12)

            collection_json = ref_work / "samplesheet.json"
            collection_json.write_text(json.dumps(entries_collection), encoding="utf-8")
            loaded_col = load_samplesheet(collection_json)
            self.assertEqual(len(loaded_col), 10)

            proc_ref = _run_ccnv(
                ref_work,
                "reference",
                "--samplesheet",
                "samplesheet.json",
                "--prefix",
                "out_ref",
                "--bin-size",
                str(_BIN_SIZE),
                "--min-cohort-size",
                "10",
                "--min-sex-group-size",
                "2",
                "--threads",
                "1",
            )
            self.assertEqual(proc_ref.returncode, 0, msg=proc_ref.stderr)
            for out_rel in ref_outputs:
                self.assertTrue(
                    (ref_work / out_rel).is_file(),
                    f"Missing reference output {out_rel}",
                )

            dat_ref = galaxy_store / "dataset_ref.dat"
            shutil.copy2(ref_work / "out_ref.ccnv", dat_ref)

            ref_qc_work = root / "work_ref_qc"
            ref_qc_work.mkdir()
            (ref_qc_work / "reference.ccnv").symlink_to(dat_ref)
            proc_qc = _run_ccnv(
                ref_qc_work,
                "ref-qc",
                "reference.ccnv",
                "--prefix",
                "out_ref",
            )
            self.assertEqual(proc_qc.returncode, 0, msg=proc_qc.stderr)
            for out_rel in ref_qc_outputs:
                qc_file = ref_qc_work / out_rel
                self.assertTrue(qc_file.is_file(), f"Missing ref-qc output {out_rel}")
                payload = json.loads(qc_file.read_text(encoding="utf-8"))
                self.assertIn(payload["overall_verdict"], {"PASS", "WARN", "FAIL"})
                self.assertEqual(payload["parameters"]["bin_size"], _BIN_SIZE)

            dat_query = galaxy_store / "dataset_query.dat"
            shutil.copy2(cohort_dats[0], dat_query)
            pred_work = root / "work_predict"
            pred_work.mkdir()
            (pred_work / "sample.ccnv").symlink_to(dat_query)
            (pred_work / "reference.ccnv").symlink_to(dat_ref)
            proc_pred = _run_ccnv(
                pred_work,
                "predict",
                "sample.ccnv",
                "reference.ccnv",
                "--prefix",
                "out_pred",
                "--mode",
                "germline",
                "--sex",
                "autosomal",
                "--plot",
                "--vcf",
                "--bins-vcf",
                "--threads",
                "1",
            )
            self.assertEqual(proc_pred.returncode, 0, msg=proc_pred.stderr)

            for stem in ("bins", "segments", "calls"):
                gz_path = pred_work / f"out_pred_{stem}.bed.gz"
                bed_path = pred_work / f"out_pred_{stem}.bed"
                with gzip.open(gz_path, "rb") as src, bed_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

            for out_rel in predict_outputs:
                out_file = pred_work / out_rel
                self.assertTrue(
                    out_file.is_file() and out_file.stat().st_size > 0,
                    f"Missing or empty predict output {out_rel}",
                )
            self.assertTrue(
                (pred_work / "out_pred_segments.bed")
                .read_text(encoding="utf-8")
                .startswith("#contig\tstart\tend\t")
            )
            with gzip.open(
                pred_work / "out_pred_segments.vcf.gz", "rt", encoding="utf-8"
            ) as vcf_h:
                self.assertTrue(vcf_h.readline().startswith("##fileformat=VCFv4.2"))
            pred_qc = json.loads(
                (pred_work / "out_pred_qc.json").read_text(encoding="utf-8")
            )
            self.assertIn("sex", pred_qc)
            self.assertIn(
                "<html",
                (pred_work / "out_pred_plot.html").read_text(encoding="utf-8").lower(),
            )


if __name__ == "__main__":
    unittest.main()
