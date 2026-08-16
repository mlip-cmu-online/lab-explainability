#!/usr/bin/env python3
"""Generate the Lab 10 report and manifest from a saved executed notebook."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import html
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


REPORT_NAME = "explainability-report.html"
MANIFEST_NAME = "explainability-manifest.json"
EVIDENCE_START = "MLIP_SUBMISSION_EVIDENCE_V1"
EVIDENCE_END = "MLIP_SUBMISSION_EVIDENCE_END"
CHECKER_VERSION = "1.0"
REQUIRED_FEATURE_NAMES = (
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "b",
    "lstat",
)
PLOT_TAGS = {
    "pdv_plot": "mlip-pdv-output",
    "one_feature_pd_plot": "mlip-pd-one-output",
    "two_feature_pd_plot": "mlip-pd-pair-output",
    "anchor_plot": "mlip-anchor-output",
    "global_shap_plot": "mlip-shap-global-output",
    "local_shap_plot": "mlip-shap-local-output",
}
PLACEHOLDERS = {"", "...", "todo", "tbd", "your answer", "your answer here"}
SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\b"
        r"\s*[:=]\s*['\"]([^'\"\s]{8,})['\"]"
    ),
    re.compile(r"\b(?:ghp|github_pat|sk|xox[baprs]|hf)_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
    re.compile(r"(?i)https?://[^\s/@:]+:[^\s/@]+@"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Lab 10 self-contained HTML report and JSON manifest."
    )
    parser.add_argument("--learner", required=True, help="Learner name shown in the report")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("explainability.ipynb"),
        help="Complete executed notebook (default: explainability.ipynb)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submission"),
        help="Output directory (default: submission)",
    )
    return parser.parse_args()


def check(
    name: str, present: bool, location: str, identifier: str, detail: str
) -> dict[str, str]:
    return {
        "name": name,
        "status": "present" if present else "missing",
        "location": location,
        "identifier": identifier,
        "detail": detail,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def joined(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value or "")


def output_text(output: dict[str, Any]) -> str:
    if output.get("output_type") == "stream":
        return joined(output.get("text"))
    if output.get("output_type") in {"display_data", "execute_result"}:
        return joined(output.get("data", {}).get("text/plain"))
    if output.get("output_type") == "error":
        return "\n".join(output.get("traceback", []))
    return ""


def extract_submission_evidence(
    notebook: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], str]:
    transcript = "\n".join(
        output_text(output)
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
        if isinstance(output, dict)
    )
    matches = re.findall(
        rf"{re.escape(EVIDENCE_START)}\s*(.*?)\s*{re.escape(EVIDENCE_END)}",
        transcript,
        flags=re.DOTALL,
    )
    if not matches:
        return None, "Run the final Save submission evidence cell and save the notebook."
    try:
        value = json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        return None, f"The final structured evidence JSON is invalid: {exc.msg}."
    if not isinstance(value, dict):
        return None, "The final structured evidence must be a JSON object."
    return value, "Found and parsed the final structured evidence output."


def cell_tags(cell: dict[str, Any]) -> set[str]:
    metadata = cell.get("metadata", {})
    tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
    return {str(tag) for tag in tags} if isinstance(tags, list) else set()


def png_from_tag(notebook: dict[str, Any], tag: str) -> Optional[str]:
    for cell in notebook.get("cells", []):
        if tag not in cell_tags(cell):
            continue
        for output in reversed(cell.get("outputs", [])):
            if not isinstance(output, dict):
                continue
            raw = output.get("data", {}).get("image/png")
            encoded = "".join(raw) if isinstance(raw, list) else str(raw or "")
            compact = re.sub(r"\s+", "", encoded)
            if not compact:
                continue
            try:
                decoded = base64.b64decode(compact, validate=True)
            except (binascii.Error, ValueError):
                continue
            if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                return compact
    return None


def required_answers(notebook: dict[str, Any]) -> tuple[bool, int, int]:
    answer_cells = [
        cell
        for cell in notebook.get("cells", [])
        if "mlip-required-answer" in cell_tags(cell)
    ]
    completed = 0
    for cell in answer_cells:
        source = joined(cell.get("source"))
        if "_Your answer:_" not in source:
            continue
        response = source.split("_Your answer:_", 1)[1]
        response = re.split(r"\n#{1,6}\s", response, maxsplit=1)[0]
        cleaned = response.strip().strip("_*").strip()
        if cleaned.casefold() not in PLACEHOLDERS and len(cleaned) >= 12:
            completed += 1
    return len(answer_cells) == 6 and completed == 6, completed, len(answer_cells)


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def valid_names(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
        and len(set(value)) == len(value)
    )


def valid_nonnegative_values(value: Any, expected: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected
        and all(is_number(item) and float(item) >= 0 for item in value)
    )


def validate_pdv(evidence: dict[str, Any]) -> tuple[bool, list[str], list[Any], str]:
    pdv = evidence.get("pdv")
    if not isinstance(pdv, dict):
        return False, [], [], "Expected a pdv object."
    names = pdv.get("feature_names")
    values = pdv.get("importance")
    valid = (
        names == list(REQUIRED_FEATURE_NAMES)
        and valid_nonnegative_values(values, len(REQUIRED_FEATURE_NAMES))
    )
    detail = (
        f"Found {len(names)} labeled finite PDV values."
        if valid
        else "Save one finite, nonnegative PDV importance value for each of the 13 expected features in dataset order."
    )
    return valid, names if valid_names(names) else [], values if isinstance(values, list) else [], detail


def validate_pd(evidence: dict[str, Any], feature_names: list[str]) -> tuple[bool, str]:
    pd = evidence.get("pd")
    if not isinstance(pd, dict):
        return False, "Expected a pd object."
    one = pd.get("one_feature_names")
    pair = pd.get("pair")
    valid = (
        isinstance(one, list)
        and len(one) >= 1
        and len(set(one)) == len(one)
        and all(name in feature_names for name in one)
        and isinstance(pair, list)
        and len(pair) == 2
        and pair[0] != pair[1]
        and all(name in feature_names for name in pair)
    )
    return valid, (
        f"Found one-feature PD for {', '.join(one)} and a two-feature PD pair {pair[0]} × {pair[1]}."
        if valid
        else "Save the selected one-feature PD names and a distinct two-feature PD pair."
    )


def validate_anchor(evidence: dict[str, Any]) -> tuple[bool, str]:
    anchor = evidence.get("anchor")
    if not isinstance(anchor, dict):
        return False, "Expected an anchor object."
    precision = anchor.get("precision")
    coverage = anchor.get("coverage")
    segments = anchor.get("segments")
    label = anchor.get("predicted_label")
    valid = (
        is_number(precision)
        and 0 <= float(precision) <= 1
        and is_number(coverage)
        and 0 <= float(coverage) <= 1
        and isinstance(segments, list)
        and bool(segments)
        and len(set(segments)) == len(segments)
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in segments)
        and isinstance(label, int)
        and not isinstance(label, bool)
        and 0 <= label <= 9
    )
    return valid, (
        f"Found label {label}, {len(segments)} retained segments, precision {float(precision):.3f}, and coverage {float(coverage):.3f}."
        if valid
        else "Save the predicted digit, retained segments, and precision/coverage values from 0 to 1."
    )


def validate_shap(evidence: dict[str, Any], feature_names: list[str]) -> tuple[bool, str]:
    shap = evidence.get("shap")
    if not isinstance(shap, dict):
        return False, "Expected a shap object."
    names = shap.get("feature_names")
    values_shape = shap.get("values_shape")
    background_shape = shap.get("background_shape")
    global_values = shap.get("global_mean_absolute")
    local_values = shap.get("local_values")
    local_index = shap.get("local_index")
    shape_valid = (
        isinstance(values_shape, list)
        and len(values_shape) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in values_shape)
    )
    valid = (
        names == feature_names
        and shape_valid
        and values_shape[1] == len(feature_names)
        and background_shape == [100, len(feature_names)]
        and valid_nonnegative_values(global_values, len(feature_names))
        and isinstance(local_values, list)
        and len(local_values) == len(feature_names)
        and all(is_number(item) for item in local_values)
        and isinstance(local_index, int)
        and not isinstance(local_index, bool)
        and 0 <= local_index < values_shape[0]
    )
    return valid, (
        f"Found {values_shape[0]} × {values_shape[1]} local attributions, a 100-row background, and matching global values."
        if valid
        else "Save matching local/global SHAP values, feature names, shapes, background size, and local row index."
    )


def has_obvious_secret(notebook: dict[str, Any]) -> bool:
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        chunks.append(joined(cell.get("source")))
        chunks.extend(
            output_text(output)
            for output in cell.get("outputs", [])
            if isinstance(output, dict)
        )
    searchable = "\n".join(chunks)
    return any(pattern.search(searchable) for pattern in SECRET_PATTERNS)


def importance_table(names: list[str], pdv: list[Any], shap_values: list[Any]) -> str:
    rows = []
    for index, name in enumerate(names):
        pdv_value = pdv[index] if index < len(pdv) and is_number(pdv[index]) else None
        shap_value = (
            shap_values[index]
            if index < len(shap_values) and is_number(shap_values[index])
            else None
        )
        rows.append((name, pdv_value, shap_value))
    rows.sort(key=lambda row: float(row[1]) if row[1] is not None else -1, reverse=True)
    body = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{format(float(pdv_value), '.6g') if pdv_value is not None else 'missing'}</td>"
        f"<td>{format(float(shap_value), '.6g') if shap_value is not None else 'missing'}</td>"
        "</tr>"
        for name, pdv_value, shap_value in rows
    )
    return (
        "<table><thead><tr><th>Feature</th><th>PDV importance</th>"
        "<th>Mean absolute SHAP</th></tr></thead>"
        f"<tbody>{body or '<tr><td colspan=\"3\">No saved values found.</td></tr>'}</tbody></table>"
    )


def figure(encoded: Optional[str], alternative: str) -> str:
    if encoded is None:
        return f'<p class="missing">Missing saved {html.escape(alternative)}.</p>'
    return (
        f'<img class="figure" src="data:image/png;base64,{encoded}" '
        f'alt="{html.escape(alternative)}">'
    )


def render_report(
    learner: str,
    generated_at: str,
    complete: bool,
    checks: list[dict[str, str]],
    notebook_path: Path,
    notebook_hash: str,
    evidence: dict[str, Any],
    images: dict[str, Optional[str]],
    safe_to_embed: bool,
) -> str:
    check_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td class=\"{item['status']}\">{item['status']}</td>"
        f"<td>{html.escape(item['identifier'])}</td>"
        f"<td>{html.escape(item['detail'])}</td>"
        "</tr>"
        for item in checks
    )
    pdv = evidence.get("pdv", {}) if isinstance(evidence.get("pdv"), dict) else {}
    shap = evidence.get("shap", {}) if isinstance(evidence.get("shap"), dict) else {}
    anchor = evidence.get("anchor", {}) if isinstance(evidence.get("anchor"), dict) else {}
    names = pdv.get("feature_names", []) if isinstance(pdv.get("feature_names"), list) else []
    pdv_values = pdv.get("importance", []) if isinstance(pdv.get("importance"), list) else []
    shap_values = shap.get("global_mean_absolute", []) if isinstance(shap.get("global_mean_absolute"), list) else []
    notebook_link = (
        '<a download="explainability.ipynb" href="data:application/x-ipynb+json;base64,'
        + base64.b64encode(notebook_path.read_bytes()).decode("ascii")
        + '">Download the complete executed notebook embedded in this report</a>.'
        if safe_to_embed
        else "Notebook embedding was withheld because a possible plaintext credential was detected."
    )
    safety_text = (
        "No obvious plaintext credential pattern was found. The checker cannot inspect text rendered inside images, so review the report before uploading it."
        if safe_to_embed
        else "A possible plaintext credential was found. Remove and revoke it, then regenerate the report."
    )
    status = "complete" if complete else "incomplete"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lab 10 Explainability Submission Report</title>
  <style>
    body {{ color: #17202a; font: 16px/1.45 system-ui, sans-serif; margin: 2rem auto; max-width: 78rem; padding: 0 1rem; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; margin: .75rem 0 1.5rem; width: 100%; }}
    th, td {{ border: 1px solid #bcccdc; padding: .55rem; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    code {{ overflow-wrap: anywhere; }}
    .present, .complete {{ color: #176b3a; font-weight: 700; }}
    .missing, .incomplete {{ color: #a61b1b; font-weight: 700; }}
    .figure {{ border: 1px solid #bcccdc; height: auto; margin: .5rem 0 1.5rem; max-width: 100%; }}
    .note {{ background: #fffbea; border-left: .3rem solid #d69e2e; padding: .7rem 1rem; }}
    @media print {{ body {{ margin: 0; max-width: none; }} .print-note {{ display: none; }} }}
  </style>
</head>
<body>
  <h1>Lab 10: Explainability with Alibi &amp; SHAP</h1>
  <table>
    <tr><th>Learner</th><td>{html.escape(learner)}</td></tr>
    <tr><th>Generated</th><td>{html.escape(generated_at)}</td></tr>
    <tr><th>Overall completeness</th><td class="{status}">{status}</td></tr>
    <tr><th>Notebook SHA-256</th><td><code>{notebook_hash}</code></td></tr>
    <tr><th>Checker version</th><td>{CHECKER_VERSION}</td></tr>
  </table>

  <h2>Completeness</h2>
  <table>
    <thead><tr><th>Required evidence</th><th>Status</th><th>Identifier</th><th>Detail</th></tr></thead>
    <tbody>{check_rows}</tbody>
  </table>

  <h2>PDV and SHAP summary</h2>
  <p>These saved quantities summarize different explanation objects; their interpretation belongs in the Canvas spot check.</p>
  {importance_table(names, pdv_values, shap_values)}
  {figure(images.get('pdv_plot'), 'PDV plot')}

  <h2>Partial-dependence plots</h2>
  {figure(images.get('one_feature_pd_plot'), 'one-feature partial-dependence plot')}
  {figure(images.get('two_feature_pd_plot'), 'two-feature partial-dependence plot')}

  <h2>Image anchor</h2>
  <table>
    <tr><th>Predicted label</th><td>{html.escape(str(anchor.get('predicted_label', 'missing')))}</td></tr>
    <tr><th>Retained segments</th><td>{html.escape(str(anchor.get('segments', 'missing')))}</td></tr>
    <tr><th>Precision</th><td>{html.escape(str(anchor.get('precision', 'missing')))}</td></tr>
    <tr><th>Coverage</th><td>{html.escape(str(anchor.get('coverage', 'missing')))}</td></tr>
  </table>
  {figure(images.get('anchor_plot'), 'image-anchor plot')}

  <h2>SHAP outputs</h2>
  {figure(images.get('global_shap_plot'), 'global mean absolute SHAP plot')}
  {figure(images.get('local_shap_plot'), 'local SHAP waterfall plot')}

  <h2>Raw audit evidence</h2>
  <p>{notebook_link}</p>
  <p class="print-note">This HTML file is self-contained. To submit PDF instead, open it in a browser and use <strong>Print → Save as PDF</strong>; include background graphics.</p>
  <p class="note">This checker verifies objective completeness and internal consistency only. Answer the explanation-comparison and local-explanation/audit spot checks separately in Canvas.</p>

  <h2>Safety check</h2>
  <p>{safety_text}</p>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    if not args.notebook.is_file():
        print(f"Notebook not found: {args.notebook}", file=sys.stderr)
        return 2
    try:
        notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read notebook JSON: {exc}", file=sys.stderr)
        return 2
    if (
        not isinstance(notebook, dict)
        or not isinstance(notebook.get("cells"), list)
        or not all(isinstance(cell, dict) for cell in notebook["cells"])
    ):
        print("Notebook JSON does not contain a cells list.", file=sys.stderr)
        return 2

    evidence, evidence_detail = extract_submission_evidence(notebook)
    evidence = evidence or {}
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    executed_count = sum(cell.get("execution_count") is not None for cell in code_cells)
    all_executed = bool(code_cells) and executed_count == len(code_cells)
    error_count = sum(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
        if isinstance(output, dict)
    )
    answers_valid, completed_answers, answer_count = required_answers(notebook)
    images = {name: png_from_tag(notebook, tag) for name, tag in PLOT_TAGS.items()}
    pdv_valid, feature_names, pdv_values, pdv_detail = validate_pdv(evidence)
    pd_valid, pd_detail = validate_pd(evidence, feature_names)
    anchor_valid, anchor_detail = validate_anchor(evidence)
    shap_valid, shap_detail = validate_shap(evidence, feature_names)
    has_secret = has_obvious_secret(notebook)

    checks = [
        check(
            "learner_name",
            bool(args.learner.strip()),
            REPORT_NAME,
            args.learner.strip() or "missing",
            "Learner name recorded." if args.learner.strip() else "Provide a nonempty learner name.",
        ),
        check(
            "executed_notebook",
            all_executed,
            "embedded notebook",
            f"{executed_count}/{len(code_cells)} code cells",
            "Every code cell has an execution count." if all_executed else "Run all cells from top to bottom and save the notebook.",
        ),
        check(
            "notebook_error_outputs",
            error_count == 0,
            "embedded notebook",
            f"{error_count} error outputs",
            "No saved error outputs were found." if error_count == 0 else "Resolve saved cell errors, rerun all cells, and save the notebook.",
        ),
        check(
            "completed_short_answers",
            answers_valid,
            "embedded notebook",
            f"{completed_answers}/6 answers",
            "All six answer fields contain non-placeholder text." if answers_valid else f"Complete all six tagged answer fields; the notebook contains {answer_count} required fields.",
        ),
        check(
            "structured_notebook_evidence",
            evidence.get("schema_version") == "1.0",
            REPORT_NAME,
            str(evidence.get("schema_version", "none")),
            evidence_detail,
        ),
        check("saved_pdv_values", pdv_valid, REPORT_NAME, f"{len(feature_names)} features", pdv_detail),
        check("saved_pd_structure", pd_valid, REPORT_NAME, "one-feature and two-feature", pd_detail),
        check("saved_anchor_result", anchor_valid, REPORT_NAME, "precision and coverage", anchor_detail),
        check("saved_shap_values", shap_valid, REPORT_NAME, "local and global", shap_detail),
    ]
    for name, tag in PLOT_TAGS.items():
        checks.append(
            check(
                name,
                images[name] is not None,
                REPORT_NAME,
                tag,
                "Found a saved PNG output in the tagged notebook cell."
                if images[name] is not None
                else "Rerun the tagged plot cell so its PNG output is saved in the notebook.",
            )
        )
    checks.append(
        check(
            "obvious_credential_leakage",
            not has_secret,
            "embedded notebook",
            "none detected" if not has_secret else "possible secret detected",
            "No obvious plaintext credential pattern was found."
            if not has_secret
            else "Remove the plaintext credential from the notebook and revoke it before submission.",
        )
    )

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    complete = all(item["status"] == "present" for item in checks)
    notebook_hash = sha256(args.notebook)
    safe_evidence = evidence if not has_secret else {}
    manifest = {
        "schema_version": "1.0",
        "checker_version": CHECKER_VERSION,
        "lab": "Lab 10: Explainability with Alibi & SHAP",
        "learner": args.learner,
        "generated_at": generated_at,
        "complete": complete,
        "report": REPORT_NAME,
        "identifiers": {"notebook_sha256": notebook_hash},
        "artifacts": {"notebook": str(args.notebook)},
        "evidence": {
            "pdv": safe_evidence.get("pdv", {}),
            "pd": safe_evidence.get("pd", {}),
            "anchor": safe_evidence.get("anchor", {}),
            "shap": safe_evidence.get("shap", {}),
        },
        "checks": checks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / REPORT_NAME
    manifest_path = args.output_dir / MANIFEST_NAME
    report_path.write_text(
        render_report(
            args.learner,
            generated_at,
            complete,
            checks,
            args.notebook,
            notebook_hash,
            safe_evidence,
            images if not has_secret else {name: None for name in images},
            not has_secret,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {manifest_path}")
    print(
        "Submission evidence is complete."
        if complete
        else "Submission evidence is incomplete; review the missing checks."
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
