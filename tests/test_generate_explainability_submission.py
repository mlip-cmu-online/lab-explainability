import base64
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate-explainability-submission.py"
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode("ascii")
PLOT_TAGS = (
    "mlip-pdv-output",
    "mlip-pd-one-output",
    "mlip-pd-pair-output",
    "mlip-anchor-output",
    "mlip-shap-global-output",
    "mlip-shap-local-output",
)


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def complete_evidence() -> dict[str, Any]:
    names = [
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
    ]
    pdv_values = [round(0.1 + index / 10, 2) for index in range(len(names))]
    local_values = [round(-0.6 + index / 10, 2) for index in range(len(names))]
    global_values = [round(0.2 + index / 10, 2) for index in range(len(names))]
    return {
        "schema_version": "1.0",
        "pdv": {
            "feature_names": names,
            "importance": pdv_values,
        },
        "pd": {
            "one_feature_names": ["rm", "lstat"],
            "pair": ["rm", "lstat"],
        },
        "anchor": {
            "predicted_label": 7,
            "segments": [1, 4, 5],
            "precision": 0.91,
            "coverage": 0.27,
        },
        "shap": {
            "feature_names": names,
            "values_shape": [102, 13],
            "background_shape": [100, 13],
            "local_index": 5,
            "local_values": local_values,
            "global_mean_absolute": global_values,
        },
    }


def write_notebook(
    path: Path,
    evidence: dict[str, Any],
    *,
    missing_plot_tag: str | None = None,
    source: str = "x = 1",
) -> None:
    cells: list[dict[str, Any]] = [
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [source],
        }
    ]
    for index, tag in enumerate(PLOT_TAGS, start=2):
        outputs = [] if tag == missing_plot_tag else [
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/png": PNG, "text/plain": "<Figure>"},
            }
        ]
        cells.append(
            {
                "cell_type": "code",
                "execution_count": index,
                "metadata": {"tags": [tag]},
                "outputs": outputs,
                "source": ["plot()"],
            }
        )
    for question in range(1, 7):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {"tags": ["mlip-required-answer"]},
                "source": [
                    f"**Question {question}:** Explain the result.\n\n"
                    f"_Your answer:_ This is a complete saved response for question {question}."
                ],
            }
        )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": 8,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [
                        "MLIP_SUBMISSION_EVIDENCE_V1\n",
                        json.dumps(evidence) + "\n",
                        "MLIP_SUBMISSION_EVIDENCE_END\n",
                    ],
                }
            ],
            "source": ["print('submission evidence')"],
        }
    )
    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


class GenerateExplainabilitySubmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.notebook = self.root / "explainability.ipynb"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def generate(self) -> subprocess.CompletedProcess[str]:
        return run(
            "python3",
            str(SCRIPT),
            "--learner",
            "Test Learner",
            "--notebook",
            str(self.notebook),
            cwd=self.root,
        )

    def test_generates_complete_self_contained_report_and_manifest(self) -> None:
        write_notebook(self.notebook, complete_evidence())

        result = self.generate()

        self.assertEqual(result.returncode, 0, result.stderr)
        report = (self.root / "submission" / "explainability-report.html").read_text(
            encoding="utf-8"
        )
        manifest = json.loads(
            (self.root / "submission" / "explainability-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(manifest["complete"])
        self.assertTrue(all(item["status"] == "present" for item in manifest["checks"]))
        self.assertEqual(report.count("data:image/png;base64,"), 6)
        self.assertIn('download="explainability.ipynb"', report)
        self.assertIn("Print → Save as PDF", report)
        self.assertIn("lstat", report)
        self.assertEqual(manifest["evidence"]["anchor"]["predicted_label"], 7)

    def test_writes_incomplete_outputs_for_missing_plot_and_bad_shap_shape(self) -> None:
        evidence = complete_evidence()
        evidence["shap"]["values_shape"] = [102, 12]
        write_notebook(
            self.notebook,
            evidence,
            missing_plot_tag="mlip-shap-local-output",
        )

        result = self.generate()

        self.assertEqual(result.returncode, 1)
        manifest = json.loads(
            (self.root / "submission" / "explainability-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        checks = {item["name"]: item["status"] for item in manifest["checks"]}
        self.assertEqual(checks["saved_shap_values"], "missing")
        self.assertEqual(checks["local_shap_plot"], "missing")

    def test_flags_secret_and_does_not_embed_learner_notebook(self) -> None:
        secret = "sk_" + "a" * 30
        write_notebook(
            self.notebook,
            complete_evidence(),
            source=f'api_key = "{secret}"',
        )

        result = self.generate()

        self.assertEqual(result.returncode, 1)
        report = (self.root / "submission" / "explainability-report.html").read_text(
            encoding="utf-8"
        )
        manifest_text = (
            self.root / "submission" / "explainability-manifest.json"
        ).read_text(encoding="utf-8")
        self.assertNotIn(secret, report + manifest_text)
        self.assertIn("embedding was withheld", report)
        manifest = json.loads(manifest_text)
        checks = {item["name"]: item["status"] for item in manifest["checks"]}
        self.assertEqual(checks["obvious_credential_leakage"], "missing")

    def test_rejects_missing_input_without_creating_outputs(self) -> None:
        result = self.generate()

        self.assertEqual(result.returncode, 2)
        self.assertIn("Notebook not found", result.stderr)
        self.assertFalse((self.root / "submission").exists())


if __name__ == "__main__":
    unittest.main()
