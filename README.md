# Lab: Explainability with Alibi & SHAP

This repository contains the notebook-first starter for Lab 10. You will use:

- partial-dependence variance (PDV) as a global feature-importance heuristic;
- partial-dependence (PD) curves to inspect average model response;
- an image anchor to inspect one image classifier prediction; and
- SHAP for local attributions and a global aggregation of their absolute values.

These are different explanation objects. They can describe model behavior, but they do not by themselves establish causation, discrimination, or fairness.

## Recommended: Colab / Notebook-First

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlip-cmu-online/lab-explainability/blob/main/explainability.ipynb)

1. Select **Open in Colab** above. If the starter repository is still private, sign in with the GitHub account you use for the course and allow Colab to access private repositories. Alternatively, upload `explainability.ipynb` directly.
2. Download `BostonHousing.csv` from this repository and upload it when the notebook's data cell prompts you.
3. Choose a CPU runtime.
4. Run the installation cell once. If the version-check cell asks you to restart, select **Runtime > Restart session**, then continue at the version-check cell without rerunning the installation cell.
5. Run all remaining cells from top to bottom.
6. Run the final **Save submission evidence** cell, then save or download an executed copy with your plots, TODO code, and short-answer cells completed.

The pinned environment targets Python 3.12, used by the current Colab runtime. The notebook uses only the included CSV and scikit-learn's bundled digits data; it does not download a pretrained model or external dataset.

## Local Fallback

Use a fresh Python 3.12 environment from this directory:

```shell
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter notebook explainability.ipynb
```

Run the notebook from top to bottom. Other Python versions are not part of the supported pinned path.

## Deliverables

Complete one executed notebook containing:

- the completed PDV TODO and the requested PD plots;
- answers to Q1–Q3 with the requested PDV/PD plots;
- the image-anchor result and a brief precision/coverage interpretation;
- Q4 comparing mean absolute SHAP importance with PDV (not with the image-anchor task); and
- Q5 stating an audit hypothesis and a concrete follow-up check without treating attribution as proof of a fairness problem.

## Generate the Submission Report

After running the final evidence cell, save the complete executed notebook as `explainability.ipynb` in this repository. From the repository root, run:

```shell
python3 scripts/generate-explainability-submission.py \
  --learner "Your name" \
  --notebook explainability.ipynb
```

The command writes:

- `submission/explainability-report.html`, the self-contained report to upload to Canvas; and
- `submission/explainability-manifest.json`, the matching machine-readable completeness record to retain with your evidence.

Open the HTML report and correct every item marked `missing`. The report embeds the saved PDV, one- and two-feature PD, Anchor, global SHAP, and local SHAP figures, their compact numeric results, and a downloadable copy of the executed notebook. If you prefer PDF, open the report in a browser and use **Print → Save as PDF**, with background graphics enabled.

The command reads only the saved notebook. It does not run Alibi or SHAP, load data, or contact an external service. It checks execution state, error outputs, the presence of all six short answers, saved plot structure, matching feature names and result shapes, Anchor precision/coverage ranges, and obvious plaintext credential patterns. It does not assess your explanation comparison or audit hypothesis; answer those two interpretation spot checks separately in Canvas.

## Explanation Assumptions

- PD and PDV use the training split as their reference distribution. Because PD varies selected features while averaging over the remaining rows, correlated inputs can produce unrealistic combinations.
- SHAP uses a seeded sample of 100 raw training rows as its background dataset. `shap.TreeExplainer` uses `feature_perturbation='interventional'` with that empirical background; its attributions can change with a different background or feature-dependence treatment.
- The global SHAP chart is the mean absolute value of local SHAP attributions over the explained test rows. It is not permutation importance or PDV.
- The image anchor concerns a separate image classifier. Do not infer agreement or disagreement between its superpixels and the tabular model's feature rankings.

## References

1. https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependence.html
2. https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependenceVariance.html
3. https://docs.seldon.io/projects/alibi/en/stable/methods/Anchors.html
4. https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html
5. https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
