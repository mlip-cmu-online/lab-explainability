# Lab 10: Explainability with Alibi & SHAP

This repository contains the notebook-first starter for Lab 10. You will use:

- partial-dependence variance (PDV) as a global feature-importance heuristic;
- partial-dependence (PD) curves to inspect average model response;
- an image anchor to inspect one image classifier prediction; and
- SHAP for local attributions and a global aggregation of their absolute values.

These are different explanation objects. They can describe model behavior, but they do not by themselves establish causation, discrimination, or fairness.

## Recommended: Colab / Notebook-First

1. Open `explainability.ipynb` in Google Colab from this repository, or upload the notebook, `requirements.txt`, and `BostonHousing.csv` to the same Colab session.
2. Choose a CPU runtime.
3. Run the installation cell once. If Colab asks for a runtime restart, restart and continue at the version-check cell.
4. Run all remaining cells from top to bottom.
5. Save or download an executed copy with your plots, TODO code, and short-answer cells completed.

The pinned environment targets Python 3.12. The notebook uses only the included CSV and scikit-learn's bundled digits data; it does not need to download a pretrained model or external dataset.

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

Submit one executed notebook containing:

- the completed PDV TODO and the requested PD plots;
- answers to Q1–Q3 with the requested PDV/PD plots;
- the image-anchor result and a brief precision/coverage interpretation;
- Q4 comparing mean absolute SHAP importance with PDV (not with the image-anchor task); and
- Q5 stating an audit hypothesis and a concrete follow-up check without treating attribution as proof of a fairness problem.

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
