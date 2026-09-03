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
