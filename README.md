# AnthroFold on Amazon SageMaker

This repository contains lightweight notebooks for deploying and invoking an AnthroFold folding model package on Amazon SageMaker async inference, plus downstream scoring of returned predictions.

The model package and SageMaker endpoint are hosted in `us-east-1`. You can run the notebooks from SageMaker Studio or another Jupyter environment in a different region, but SageMaker API calls in the notebooks target `us-east-1`. Using an S3 bucket outside `us-east-1` can incur cross-region data transfer charges.

## End-to-End Workflow

The six notebooks are intended to be run in order:

1. **`1-deploy-endpoint.ipynb`** — Subscribe to the AnthroFold AWS Marketplace listing, then stand up an async endpoint from its model package ARN. Documents the input CSV format and writes `endpoint_name.txt` for the other notebooks. First deploy takes ~1 hour while the container loads the bundled MSA databases (see [Cold Start](#cold-start)).
2. **`2-invoke-endpoint.ipynb`** — Read your input CSV, batch the jobs, submit them to the endpoint, and save returned mmCIFs + confidence JSONs under `outputs/invoke_<timestamp>/`. The default points at `examples/structure_determination_input.csv`.
3. **`4-score-dockq.ipynb`** — *(Optional, requires ground-truth CIFs.)* Score each returned structure against the experimental complex with DockQ. Outputs one row per prediction.
4. **`5-predict-epitope.ipynb`** — Extract the predicted epitope (antigen residues in heavy-atom contact with the binder) for each returned structure. No ground truth required.
5. **`6-binding-score.ipynb`** — *(Optional.)* Rank candidate binders per antigen using the model's three binding-score signals (consensus percentile rank). No ground truth required.
6. **`3-cleanup-endpoint.ipynb`** — Delete the endpoint, endpoint config, and model when you are done. Stops incurring per-instance-hour charges.

Notebooks 4, 5, and 6 are independent and can be run in any order; they consume the `cifs/` and `confidence/` subdirs written by notebook 2.

## Recommended Instance

The async endpoint runs on one of the listing's supported instance types: `ml.p5.48xlarge` (8× H100), `ml.p4de.24xlarge` (8× A100 80 GB), or `ml.p5e.48xlarge` (8× H200). Endpoint quota for these is often 0 by default — request an increase before deploying.

## Cold Start

Initial endpoint startup takes about 1 hour while the container loads the bundled MSA databases. Per-example runtime is then 5-10 minutes depending on complex size. If deployment fails due to insufficient capacity, the deploy cell cleans up and exits — re-run it to try again.

## Training and Template Cutoff

The model and its template database use a release-date cutoff of **2021-09-30**, matching the AlphaFold 3 training and inference protocol. Templates from PDB entries released after this date are filtered out during inference.

## Notebooks

- `1-deploy-endpoint.ipynb`: deploy a SageMaker async endpoint from a ModelPackage ARN. Writes `endpoint_name.txt` for the other notebooks to read. Documents the input CSV format.
- `2-invoke-endpoint.ipynb`: read an input CSV, split into async batches, invoke the endpoint, save mmCIF output and confidence summaries, and optionally visualize the first structure. The default batch size is 4 jobs per request.
- `3-cleanup-endpoint.ipynb`: delete the endpoint, endpoint config, and model when you are finished.
- `4-score-dockq.ipynb`: score returned predictions against experimental ground-truth structures with DockQ, one row per prediction.
- `5-predict-epitope.ipynb`: extract predicted epitope residues from returned complex structures (heavy-atom contact, no ground truth required).
- `6-binding-score.ipynb`: rank candidate binders per antigen by consensus percentile rank over the model's three binding-score signals.

Supporting Python helpers live in `src/`.

## Inputs and Outputs

Inputs are six-column CSVs documented in `1-deploy-endpoint.ipynb`. The
recommended sixth column (`exp_structure`) carries the ground-truth structure
filename so predictions can be matched back during DockQ scoring. See
`examples/structure_determination_input.csv` for a four-row example.

The `antigen_seq` and `binder_seq` columns may each list several chains separated by `/` (e.g. a multi-chain antigen or a VH/VL antibody). DockQ scoring (notebook 4) and binding-score (notebook 6) resolve and score every antigen chain (homo- and hetero-multimer antigens included); epitope prediction (notebook 5) reports the first antigen chain.

Successful async inference returns a JSON object with a `predictions` list. Each prediction includes `cif_content` (an mmCIF string) and a `confidence.summary` object containing `iptm`, `plddt`, `ptm`, per-chain metrics, and three `binding_score_1/2/3` signals (`binding_score_2` requires an antibody binder).

## Setup

Install the dependencies in the notebook kernel environment using your preferred environment manager. `requirements.txt` lists the packages used by the notebooks.

In SageMaker Studio, credentials come from the execution role attached to the Studio domain, user profile, or space. If running locally, configure AWS credentials with your normal AWS CLI/profile setup before opening the notebooks.
