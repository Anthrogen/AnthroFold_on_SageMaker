# AnthroFold on Amazon SageMaker

This repository contains lightweight notebooks for deploying and invoking an AnthroFold folding model package on Amazon SageMaker async inference, plus downstream scoring of returned predictions.

The model package and SageMaker endpoint are hosted in `us-east-1`. You can run the notebooks from SageMaker Studio or another Jupyter environment in a different region, but SageMaker API calls in the notebooks target `us-east-1`. Using an S3 bucket outside `us-east-1` can incur cross-region data transfer charges.

## End-to-End Workflow

The five notebooks are intended to be run in order:

1. **`1-deploy-endpoint.ipynb`** — Stand up an async SageMaker endpoint from the model package ARN. Documents the input CSV format. Writes `endpoint_name.txt` for the other notebooks to read. Initial deployment takes ~1 hour while the container downloads bundled MSA databases (see [Cold Start](#cold-start)).
2. **`2-invoke-endpoint.ipynb`** — Read your input CSV, batch the jobs, submit them to the endpoint, and save returned mmCIFs + confidence JSONs under `outputs/invoke_<timestamp>/`. The default points at `examples/structure_determination_input.csv`.
3. **`4-score-dockq.ipynb`** — *(Optional, requires ground-truth CIFs.)* Score each returned structure against the experimental complex with DockQ. Outputs one row per prediction.
4. **`5-predict-epitope.ipynb`** — Extract the predicted epitope (antigen residues in heavy-atom contact with the binder) for each returned structure. No ground truth required.
5. **`3-cleanup-endpoint.ipynb`** — Delete the endpoint, endpoint config, and model when you are done. Stops incurring per-instance-hour charges.

Notebooks 4 and 5 are independent and can be run in either order; both consume the `cifs/` subdir written by notebook 2.

## Recommended Instance

For the full antibody-antigen size range of 2048 total residues, use an instance with 80 GB-class GPU memory, for example `ml.p4de.24xlarge` (8× A100 80 GB) or `ml.p5.48xlarge` (8× H100 80 GB). Verify your endpoint quota for the chosen instance type before deploying. Smaller shapes such as `ml.g5.12xlarge` (4× A10G 24 GB) can serve smaller complexes if capacity for the 80 GB instances isn't available.

## Cold Start

Initial endpoint startup typically takes about 1 hour while the container downloads the bundled MSA databases. Per-example runtime is then about 5-10 minutes depending on complex size. If deployment fails due to insufficient capacity, the deploy cell cleans up the failed resources and exits — simply re-run the deploy cell to try again.

## Training and Template Cutoff

The model and its template database use a release-date cutoff of **2021-09-30**, matching the AlphaFold 3 training and inference protocol. Templates from PDB entries released after this date are filtered out during inference.

## Notebooks

- `1-deploy-endpoint.ipynb`: deploy a SageMaker async endpoint from a ModelPackage ARN. Writes `endpoint_name.txt` for the other notebooks to read. Documents the input CSV format.
- `2-invoke-endpoint.ipynb`: read an input CSV, split into async batches, invoke the endpoint, save mmCIF output and confidence summaries, and optionally visualize the first structure. The default batch size is 4 jobs per request.
- `3-cleanup-endpoint.ipynb`: delete the endpoint, endpoint config, and model when you are finished.
- `4-score-dockq.ipynb`: score returned predictions against experimental ground-truth structures with DockQ, one row per prediction.
- `5-predict-epitope.ipynb`: extract predicted epitope residues from returned complex structures (heavy-atom contact, no ground truth required).

Supporting Python helpers live in `src/`.

## Inputs and Outputs

Inputs are six-column CSVs documented in `1-deploy-endpoint.ipynb`. The
recommended sixth column (`exp_structure`) carries the ground-truth structure
filename so predictions can be matched back during DockQ scoring. See
`examples/structure_determination_input.csv` for a four-row example.

Successful async inference returns a JSON object with a `predictions` list. Each prediction includes `cif_content` (an mmCIF string) and a `confidence.summary` object containing `iptm`, `plddt`, `ptm`, and related per-chain metrics.

## Setup

Install the dependencies in the notebook kernel environment using your preferred environment manager. `requirements.txt` lists the packages used by the notebooks.

In SageMaker Studio, credentials come from the execution role attached to the Studio domain, user profile, or space. If running locally, configure AWS credentials with your normal AWS CLI/profile setup before opening the notebooks.
