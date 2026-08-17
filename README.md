# AnthroFold on Amazon SageMaker

This repository contains lightweight notebooks for deploying and invoking an AnthroFold folding model package on Amazon SageMaker async inference, plus downstream scoring of returned predictions.

**AnthroFold folds from precomputed MSAs, and those MSAs must come from the
AnthroFold MSA pipeline** — see
[Prerequisite: precomputed MSAs](#prerequisite-precomputed-msas).

The model package and SageMaker endpoint are hosted in `us-east-1`. You can run the notebooks from SageMaker Studio or another Jupyter environment in a different region, but SageMaker API calls in the notebooks target `us-east-1`. Using an S3 bucket outside `us-east-1` can incur cross-region data transfer charges.

## End-to-End Workflow

Everything is driven by one CSV of binder and antigen sequences, one complex per
row. You author it as `complex_input.csv`; the MSA step turns that into
`complex_input_with_msas.csv`, which is the file the notebooks read. The schema is
below.

AnthroFold folds from precomputed MSAs, so preparing them is the first step. Once
your CSV carries MSA columns, the six notebooks are intended to be run in order:

1. **`1-deploy-endpoint.ipynb`** — Subscribe to the AnthroFold AWS Marketplace listing, then stand up an async endpoint from its model package ARN. Writes `endpoint_name.txt` for the other notebooks. First deploy takes about an hour (see [Cold Start](#cold-start)).
2. **`2-invoke-endpoint.ipynb`** — Read your input CSV, batch the jobs, submit them to the endpoint, and save returned mmCIFs + confidence JSONs under `outputs/invoke_<timestamp>/`. Batches are packed to fit the AWS Marketplace endpoint's 25 MB invocation-input limit rather than to a fixed job count. The default points at `examples/abag_demo/complex_input_with_msas.csv`.
3. **`4-score-dockq.ipynb`** — *(Optional, requires ground-truth mmCIFs.)* Score each returned structure against the experimental complex with DockQ. Outputs one row per prediction.
4. **`5-predict-epitope.ipynb`** — *(Optional.)* Extract the predicted epitope (antigen residues in heavy-atom contact with the binder) for each returned structure. No ground truth required.
5. **`6-binding-score.ipynb`** — *(Optional.)* Rank candidate binders per antigen using the model's three binding-score signals (consensus percentile rank). No ground truth required.
6. **`3-cleanup-endpoint.ipynb`** — Delete the endpoint, endpoint config, and model when you are done. Stops incurring per-instance-hour charges.

Notebooks 4, 5, and 6 are independent, can be run in any order, and can be
skipped; they consume the `cifs/` and `confidence/` subdirs written by notebook 2.
Supporting Python helpers live in `src/`.

## Input CSV

One row per complex — the binder and antigen sequences you want folded together.
`examples/abag_demo/complex_input.csv` is a four-row example.

Each example lives in its own directory under `examples/`, holding everything that
belongs to it:

```
examples/abag_demo/
├── complex_input.csv              # the sequences
├── complex_input_with_msas.csv    # the same rows, with MSA columns filled in
├── msas/                          # the MSAs those columns point at
└── exp_structures/                # ground-truth mmCIFs, for DockQ scoring
```

`abag_demo` ships complete, MSAs included, so notebook 2 runs against
`complex_input_with_msas.csv` as delivered. Its 8fdd row (PDB 8FDD) has a
16-residue peptide antigen with no homologues, so notebook 2 emits a warning for
it — that is expected, not a failure.

Keep your own inputs in the same shape — one directory per input set, with its
`msas/` folder inside it. `src/msa_client.py --bundle-dir <directory>` creates that
layout directly. Sequence-addressed MSA directories can also be safely reused by
several CSVs, but the generated CSV and the relative `msas/` tree remain one
handoff unit.

| Column | Required | Description |
|---|---|---|
| `antigen_seq` | yes | Antigen amino-acid sequence. Several chains are pipe-separated. |
| `binder_seq` | yes | Binder amino-acid sequence. Several chains are pipe-separated. |
| `binder_mode` | no | `1` for antibody-like binders, `0` for general protein binders. |
| `antibody_form` | no | `VH/VL` (paired heavy and light chains), `SS` (single-chain), or `None`. |
| `antigen_temp` | no | Antigen template field from the input schema, or `None`. |
| `exp_structure` | recommended | Ground-truth structure filename, used to match predictions back during DockQ scoring. Must be unique per row — downstream outputs are keyed by it. |
| `antigen_msa` | yes | One MSA directory per antigen chain, pipe-separated, in the same order as `antigen_seq`. |
| `binder_msa` | yes | One MSA directory per binder chain, pipe-separated, in the same order as `binder_seq`. |

Pipe-separated means a literal `|` between chains, for example a VH/VL antibody:

```
antigen_seq,binder_seq,...
MQKK...,QVQLVESG...|DIQMTQSP...,...
```

The two MSA columns are filled in for you by the MSA step. Everything else you
author. `binder_mode`, `antibody_form` and `antigen_temp` are carried in the CSV
for your own bookkeeping — no code reads them, and they do not change what is
folded.

A complex may not exceed **2048 residues** in total; notebook 2 filters larger
ones out before submitting.

## Prerequisite: precomputed MSAs

AnthroFold does not search MSAs — it folds from MSAs supplied with your input. So
before running the notebooks, someone needs to have run the MSA step for your
sequences.

**If you have been handed a CSV and an `msas/` folder, you are ready.** Set
`INPUT_PATH` in notebook 2 to the CSV and continue:

```python
INPUT_PATH = "examples/abag_demo/complex_input_with_msas.csv"
```

That CSV is the ordinary input CSV with two extra columns, `antigen_msa` and
`binder_msa`, pointing at the MSAs in the folder next to it. You do not need to
edit them.

**Those MSAs must come from the AnthroFold MSA pipeline.** It reproduces the
search the model was developed and evaluated against, so its MSAs match what
AnthroFold expects in depth, pairing and formatting. MSAs generated another way —
a local ColabFold run, an in-house pipeline, a public server — are not equivalent
and are not supported. Notebook 2 checks that MSAs are present, well-formed and
matched to the right chain, but it cannot tell you where they came from, so
provenance is yours to keep straight.

Two things to know:

- **Keep the CSV and its `msas/` folder together.** The CSV refers to the folder
  by relative path, so moving one without the other breaks it. Move or rename the
  containing directory freely.
- **Notebook 2 checks the MSAs before submitting anything.** If something is
  missing or mismatched it stops and lists every problem at once, before any
  compute is spent. If that happens, take the message back to whoever generated
  the MSAs.

Generating them is documented separately, in
[docs/msa-pipeline.md](docs/msa-pipeline.md#quick-start) — that is a more technical step,
run once against a separate MSA endpoint, and it does not have to be done by the
same person running these notebooks. The supplied lifecycle client deploys, waits
through database preparation, reports status, and tears that endpoint down; once it
is ready, generating a notebook-ready bundle is one command:

```bash
python src/msa_client.py --input complex_input.csv
```

## Supported Instance Types

The async endpoint runs on one of the listing's supported instance types: `ml.p4de.24xlarge` (8× A100 80 GB), `ml.p5.48xlarge` (8× H100), or `ml.p5e.48xlarge` (8× H200). Any of the three is sufficient; `ml.p4de.24xlarge` is usually the easiest to obtain. Endpoint quota for these is often 0 by default — request an increase before deploying.

## Cold Start

Initial endpoint startup takes about 1 hour while the container loads its model weights and reference data. Per-example runtime is then 5-10 minutes depending on complex size. If deployment fails due to insufficient capacity, the deploy cell cleans up and exits — re-run it to try again.

The deploy notebook enables SageMaker network isolation explicitly: the folding
container does not need outbound network access after its packaged model data has
been attached. Async success and failure outputs use separate S3 paths, and
notebook 2 rejects empty, partial, duplicate, or malformed prediction responses
instead of treating a delivered JSON object as a successful batch.

## Training and Template Cutoff

The model and its template database use a release-date cutoff of **2021-09-30**, matching the AlphaFold 3 training and inference protocol. Templates from PDB entries released after this date are filtered out during inference.

## Outputs

Multi-chain handling follows from the input: notebooks 4, 5 and 6 resolve every
antigen chain, homo- and hetero-multimer antigens included. DockQ is reported over
all antibody-antigen interfaces, and epitope residues are reported across every
antigen chain.

Successful async inference returns a JSON object with a `predictions` list. Each prediction includes `cif_content` (an mmCIF string) and a `confidence.summary` object containing `iptm`, `plddt`, `ptm`, per-chain metrics, and three `binding_score_1/2/3` signals (`binding_score_2` requires an antibody binder).

## Setup

Install the dependencies in the notebook kernel environment using your preferred environment manager. `requirements.txt` lists the packages used by the notebooks.

In SageMaker Studio, credentials come from the execution role attached to the Studio domain, user profile, or space. If running locally, configure AWS credentials with your normal AWS CLI/profile setup before opening the notebooks.
