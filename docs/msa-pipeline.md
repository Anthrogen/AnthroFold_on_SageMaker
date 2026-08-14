# Precomputing MSAs

AnthroFold does not search MSAs. It folds from MSAs you supply, generated once by
a separate MSA endpoint and reused across as many folding runs as you like.

Use MSAs from this pipeline. It reproduces the search AnthroFold was developed and
evaluated against, so its MSAs match what the model expects in depth, pairing and
formatting; MSAs built another way are not equivalent and are not supported.

This document is for whoever runs that step. If you are running the notebooks and
someone has already given you a CSV and an `msas/` folder, you do not need any of
this — see the README.

## Quick start

```bash
# 1. Deploy once and wait through the database load.
export ANTHROFOLD_EXECUTION_ROLE_ARN=arn:aws:iam::<account>:role/<role>
python src/deploy_msa_endpoint.py

# 2. Search every unique chain and build the notebook input bundle.
python src/msa_client.py --input path/to/complex_input.csv

# 3. Keep the endpoint for more CSVs; delete it only when all searches are done.
python src/teardown_msa_endpoint.py
```

Step 2 produces `complex_input_with_msas.csv` and an adjacent `msas/` directory.
Point `INPUT_PATH` in `2-invoke-endpoint.ipynb` at that generated CSV. The final
client success message means every relative MSA path has already passed the same
chain/order/content validation used by the notebook. It also reports any rows that
notebook 2 will exclude because the complex exceeds 2048 residues or its inlined
request exceeds the Marketplace endpoint's 25 MB input limit; the remaining rows
are still ready to run.

To keep generated files separate from the source CSV, use one output option:

```bash
python src/msa_client.py --input path/to/complex_input.csv --bundle-dir path/to/msa_bundle
```

That directory will contain the generated CSV and its `msas/` folder.

## What you need

- Access to the MSA model package from your account, and the ARN you were given.
- A SageMaker execution role that can pull the package and read its database, plus
  an S3 bucket the client can write requests to and read results from. The standard
  role is scoped to buckets named `sagemaker-*`.
- Credentials for the person deploying, with `sagemaker:CreateModel`,
  `sagemaker:CreateEndpointConfig`, `sagemaker:CreateEndpoint`, `iam:PassRole`,
  `sagemaker:InvokeEndpointAsync`, and the matching describe/delete permissions.
  The caller also needs `s3:ListBucket` / `s3:GetBucketLocation` and object
  read/write access on the async S3 prefix; creating the default bucket requires
  `s3:CreateBucket` if it does not already exist.
- Someone running only the MSA client needs `sagemaker:InvokeEndpointAsync` plus
  `sagemaker:DescribeEndpoint`, `sagemaker:DescribeEndpointConfig`, and
  `sagemaker:DescribeModel`, and read/write access to the async S3 bucket.
- Python 3 with `boto3` (included in `requirements.txt`).

## Deploying the MSA endpoint

The lifecycle client defaults to the published MSA package
`arn:aws:sagemaker:us-east-1:038462780959:model-package/anthrofold-msa-search/2`,
`ml.g5.12xlarge`, and your account's standard
`sagemaker-us-east-1-<account>` bucket. Give it the SageMaker execution role:

```bash
export ANTHROFOLD_EXECUTION_ROLE_ARN=arn:aws:iam::<account>:role/<role>
python src/deploy_msa_endpoint.py
```

Or pass `--role-arn` directly. Before creating anything, the command verifies the
package, instance type, caller identity, and bucket, and prints the role it will
pass to SageMaker. It then creates an async endpoint
with network isolation disabled and one concurrent invocation per instance, as the
package requires. Successes and failures are written to separate S3 prefixes, so a
failed async request cannot leave the client waiting for an output that will never
exist.

The package currently permits `ml.g5.12xlarge`, `ml.g5.48xlarge`,
`ml.p4de.24xlarge`, `ml.p5.48xlarge`, and `ml.p5e.48xlarge`. Your account still
needs endpoint quota and AWS needs available capacity for the selected type.
`ml.g5.12xlarge` is the default.

Capacity errors are retried with clean, uniquely named endpoint configs; permission,
quota, package, and container errors stop immediately. The default is five capacity
attempts with a two-minute backoff. Change it with `--max-attempts` and
`--capacity-backoff-seconds`.

### Normal deployment waits

The container copies roughly 1 TB of pinned ColabFold databases to local NVMe and
prepares them for search. During validation, an unavailable-capacity attempt took
about 38 minutes to resolve, and capacity landed on attempt 4. Once capacity landed,
the database copy and verification took about 54 minutes and the endpoint became
usable about 59 minutes after creation. Keep the command running while it prints
capacity and database-preparation progress.

SageMaker can report `InService` before the database preparation is complete. The
deploy client therefore waits beyond that state and finishes with a small
end-to-end async request. It declares success only after that request returns a
valid MSA. It writes:

```
msa_endpoint_state.json   # created immediately; enough to resume/status/delete
msa_endpoint_name.txt     # created only when inference is actually ready
```

These files are written at the repository root regardless of the directory from
which the script is called. A second deploy will not overwrite a state file that
still points to live resources; either reuse/tear down that endpoint or give a
deliberate second deployment its own `--state-file` and `--sidecar`. Useful
lifecycle commands are:

```bash
python src/msa_endpoint.py status   # capacity state + database milestone
python src/msa_endpoint.py wait     # resume waiting after a disconnect
python src/teardown_msa_endpoint.py # exact endpoint/config/model, with confirmation
```

Use `deploy --dry-run` to validate and print the configuration without creating
resources. `deploy --no-wait` submits one capacity attempt and leaves `wait` for a
later shell or notebook session; keep the normal `deploy` command attached when you
want its automatic clean capacity retries.

## Running the client

`src/msa_client.py` reads the CSV you would otherwise hand to notebook 2,
searches every distinct protein sequence in it, writes the MSAs to disk, and
emits a copy of the CSV with the MSA columns filled in.

```bash
python src/msa_client.py --input complex_input.csv
```

When the lifecycle client was used, that is the whole command: endpoint name and
bucket are discovered from its sidecar/state. You can still pass `--endpoint-name`
and `--bucket`, or set `ANTHROFOLD_MSA_ENDPOINT_NAME`, when targeting an endpoint
created elsewhere. Before waiting, the client confirms the endpoint is async. It
does not check the model package: that ARN changes with each new version.

| Flag | Default | Notes |
|---|---|---|
| `--input` | — | CSV with `antigen_seq` / `binder_seq` |
| `--endpoint-name` | lifecycle state | An already-deployed MSA endpoint |
| `--bucket` | lifecycle state or `sagemaker-<region>-<account>` | Async request/result bucket |
| `--bundle-dir` | — | Dedicated directory containing the generated CSV and `msas/`; simplest handoff option |
| `--out-dir` | `msas/` next to the output CSV | Where the a3m files are written |
| `--output-csv` | `<input>_with_msas.csv` | Next to the input by default |
| `--sequences-per-request` | 4 | Sequences per invocation |
| `--prefix` | `anthrofold-msa/requests` | S3 key root; a unique run id is appended |
| `--region` | `us-east-1` | |
| `--poll-seconds` | 30 | How often to check S3 for a finished invocation |
| `--request-ttl-seconds` | 1800 | Maximum time a request may wait in SageMaker's queue |
| `--max-wait-seconds` | 5700 | Maximum queue, processing, and result-upload wait for one attempt |
| `--max-invocation-attempts` | 3 | Attempts for transient serving failures |
| `--invocation-retry-seconds` | 15 | Delay before a transient retry |
| `--readiness-timeout-seconds` | 14400 | Maximum endpoint preparation wait |
| `--readiness` | `auto` | Use the ready log marker, with an async check when logs are unavailable |
| `--no-wait-for-ready` | off | Skip readiness preflight; intended only for controlled diagnostics |

Sequences are deduplicated first, so an antigen shared by two hundred rows is
searched once. The work is split across several invocations, each written to disk
as it returns.

## What it produces

Before:

```
my_project/
└── complex_input.csv      # six-column CSV, sequences only
```

After:

```
my_project/
├── complex_input.csv             # unchanged
├── complex_input_with_msas.csv   # the CSV plus antigen_msa / binder_msa
└── msas/
    ├── 1f100450363f/
    │   ├── non_pairing.a3m
    │   └── pairing.a3m
    ├── 15328abe5d2e/
    └── ...                                  # one directory per distinct sequence
```

One directory per *distinct sequence*, not per row — rows sharing an antigen
reference the same directory.

## How the CSV references the MSAs

`antigen_msa` and `binder_msa` hold one MSA directory per chain, `|`-separated,
matched to `antigen_seq` and `binder_seq` **by position**:

```
exp_structure,antigen_msa,binder_msa
8fdd-assembly1.cif,msas/5d82012e29fb,msas/1f899a604227|msas/4d249b37df29
```

Each column must have exactly as many entries as the sequence column it mirrors.
A VH/VL antibody has two binder chains, so `binder_msa` has two directories, in
the same order.

Paths are always written relative to the output CSV. The default puts both the CSV
and `msas/` in one movable directory. An explicit `--out-dir` must remain inside
the output CSV's directory; the client refuses paths that would escape the bundle.
Use `--bundle-dir` when you want one clean directory to copy or archive.

## Sizing an invocation

`--sequences-per-request` is bounded by time and practical response size. Each
invocation must finish inside SageMaker's 3600-second cap. Four sequences is the
package author's recommended default and usually keeps the inline-a3m result near
15–20 MB. Search time depends strongly on the sequences; the client prints a
heartbeat every minute rather than estimating completion from sequence count.

In a 313-sequence validation run, 312 searched sequences produced 78
four-sequence requests. Client-visible time per request was 3.1 minutes at the
median and 4.9 minutes at p95, with a 0.8–6.2 minute observed range. The 78
sequential requests totaled about 4 hours 4 minutes. Use these as planning
figures, not per-sequence guarantees.

The response is not subject to the Marketplace endpoint's 25 MB *input* limit —
SageMaker writes it to S3
and the client fetches it from there — but smaller result objects are faster to
transfer, retry, and inspect. Lower `--sequences-per-request` if invocations time
out or unusually deep MSAs make results unwieldy.

## Failure handling

`InvokeEndpointAsync` returning HTTP 202 means SageMaker accepted the request; it
does not mean the search succeeded. Endpoints created by the lifecycle client have
both `S3OutputPath` and `S3FailurePath`, and the MSA client polls both. It prints the
inference ID for every attempt, reports a failure immediately, and retries a
transient SageMaker/container handoff failure up to three times by default.

Endpoint state and readable CloudWatch handoff errors are checked as a second line
of defence while S3 is being polled.

For compatibility with an older endpoint that lacks `S3FailurePath`, the client
uses the endpoint's CloudWatch error stream as a terminal-failure fallback. It
refuses to start if neither mechanism is readable, because that combination could
otherwise look like a search that is still running. All waits are bounded, and
MSAs from completed batches remain safely checkpointed for a re-run.

## Resuming

MSAs are written after every invocation, so an interrupted run loses nothing —
re-run the identical command and only the unsearched sequences are searched.
Directories are named by sequence content, so a re-run never disturbs another
sequence's results and two output folders can be merged by copying them together.

MSAs are verified on reload rather than merely counted, so a missing,
half-written, mismatched, or invalid no-hits pair is re-searched instead of
trusted. The output CSV is refreshed as the run proceeds and always contains
every complete row, so a partial run still yields a CSV that can be folded.

The client locks the output CSV and MSA directory for its process lifetime. A
second client pointed at the same bundle exits immediately with the other process's
identifier instead of racing or silently mixing writes.

## Handing the MSAs on

Generating MSAs and running the notebooks are often done by different people. Hand
over **the generated CSV and its `msas/` folder** as a unit — by copy, archive, or
shared volume. The recipient points `INPUT_PATH` at the CSV.

Because the MSA paths are relative and the directory names carry no ordering, the
bundle works wherever it lands. The recipient can extend it: add rows to the CSV
and re-run the client, and only the new sequences are searched.

## Short chains

Chains of four residues or fewer are not searched. The model treats very short
peptides as single-sequence, so the client writes a query-only MSA for them
directly. They still get a directory, so the positional columns line up.

A slightly longer peptide can also come back with only its query row — a short
sequence may simply have no homologues. Notebook 2 reports that as a warning at 30
residues or fewer, and as an error above that length, where an empty result means
the search failed.

## What notebook 2 checks

Before submitting anything, notebook 2 verifies every chain: that its MSA files
exist, that each one belongs to that chain, and that the search found hits.
Problems are reported together rather than one at a time. Nothing is uploaded
until the input is clean, so a mistake here costs nothing but a re-run.

The MSA client also measures each fully inlined row against notebook 2's 2048-residue
and 25 MB Marketplace request limits, one row at a time to keep memory bounded. Rows over those
model/service limits are reported clearly and notebook 2 excludes them; their MSAs
remain valid and can be kept, split into smaller complexes where scientifically
appropriate, or removed from the folding CSV.

The MSA client now runs the same validation before declaring its generated CSV
complete, so its final success line means the CSV and relative MSA paths are ready
to use directly as notebook 2's `INPUT_PATH`.

## Teardown

Keep the endpoint open while generating every MSA bundle you need; the expensive
database load is paid once per endpoint lifetime. When finished:

```bash
python src/teardown_msa_endpoint.py
```

The command resolves and prints the exact endpoint, endpoint config, and model,
asks for confirmation, deletes them in dependency order, and removes the ready
sidecar. For non-interactive automation, add `--yes`. The state JSON is retained as
an audit record of what was removed. Teardown does not delete async request/result
objects from S3; retain them for your audit policy or remove them after the local
MSA bundle has been validated.
