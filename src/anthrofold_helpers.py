import json
import re
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Async invocation limits, all at their documented maxima. Note the API defaults are
# lower than the maxima (InvocationTimeoutSeconds defaults to 900), so these are
# always passed explicitly rather than left to the service.
MAX_INVOCATION_TIMEOUT = 3600      # processing, once the request leaves the queue
MAX_REQUEST_TTL = 21600            # how long a request may sit in the queue
# A request can spend the full TTL queued and then the full timeout processing, so a
# client that stops polling any earlier abandons work that SageMaker still runs, and
# that you are still billed for.
MAX_RESULT_WAIT = MAX_REQUEST_TTL + MAX_INVOCATION_TIMEOUT

# Chains of this length or shorter are featurised at depth 1 from the query alone,
# so a query-only MSA for them is the expected result, not a failed search.
SHORT_CHAIN_MAX_LEN = 4

# Above this length, an MSA holding only the query means the search failed. At or
# below it, it usually means the chain is a peptide with no homologues to find —
# a normal result, not an error.
PEPTIDE_MAX_LEN = 30

# S3 error codes worth retrying while polling for an async result (transient).
_RETRYABLE_S3 = {"SlowDown", "RequestTimeout", "ServiceUnavailable", "InternalError", "Throttling", "ThrottlingException"}


class AsyncInferenceFailure(RuntimeError):
    def __init__(self, failure_s3_uri, body):
        location = failure_s3_uri or "endpoint diagnostics"
        super().__init__(f"Async inference failed; details from: {location}")
        self.failure_s3_uri = failure_s3_uri
        self.body = body


def get_sagemaker_context(region="us-east-1", s3_region=None):
    s3_region = s3_region or region
    boto_session = boto3.Session(region_name=region)
    account_id = boto_session.client("sts").get_caller_identity()["Account"]

    return {
        "account_id": account_id,
        "runtime_client": boto_session.client("sagemaker-runtime"),
        "s3_client": boto3.client("s3", region_name=s3_region),
    }


def load_jobs(path):
    with Path(path).open("r", encoding="utf-8") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list):
        raise ValueError("Input JSON must be a list of prediction jobs.")
    return jobs


def _a3m_query_and_depth(text):
    """The first sequence in an a3m and the number of sequences it holds.

    Every a3m the MSA pipeline writes starts with '>query' followed by the chain
    sequence verbatim, so the first sequence identifies which chain the file
    belongs to. A depth of 1 means the file holds only the query.
    """
    query = None
    depth = 0
    for line in text.splitlines():
        if line.startswith(">"):
            depth += 1
        elif depth == 1 and query is None:
            query = line.strip()
    return query, depth


def _protein_chains(jobs):
    """Every protein chain across all jobs, as (job_name, chain_index, chain)."""
    for job in jobs:
        name = job.get("name", "<unnamed>")
        index = 0
        for entry in job.get("sequences", []):
            chain = entry.get("proteinChain")
            if chain is None:
                continue  # ligand / RNA / DNA chains carry no protein MSA
            yield name, index, chain
            index += 1


def _chain_msa_problems(name, index, chain, root):
    """Problems with one chain's MSAs, as (errors, warnings).

    Catches the three ways a CSV-supplied MSA goes wrong. The mismatch case
    matters most: chains and MSA directories are matched by position, and nothing
    downstream re-checks them — the endpoint reads whatever a3m it is handed, so a
    mis-wired path folds silently and returns a plausible wrong structure.
    """
    errors, warnings = [], []
    sequence = chain.get("sequence", "")
    where = f"job {name!r} chain {index} (length {len(sequence)})"
    # The model discards any MSA for a chain this short and featurises it at depth 1
    # from the query alone, so a query-only a3m here is the expected result and a
    # missing one costs nothing.
    short = len(sequence) <= SHORT_CHAIN_MAX_LEN
    unpaired_query_only = False

    for path_key, inline_key, label in (
        ("unpairedMsaPath", "unpairedMsa", "unpaired"),
        ("pairedMsaPath", "pairedMsa", "paired"),
    ):
        text = chain.get(inline_key)
        source = "inline"
        if text is None:
            path = chain.get(path_key)
            if not path:
                errors.append(
                    f"{where}: no MSA for the {label} channel. Supply both paired "
                    "and unpaired "
                    "MSAs from the AnthroFold MSA pipeline."
                )
                continue
            p = Path(path)
            if root is not None and not p.is_absolute():
                p = root / p
            source = str(p)
            if not p.exists():
                errors.append(f"{where}: {label} MSA not found at {p}")
                continue
            text = p.read_text(encoding="utf-8")

        query, depth = _a3m_query_and_depth(text)
        if query is None:
            errors.append(f"{where}: {label} MSA at {source} is empty")
        elif query != sequence:
            errors.append(
                f"{where}: {label} MSA at {source} is for a different chain — its "
                f"query sequence starts {query[:20]!r} but this chain starts "
                f"{sequence[:20]!r}. MSA columns are matched to sequence columns by "
                "position; check the order."
            )
        elif depth <= 1 and not short:
            # A search that finds nothing still writes a well-formed query-only
            # a3m and returns normally, so it never surfaces as an error upstream.
            if label != "unpaired":
                if not unpaired_query_only:
                    warnings.append(
                        f"{where}: paired MSA at {source} contains only the query "
                        "sequence — pairing was lost for this chain."
                    )
            elif len(sequence) > PEPTIDE_MAX_LEN:
                unpaired_query_only = True
                errors.append(
                    f"{where}: {label} MSA at {source} contains only the query "
                    "sequence — the search found no hits. Re-run the MSA step "
                    "for this chain, or remove the row."
                )
            else:
                unpaired_query_only = True
                # A peptide this short often has no homologues; the model folds it
                # single-sequence either way, so this is worth surfacing, not blocking.
                warnings.append(
                    f"{where}: {label} MSA at {source} contains only the query "
                    "sequence. Expected for a peptide this short, which will fold "
                    "single-sequence."
                )

    return errors, warnings


def validate_msas(jobs, base_dir=None):
    """Check every chain's MSA before anything is uploaded or invoked.

    Reports every bad row at once rather than dying on the first, so a large CSV
    can be fixed in one pass. Returns a summary; raises ValueError listing all
    errors.
    """
    root = Path(base_dir) if base_dir else None
    errors, warnings, n_chains = [], [], 0
    for name, index, chain in _protein_chains(jobs):
        n_chains += 1
        chain_errors, chain_warnings = _chain_msa_problems(name, index, chain, root)
        errors.extend(chain_errors)
        warnings.extend(chain_warnings)

    if errors:
        raise ValueError(
            f"{len(errors)} MSA problem(s) across {len(jobs)} job(s):\n  - "
            + "\n  - ".join(errors)
        )
    return {"n_jobs": len(jobs), "n_chains": n_chains, "warnings": warnings}


def inline_msas(jobs, base_dir=None, require=True):
    """Embed each protein chain's precomputed MSA into the request JSON.

    AnthroFold requires you to supply precomputed MSAs. Generate them with the
    AnthroFold MSA pipeline (``src/msa_client.py``), which writes per-chain ``pairing.a3m`` /
    ``non_pairing.a3m`` and an input JSON that carries ``unpairedMsaPath`` /
    ``pairedMsaPath`` on each ``proteinChain``.

    A SageMaker invocation is a single JSON payload with no shared filesystem, so
    those *paths* cannot cross the wire. This reads the a3m files and inlines
    their TEXT as ``unpairedMsa`` / ``pairedMsa`` (the fields the endpoint
    consumes), dropping the path fields. Chains that already carry inline
    ``unpairedMsa``/``pairedMsa`` are left as-is.

    Args:
        jobs: prediction jobs (from ``load_jobs`` / ``csv_to_jobs``).
        base_dir: optional root to resolve relative MSA paths against.
        require: if True (default), require both paired and unpaired MSAs for
            every protein chain.

    Returns:
        The same ``jobs`` list, with inline MSAs on every protein chain.
    """
    root = Path(base_dir) if base_dir else None
    for job in jobs:
        name = job.get("name", "<unnamed>")
        for entry in job.get("sequences", []):
            chain = entry.get("proteinChain")
            if chain is None:
                continue  # ligand / RNA / DNA chains carry no protein MSA
            for path_key, inline_key in (
                ("unpairedMsaPath", "unpairedMsa"),
                ("pairedMsaPath", "pairedMsa"),
            ):
                path = chain.pop(path_key, None)
                if not path:
                    continue
                p = Path(path)
                if root is not None and not p.is_absolute():
                    p = root / p
                if not p.exists():
                    raise FileNotFoundError(f"{name}: MSA file for {path_key} not found: {p}")
                text = p.read_text(encoding="utf-8")
                # Safety net for callers that skip validate_msas: an MSA wired to
                # the wrong chain would otherwise fold silently.
                query, _ = _a3m_query_and_depth(text)
                sequence = chain.get("sequence", "")
                if query is not None and query != sequence:
                    raise ValueError(
                        f"{name}: {p} is an MSA for a different chain (query starts "
                        f"{query[:20]!r}, chain starts {sequence[:20]!r})."
                    )
                chain[inline_key] = text
            missing_inline = [
                key for key in ("unpairedMsa", "pairedMsa") if key not in chain
            ]
            if require and missing_inline:
                seq = chain.get("sequence", "")
                raise ValueError(
                    f"{name}: a protein chain (sequence {seq[:12]}...) is missing "
                    f"{', '.join(missing_inline)}. AnthroFold requires both paired "
                    "and unpaired precomputed MSAs from the AnthroFold MSA pipeline: "
                    "python src/msa_client.py --input <your.csv> ..."
                )
    return jobs


def total_residues(job):
    total = 0
    for entry in job.get("sequences", []):
        chain = entry.get("proteinChain", {})
        count = int(chain.get("count", 1) or 1)
        total += len(chain.get("sequence", "")) * max(count, 1)
    return total


def chain_sizes(job):
    return [
        len(entry.get("proteinChain", {}).get("sequence", ""))
        for entry in job.get("sequences", [])
    ]


def batch_label(batch_index, batch_jobs):
    total = sum(total_residues(job) for job in batch_jobs)
    return f"batch_{batch_index:03d}_n{len(batch_jobs)}_aa{total}"


# AWS Marketplace endpoints reject an invocation whose input data exceeds 25 MB.
# With MSAs inlined
# the request is dominated by a3m text, not by sequence length, so this bound has to
# be enforced on serialized bytes -- max_residues does not track it even loosely. A
# 2-complex / 6-chain request measured 24.8 MB against real MSAs.
MAX_PAYLOAD_BYTES = 25 * 1000 * 1000
# Leave room for the JSON array's own punctuation and any header overhead.
_PAYLOAD_MARGIN = 64 * 1024


def payload_bytes(job):
    """Serialized size of one job, as it will be sent."""
    return len(json.dumps(job).encode("utf-8"))


def plan_batches(
    jobs,
    batch_size=None,
    max_residues=2048,
    sort_by_size=True,
    limit_batches=None,
    max_payload_bytes=MAX_PAYLOAD_BYTES,
):
    """Pack jobs into as few requests as fit under the payload cap.

    MSAs are never trimmed to make something fit — a shallower MSA is a different
    (worse) prediction, not a smaller version of the same one. Packing is by
    serialized bytes; ``batch_size`` is an optional extra cap, off by default, so
    batches fill the 25 MB budget rather than stopping at an arbitrary job count.
    """
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be >= 1 or None")

    budget = max_payload_bytes - _PAYLOAD_MARGIN
    selected, excluded = [], []
    for job in jobs:
        # A job that cannot fit in a request of its own is unsendable, exactly like
        # one that exceeds the residue cap. It is reported, never truncated.
        if total_residues(job) > max_residues or payload_bytes(job) > budget:
            excluded.append(job)
        else:
            selected.append(job)

    # First-fit-decreasing: placing the biggest jobs first leaves the small ones to
    # top up whatever room is left, which packs tighter than ascending order.
    order = sorted(
        selected,
        key=lambda job: (-payload_bytes(job), str(job.get("name", ""))),
    ) if sort_by_size else list(selected)

    batches = []
    batch_bytes = []
    for job in order:
        size = payload_bytes(job) + 1  # + the array separator
        for i, used in enumerate(batch_bytes):
            if used + size <= budget and (batch_size is None or len(batches[i]) < batch_size):
                batches[i].append(job)
                batch_bytes[i] = used + size
                break
        else:
            batches.append([job])
            batch_bytes.append(size + 2)  # + the enclosing brackets

    if sort_by_size:
        # Submit smallest-first so a quick batch confirms the endpoint is healthy
        # before the long ones go.
        paired = sorted(zip(batches, batch_bytes), key=lambda p: (p[1], p[0][0].get("name", "")))
        batches = [b for b, _ in paired]

    if limit_batches is not None:
        batches = batches[:limit_batches]
    return batches, excluded


def batch_manifest(jobs, batches, excluded, batch_size, max_residues):
    return {
        "input_count": len(jobs),
        "selected_count": sum(len(batch) for batch in batches),
        "excluded_count": len(excluded),
        "batch_size": batch_size,
        "max_residues": max_residues,
        "excluded": [
            {
                "name": job.get("name"),
                "total_residues": total_residues(job),
                "chain_sizes": chain_sizes(job),
                "payload_bytes": payload_bytes(job),
                "reason": (
                    "residues" if total_residues(job) > max_residues else "payload_bytes"
                ),
            }
            for job in excluded
        ],
        "batches": [
            {
                "batch_index": i,
                "batch_label": batch_label(i, batch),
                "names": [job.get("name") for job in batch],
                "total_residues": [total_residues(job) for job in batch],
                "chain_sizes": [chain_sizes(job) for job in batch],
                "payload_bytes": sum(payload_bytes(job) for job in batch),
            }
            for i, batch in enumerate(batches, start=1)
        ],
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def upload_jobs(s3_client, bucket, prefix, jobs, name):
    key = f"{prefix.strip('/')}/{safe_name(name)}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(jobs).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def invoke_async(
    *,
    runtime_client,
    endpoint_name,
    input_s3_uri,
    invocation_timeout_seconds=MAX_INVOCATION_TIMEOUT,
    request_ttl_seconds=MAX_REQUEST_TTL,
    output_path_extension=None,
):
    if invocation_timeout_seconds > MAX_INVOCATION_TIMEOUT:
        raise ValueError(f"InvocationTimeoutSeconds max is {MAX_INVOCATION_TIMEOUT}.")
    if request_ttl_seconds > MAX_REQUEST_TTL:
        raise ValueError(f"RequestTTLSeconds max is {MAX_REQUEST_TTL}.")

    kwargs = {
        "EndpointName": endpoint_name,
        "InputLocation": input_s3_uri,
        "ContentType": "application/json",
        "InvocationTimeoutSeconds": invocation_timeout_seconds,
        "RequestTTLSeconds": request_ttl_seconds,
    }
    if output_path_extension:
        kwargs["S3OutputPathExtension"] = output_path_extension
    return runtime_client.invoke_endpoint_async(**kwargs)


def wait_for_result(
    *,
    s3_client,
    output_s3_uri,
    failure_s3_uri=None,
    poll_seconds=30,
    max_wait_seconds=MAX_RESULT_WAIT,
    terminal_failure_check=None,
):
    """Poll S3 for the async output (or failure) object until one appears.

    Only NoSuchKey is treated as "not ready yet"; other ClientErrors
    (AccessDenied, NoCredentials, etc.) propagate immediately rather than
    being silently retried.
    """
    out_bucket, out_key = split_s3_uri(output_s3_uri)
    fail_bucket = fail_key = None
    if failure_s3_uri:
        fail_bucket, fail_key = split_s3_uri(failure_s3_uri)

    deadline = time.monotonic() + max_wait_seconds
    start = time.monotonic()
    while time.monotonic() < deadline:
        try:
            obj = s3_client.get_object(Bucket=out_bucket, Key=out_key)
            return json.loads(obj["Body"].read())
        except s3_client.exceptions.NoSuchKey:
            pass
        except ClientError as exc:
            # Retry transient S3 errors (throttling / brief outages); surface real
            # ones (AccessDenied, NoSuchBucket, ...) immediately.
            if exc.response.get("Error", {}).get("Code") not in _RETRYABLE_S3:
                raise

        if fail_bucket and fail_key:
            try:
                obj = s3_client.get_object(Bucket=fail_bucket, Key=fail_key)
                body = obj["Body"].read().decode("utf-8", errors="replace")
                raise AsyncInferenceFailure(failure_s3_uri, body)
            except s3_client.exceptions.NoSuchKey:
                pass
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in _RETRYABLE_S3:
                    raise

        # A correctly deployed async endpoint writes a failure object. This
        # callback is a second line of defence for endpoint deletion/failure and
        # for older endpoint configs that omitted S3FailurePath. It must return a
        # non-empty diagnostic only for a terminal failure.
        if terminal_failure_check is not None:
            diagnostic = terminal_failure_check()
            if diagnostic:
                raise AsyncInferenceFailure(None, diagnostic)

        # One heartbeat per minute, not every poll, to keep notebook output tidy.
        elapsed = int(time.monotonic() - start)
        if elapsed > 0 and elapsed % 60 < poll_seconds:
            print(f"  ...still running ({elapsed // 60} min)")
        time.sleep(poll_seconds)

    destinations = f"success={output_s3_uri}"
    if failure_s3_uri:
        destinations += f", failure={failure_s3_uri}"
    raise TimeoutError(
        f"No async success or failure object appeared within {max_wait_seconds}s "
        f"({destinations})."
    )


def summarize_predictions(result):
    rows = []
    for pred in result.get("predictions") or []:
        conf = pred.get("confidence", {}) or {}
        summary = conf.get("summary", conf)  # handler may nest metrics under 'summary' or return them flat
        rows.append(
            {
                "name": pred.get("name", "<unnamed>"),
                "iptm": summary.get("iptm"),
                "plddt": summary.get("plddt"),
                "ptm": summary.get("ptm"),
                "ranking_score": summary.get("ranking_score"),
                "has_clash": summary.get("has_clash"),
                "cif_chars": len(pred.get("cif_content", "") or ""),
            }
        )
    return rows


def save_predictions(result, output_dir="outputs", prefix=None):
    output = Path(output_dir)
    cif_dir = output / "cifs"
    confidence_dir = output / "confidence"
    cif_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for pred in result.get("predictions") or []:
        name = safe_name(str(pred.get("name", "prediction")))
        stem = f"{safe_name(prefix)}_{name}" if prefix else name
        cif_path = cif_dir / f"{stem}.cif"
        confidence_path = confidence_dir / f"{stem}.json"
        cif_path.write_text(pred.get("cif_content", "") or "", encoding="utf-8")
        confidence_path.write_text(
            json.dumps(pred.get("confidence", {}) or {}, indent=2),
            encoding="utf-8",
        )
        saved.append({"name": name, "cif": str(cif_path), "confidence": str(confidence_path)})
    return saved


def quick_cif_plot(cif_text, width=800, height=500):
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.addModel(cif_text, "cif")
    view.setStyle({"cartoon": {"colorscheme": "chain"}})
    view.zoomTo()
    view.show()


def split_s3_uri(uri):
    bucket, _, key = uri.replace("s3://", "", 1).partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unnamed"
