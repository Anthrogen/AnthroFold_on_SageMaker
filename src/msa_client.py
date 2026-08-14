"""Run the AnthroFold MSA search over your input CSV.

    python src/msa_client.py --input complex_input.csv

AnthroFold does not search MSAs — you supply them. This sends every unique protein
sequence in your CSV to the MSA endpoint, writes the returned MSAs to disk, and
emits a copy of your CSV with `antigen_msa` / `binder_msa` filled in. Point
`INPUT_PATH` in 2-invoke-endpoint.ipynb at that CSV.

CSV in, CSV out; the JSON request exists only in flight. MSA directories are
recorded relative to the output CSV, so keep the CSV and the MSA folder together.
"""

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anthrofold_helpers import (AsyncInferenceFailure, MAX_INVOCATION_TIMEOUT,
                                PEPTIDE_MAX_LEN, SHORT_CHAIN_MAX_LEN, invoke_async,
                                inline_msas, plan_batches, upload_jobs,
                                total_residues, validate_msas, wait_for_result)
from az_adapter import (PAIRED_A3M, SEQ_COLUMNS, UNPAIRED_A3M, csv_to_jobs,
                        split_chains)
from msa_endpoint import (DEFAULT_READY_TIMEOUT, DEFAULT_SIDECAR,
                          DEFAULT_STATE_FILE, _is_transient_invocation_failure,
                          wait_until_ready)

# Sequences per invocation, bounded by TIME, not by response size.
#
# The 25 MB Marketplace cap is on "input data per invocation" only. The response is
# not part of the payload: SageMaker uploads the container's HTTP response to S3 and
# the client fetches it from there, so a large a3m response is an S3 object, not an
# oversized reply. The request itself is a few kB of sequence either way.
#
# What does bind is InvocationTimeoutSeconds, which maxes at 3600s. Search duration
# varies substantially with the sequences. The package's own serving guide
# recommends four sequences so typical inline-a3m responses stay around 15--20 MB.
SEQUENCES_PER_REQUEST = 4

DEFAULT_MAX_INVOCATION_ATTEMPTS = 3
DEFAULT_INVOCATION_RETRY_SECONDS = 15
# This client submits one request at a time to an endpoint configured for one
# concurrent invocation. Thirty minutes is therefore ample queue time; adding the
# package's 60-minute processing limit plus five minutes for result upload/polling
# gives a bounded 95-minute client wait.
DEFAULT_REQUEST_TTL_SECONDS = 30 * 60
DEFAULT_RESULT_WAIT_SECONDS = (
    DEFAULT_REQUEST_TTL_SECONDS + MAX_INVOCATION_TIMEOUT + 5 * 60
)


class OutputLocks:
    """Process-lifetime advisory locks for the output CSV and MSA directory."""

    def __init__(self, paths):
        self._handles = []
        try:
            for path in sorted({Path(path).resolve() for path in paths}, key=str):
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    handle.seek(0)
                    owner = handle.read().strip() or "another process"
                    handle.close()
                    raise RuntimeError(
                        f"Another MSA client is writing this bundle ({path}; {owner}). "
                        "Wait for it to finish, or choose a different --bundle-dir."
                    ) from exc
                handle.seek(0)
                handle.truncate()
                handle.write(
                    f"pid={os.getpid()} started="
                    f"{datetime.now(timezone.utc).isoformat()}\n"
                )
                handle.flush()
                self._handles.append(handle)
        except Exception:
            self.close()
            raise

    def close(self):
        while self._handles:
            handle = self._handles.pop()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()



def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in SEQ_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV at {path} is missing required column(s): {', '.join(missing)}. "
                f"Found: {reader.fieldnames}"
            )
        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV at {path} has a header but no data rows.")
        return rows, list(reader.fieldnames)


def unique_sequences(rows):
    """Every distinct protein sequence in the CSV, in stable first-seen order.

    The search is per sequence, not per row: a sequence used by ten rows is
    searched once and all ten reference the same MSA directory.
    """
    seen = {}
    for row in rows:
        for column in SEQ_COLUMNS:
            for chain in split_chains(row.get(column, "")):
                seen.setdefault(chain, None)
    return list(seen)


def chunk_to_jobs(sequences, index):
    """One request's worth of sequences, in the shape the endpoint accepts.

    Requests are chunked by search time. The request itself is tiny either way -- a
    few kB of sequence -- and the response goes to S3 rather than back over the
    wire, so size is not the constraint. One invocation for a whole CSV would
    simply never finish inside InvocationTimeoutSeconds.
    """
    return [{
        "name": f"msa_chunk_{index:04d}",
        "sequences": [{"proteinChain": {"sequence": s, "count": 1, "modifications": []}}
                      for s in sequences],
        "covalent_bonds": [],
    }]


def msa_dir_name(sequence):
    """Directory name for a sequence: the first 12 hex characters of its SHA-256.

    Content-addressed, so a sequence always lands in the same directory. That is
    what lets resume work with no index file, keeps a re-run from overwriting a
    different sequence's MSAs, and lets two output folders be merged — identical
    sequences collide by design rather than by accident.
    """
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:12]


def _a3m_query_and_has_hits(path):
    """Return an A3M's query and whether it contains at least one hit.

    Stop at the second header so resuming does not read multi-megabyte alignment
    files into memory merely to decide whether a sequence is already complete.
    """
    query_lines = []
    depth = 0
    try:
        with open(path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if line.startswith(">"):
                    depth += 1
                    if depth == 2:
                        return ("".join(query_lines) or None), True
                elif depth == 1 and line:
                    query_lines.append(line)
    except OSError:
        return None, False
    return ("".join(query_lines) or None), False


def _a3m_text_query_depth(text):
    """Return the first A3M record's sequence and total record count."""
    query_lines = []
    depth = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            depth += 1
        elif depth == 1 and line:
            query_lines.append(line)
    return ("".join(query_lines) or None), depth


def dummy_msas(sequences):
    """The query-only a3m a chain gets when there is nothing to align it to.

    Byte-identical to what the MSA pipeline writes for a search that finds
    nothing, so a short chain's files are indistinguishable from a searched one.
    The model builds the same depth-1 MSA internally for chains this short, so
    writing it here just keeps every chain represented on disk.
    """
    return {s: (f">query\n{s}\n", f">query\n{s}\n") for s in sequences}


def existing_msas(out_dir):
    """sequence -> directory for MSAs already on disk.

    Read from the directories themselves rather than an index: every a3m states
    its own sequence as its query row, so the mapping is derivable and there is no
    separate file to lose, move, or disagree with the contents. Scanning is cheap —
    two lines per directory.

    A directory counts only when both a3m files are present and carry the same
    sequence, so a truncated or half-written one is re-searched rather than
    trusted. Names are not interpreted, so directories from an older numbered
    layout are still picked up.
    """
    root = Path(out_dir)
    found = {}
    if not root.is_dir():
        return found
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        seq, unpaired_has_hits = _a3m_query_and_has_hits(d / UNPAIRED_A3M)
        paired_seq, _ = _a3m_query_and_has_hits(d / PAIRED_A3M)
        # Query-only is a valid completed result for peptides. For a longer
        # sequence it is the search pipeline's silent no-hits failure mode, so do
        # not resume past it: leave it out and let the client search it again.
        if (
            seq
            and paired_seq == seq
            and (len(seq) <= PEPTIDE_MAX_LEN or unpaired_has_hits)
        ):
            found[seq] = d
    return found


def extract_msas(response):
    """Map sequence -> (unpaired_a3m, paired_a3m) from the endpoint response.

    The response is the request array with `unpairedMsa` / `pairedMsa` added to
    each proteinChain — what the MSA pipeline's `--inline` mode returns.
    """
    # The contract is a bare task array, but some handlers wrap it in an
    # envelope. Unwrap rather than fail.
    if isinstance(response, dict):
        for key in ("predictions", "msas", "tasks", "jobs", "output", "result"):
            if isinstance(response.get(key), list):
                response = response[key]
                break
    if not isinstance(response, list):
        keys = sorted(response) if isinstance(response, dict) else None
        raise ValueError(
            "Unexpected MSA endpoint response: expected a list of tasks (or a dict "
            f"wrapping one), got {type(response).__name__}"
            + (f" with keys {keys}." if keys else ".")
        )
    msas = {}
    for task in response:
        for entry in task.get("sequences", []):
            chain = entry.get("proteinChain")
            if chain is None:
                continue
            sequence = chain.get("sequence")
            unpaired, paired = chain.get("unpairedMsa"), chain.get("pairedMsa")
            if sequence and isinstance(unpaired, str) and isinstance(paired, str):
                if not unpaired.strip() or not paired.strip():
                    raise ValueError(
                        f"The MSA endpoint returned an empty inline MSA for sequence "
                        f"{sequence[:24]}..."
                    )
                unpaired_query, unpaired_depth = _a3m_text_query_depth(unpaired)
                paired_query, _ = _a3m_text_query_depth(paired)
                if unpaired_query != sequence or paired_query != sequence:
                    raise ValueError(
                        "The MSA endpoint returned an A3M whose query does not "
                        f"match its sequence field: {sequence[:24]}..."
                    )
                if unpaired_depth <= 1 and len(sequence) > PEPTIDE_MAX_LEN:
                    raise ValueError(
                        "The MSA search returned no hits for a non-peptide sequence "
                        f"(length {len(sequence)}, starts {sequence[:24]}...). "
                        "Nothing from this invocation will be checkpointed."
                    )
                if sequence in msas:
                    raise ValueError(
                        "The MSA endpoint returned the same sequence more than once "
                        f"in one response: {sequence[:24]}..."
                    )
                msas[sequence] = (unpaired, paired)
    if not msas:
        raise ValueError(
            "The MSA endpoint returned no MSAs. Check that --endpoint-name names "
            "an AnthroFold MSA endpoint."
        )
    return msas


def write_msa_dirs(msas, out_dir):
    """Write one directory per sequence, named by content; return sequence -> dir.

    Writes are atomic: a temporary name renamed into place, so a process killed
    mid-write leaves either the previous state or nothing, never a partial a3m
    that later looks complete.
    """
    out = Path(out_dir).resolve()
    dirs = {}
    for sequence, (unpaired, paired) in msas.items():
        d = out / msa_dir_name(sequence)
        d.mkdir(parents=True, exist_ok=True)
        for name, text in ((UNPAIRED_A3M, unpaired), (PAIRED_A3M, paired)):
            tmp = d / f".{name}.partial"
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, d / name)
        dirs[sequence] = d
    return dirs


def write_rows_with_msas(rows, fieldnames, seq_to_dir, out_csv, relative_to=None,
                         skip_incomplete=False):
    """Emit the input CSV with antigen_msa / binder_msa added.

    Entries are positional: the nth directory in an MSA column belongs to the nth
    chain of the sequence column it mirrors.

    ``skip_incomplete`` omits rows whose chains are not all searched yet, so a run
    in progress still leaves a CSV that is valid for the rows it does contain. A
    row with some MSA columns blank would be rejected downstream anyway, so a
    partial row is worth less than no row. Off at the end of a run, where a missing
    sequence is a real error.

    Returns (path, n_rows_written).
    """
    out_path = Path(out_csv).resolve()
    msa_columns = [c.replace("_seq", "_msa") for c in SEQ_COLUMNS]
    out_fields = list(fieldnames) + [c for c in msa_columns if c not in fieldnames]
    root = Path(relative_to).resolve() if relative_to else None

    missing, prepared = set(), []
    for row in rows:
        out_row = dict(row)
        row_missing = False
        for seq_column, msa_column in zip(SEQ_COLUMNS, msa_columns):
            paths = []
            for chain in split_chains(row.get(seq_column, "")):
                d = seq_to_dir.get(chain)
                if d is None:
                    missing.add(chain[:20])
                    row_missing = True
                    continue
                d = Path(d).resolve()
                if root is not None:
                    try:
                        d.relative_to(root)
                    except ValueError as exc:
                        raise ValueError(
                            f"MSA directory {d} is outside the output CSV directory "
                            f"{root}; refusing to write an escaping bundle path."
                        ) from exc
                    d = Path(os.path.relpath(d, start=root))
                paths.append(d.as_posix())
            out_row[msa_column] = "|".join(paths)
        if row_missing and skip_incomplete:
            continue
        prepared.append(out_row)

    if missing and not skip_incomplete:
        raise ValueError(
            f"The MSA bundle is incomplete: the endpoint returned no MSA for "
            f"{len(missing)} sequence(s), e.g. "
            f"{sorted(missing)[:3]}. The last valid output CSV at {out_path} was "
            "left untouched."
        )

    # Atomic for the same reason as each a3m write: interruption must not replace a
    # valid checkpoint CSV with a half-written one.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.partial")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(prepared)
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return str(out_path), len(prepared)


def _lifecycle_state(path=DEFAULT_STATE_FILE):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def resolve_endpoint_name(explicit=None, sidecar=DEFAULT_SIDECAR, state=None):
    """Resolve the MSA endpoint without depending on the caller's directory."""
    candidates = (
        explicit,
        os.environ.get("ANTHROFOLD_MSA_ENDPOINT_NAME"),
        (state or {}).get("endpoint_name"),
    )
    for value in candidates:
        if value and str(value).strip():
            return str(value).strip()
    sidecar = Path(sidecar)
    if sidecar.exists():
        value = sidecar.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise ValueError(
        "MSA endpoint name is not set. Run `python src/msa_endpoint.py deploy`, "
        "pass --endpoint-name, or set ANTHROFOLD_MSA_ENDPOINT_NAME."
    )


def endpoint_async_config(sm, endpoint_name):
    """Return the endpoint's async serving configuration.

    Only the async contract is checked: this client submits with
    InvokeEndpointAsync and reads the result from S3, so a real-time endpoint
    cannot serve it. The model package, its concurrency and its network settings
    belong to whoever deployed the endpoint — a serving endpoint has already
    proved they work, and the package ARN changes with every new version, rename
    or Marketplace listing.
    """
    endpoint = sm.describe_endpoint(EndpointName=endpoint_name)
    config = sm.describe_endpoint_config(
        EndpointConfigName=endpoint["EndpointConfigName"])
    async_config = config.get("AsyncInferenceConfig")
    if not async_config:
        raise RuntimeError(
            f"Endpoint {endpoint_name} is not an async endpoint. This client submits "
            "with InvokeEndpointAsync and reads the result from S3."
        )
    return async_config


def _cloudwatch_server_error(logs, endpoint_name, start_ms, inference_id=None):
    """Return a SageMaker handoff failure emitted after ``start_ms``, if any."""
    events = []
    pattern = '"Received server error"'
    if inference_id:
        # SageMaker's data-log correlation id is the async InferenceId. Matching
        # both avoids treating another user's failed request as this one's.
        pattern += f' "{inference_id}"'
    pages = logs.get_paginator("filter_log_events").paginate(
        logGroupName=f"/aws/sagemaker/Endpoints/{endpoint_name}",
        startTime=start_ms,
        filterPattern=pattern,
        PaginationConfig={"MaxItems": 20},
    )
    for page in pages:
        events.extend(page.get("events", []))
    events.sort(key=lambda event: event["timestamp"])
    return events[0].get("message", "").strip() if events else None


def terminal_failure_checker(
    sm,
    logs,
    endpoint_name,
    submitted_ms,
    check_logs,
    inference_id=None,
    logs_required=False,
):
    """Build a poll callback for failures that may not create an S3 object."""
    def check():
        endpoint = sm.describe_endpoint(EndpointName=endpoint_name)
        status = endpoint.get("EndpointStatus")
        if status in {"Failed", "Deleting", "OutOfService"}:
            reason = endpoint.get("FailureReason", "no failure reason returned")
            return f"Endpoint entered {status}: {reason}"
        if check_logs:
            try:
                return _cloudwatch_server_error(
                    logs, endpoint_name, submitted_ms, inference_id=inference_id
                )
            except Exception as exc:
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
                if code in {
                    "AccessDenied",
                    "AccessDeniedException",
                    "ResourceNotFoundException",
                    "Throttling",
                    "ThrottlingException",
                    "ServiceUnavailableException",
                }:
                    # A configured S3 failure destination remains authoritative.
                    # A legacy endpoint has no other terminal-failure channel, so
                    # losing log visibility must stop rather than turn into a long
                    # blind poll.
                    if logs_required and code in {
                        "AccessDenied",
                        "AccessDeniedException",
                        "ResourceNotFoundException",
                    }:
                        return (
                            "CloudWatch terminal-failure diagnostics became "
                            f"unavailable ({code}) and this endpoint has no "
                            "S3FailurePath."
                        )
                    return None
                raise
        return None

    return check


def validate_notebook_sendability(jobs, base_dir):
    """Report whether each job can be sent alone within notebook limits.

    Work one job at a time so a large bundle never holds every A3M in memory at
    once. Notebook 2 may combine several jobs into a batch, but its packer can
    always send a job that fits alone.
    """
    problems = []
    largest_payload = 0
    for job in jobs:
        inlined = copy.deepcopy(job)
        inline_msas([inlined], base_dir=base_dir, require=True)
        batches, excluded = plan_batches([inlined])
        payload_size = len(json.dumps([inlined]).encode("utf-8"))
        largest_payload = max(largest_payload, payload_size)
        if excluded or not batches:
            residues = total_residues(inlined)
            problems.append({
                "name": job.get("name", "<unnamed>"),
                "payload_bytes": payload_size,
                "total_residues": residues,
                "reason": "residue_limit" if residues > 2048 else "payload_limit",
            })
    return {
        "n_jobs": len(jobs),
        "foldable_jobs": len(jobs) - len(problems),
        "excluded_jobs": problems,
        "largest_single_job_payload_bytes": largest_payload,
    }


def main():
    p = argparse.ArgumentParser(
        prog="msa_client", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Input CSV.")
    p.add_argument("--endpoint-name", help="Deployed MSA endpoint. Default: lifecycle sidecar/state or ANTHROFOLD_MSA_ENDPOINT_NAME.")
    p.add_argument("--bucket", help="S3 bucket for async I/O. Default: lifecycle state or sagemaker-<region>-<account>.")
    p.add_argument("--prefix", default=None, help="S3 key root. A collision-proof run id is always appended.")
    p.add_argument("--bundle-dir", default=None,
                   help="Put the generated CSV and msas/ together in this directory. "
                        "Cannot be combined with --out-dir or --output-csv.")
    p.add_argument("--out-dir", "--out_dir", dest="out_dir", default=None,
                   help="Where to write a3m files. Default: an msas/ directory next to the output CSV.")
    p.add_argument("--output-csv", default=None,
                   help="Output CSV. Default: <input stem>_with_msas.csv next to the input.")
    p.add_argument("--sequences-per-request", type=int, default=SEQUENCES_PER_REQUEST,
                   help="Sequences per invocation. Bounded by search time, not "
                        f"payload: each invocation must finish inside the "
                        f"{MAX_INVOCATION_TIMEOUT}s processing timeout. "
                        f"Default {SEQUENCES_PER_REQUEST}.")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--request-ttl-seconds", type=int,
                   default=DEFAULT_REQUEST_TTL_SECONDS,
                   help="Maximum queue wait before SageMaker expires a request. "
                        f"Default {DEFAULT_REQUEST_TTL_SECONDS}.")
    p.add_argument("--max-wait-seconds", type=int,
                   default=DEFAULT_RESULT_WAIT_SECONDS,
                   help="Maximum client wait per attempt, including queue and "
                        f"processing. Default {DEFAULT_RESULT_WAIT_SECONDS}.")
    p.add_argument("--max-invocation-attempts", type=int,
                   default=DEFAULT_MAX_INVOCATION_ATTEMPTS,
                   help="Attempts for transient SageMaker/container handoff "
                        f"failures. Default {DEFAULT_MAX_INVOCATION_ATTEMPTS}.")
    p.add_argument("--invocation-retry-seconds", type=int,
                   default=DEFAULT_INVOCATION_RETRY_SECONDS,
                   help="Delay between transient invocation attempts. "
                        f"Default {DEFAULT_INVOCATION_RETRY_SECONDS}.")
    p.add_argument("--readiness-timeout-seconds", type=int, default=DEFAULT_READY_TIMEOUT,
                   help="How long to wait for endpoint database preparation. "
                        f"Default {DEFAULT_READY_TIMEOUT}.")
    p.add_argument("--readiness", choices=("auto", "logs", "probe"), default="auto",
                   help="Readiness check: CloudWatch marker with async-probe fallback by default.")
    p.add_argument("--no-wait-for-ready", action="store_true",
                   help="Skip the preflight readiness check (not recommended).")
    args = p.parse_args()

    if args.sequences_per_request < 1:
        p.error("--sequences-per-request must be at least 1")
    if args.poll_seconds < 1:
        p.error("--poll-seconds must be at least 1")
    if not 60 <= args.request_ttl_seconds <= 21600:
        p.error("--request-ttl-seconds must be between 60 and 21600")
    if args.max_wait_seconds < args.request_ttl_seconds:
        p.error("--max-wait-seconds must be at least --request-ttl-seconds")
    if args.max_invocation_attempts < 1:
        p.error("--max-invocation-attempts must be at least 1")
    if args.invocation_retry_seconds < 0:
        p.error("--invocation-retry-seconds cannot be negative")
    if args.readiness_timeout_seconds < 1:
        p.error("--readiness-timeout-seconds must be at least 1")

    import boto3

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        p.error(f"--input is not a readable file: {input_path}")
    if args.bundle_dir and (args.output_csv or args.out_dir):
        p.error("--bundle-dir cannot be combined with --output-csv or --out-dir")
    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir).expanduser().resolve()
        out_csv = bundle_dir / f"{input_path.stem}_with_msas.csv"
        out_dir = bundle_dir / "msas"
    else:
        out_csv = (
            Path(args.output_csv).expanduser().resolve()
            if args.output_csv
            else input_path.with_name(input_path.stem + "_with_msas.csv")
        )
        out_dir = (
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir
            else out_csv.parent / "msas"
        )
    if out_csv == input_path:
        p.error("--output-csv must not overwrite --input")
    try:
        out_dir.relative_to(out_csv.parent)
    except ValueError:
        p.error(
            "--out-dir must be inside the output CSV directory so the generated "
            "bundle has portable, non-escaping relative paths. Use --bundle-dir "
            "for a dedicated output directory."
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_locks = OutputLocks((
        out_dir / ".anthrofold-msa.lock",
        out_csv.parent / f".{out_csv.name}.anthrofold-msa.lock",
    ))

    rows, fieldnames = read_rows(input_path)

    # Validate the CSV with the same rules the invoke notebook will apply, before
    # spending any GPU time. Otherwise a malformed row — an empty sequence column,
    # a duplicate job name — is only discovered by csv_to_jobs after the whole
    # search has run, and the output CSV is unusable.
    csv_to_jobs(input_path)

    sequences = unique_sequences(rows)

    # Anything already searched by an earlier run is reused: a large CSV takes many
    # invocations, and a failure partway through should not discard the work done.
    seq_to_dir = existing_msas(out_dir)
    todo = [s for s in sequences if s not in seq_to_dir]

    # The model discards any MSA for a chain of <= SHORT_CHAIN_MAX_LEN residues and
    # featurises it at depth 1 from the query alone, so searching one is a wasted
    # invocation. Write the same query-only a3m locally instead: every chain still
    # gets a directory, so the CSV's positional MSA columns have no gaps.
    short = [s for s in todo if len(s) <= SHORT_CHAIN_MAX_LEN]
    todo = [s for s in todo if len(s) > SHORT_CHAIN_MAX_LEN]
    if short:
        seq_to_dir.update(write_msa_dirs(dummy_msas(short), out_dir))
        print(f"{len(short)} chain(s) of <= {SHORT_CHAIN_MAX_LEN} residues written as "
              "query-only MSAs without searching (the model ignores MSAs for these).")

    chunks = [todo[i:i + args.sequences_per_request]
              for i in range(0, len(todo), args.sequences_per_request)]
    print(f"{len(rows)} row(s), {len(sequences)} unique protein sequence(s).")
    if len(todo) < len(sequences):
        print(f"  {len(sequences) - len(todo)} already on disk from an earlier run; "
              f"{len(todo)} left.")
    print(f"  {len(chunks)} invocation(s) of up to {args.sequences_per_request} sequence(s).")

    csv_root = out_csv.parent

    endpoint_name = None
    if chunks:
        session = boto3.Session(region_name=args.region)
        state = _lifecycle_state()
        endpoint_name = resolve_endpoint_name(args.endpoint_name, state=state)
        state_for_endpoint = (
            state if state.get("endpoint_name") == endpoint_name else {}
        )
        account_id = session.client("sts").get_caller_identity()["Account"]
        bucket = (
            args.bucket
            or state_for_endpoint.get("bucket")
            or f"sagemaker-{args.region}-{account_id}"
        )
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"-{uuid.uuid4().hex[:8]}"
        )
        prefix = f"{(args.prefix or 'anthrofold-msa/requests').strip('/')}/{run_id}"
        runtime, s3 = session.client("sagemaker-runtime"), session.client("s3")
        sm, logs = session.client("sagemaker"), session.client("logs")
        print(f"  endpoint: {endpoint_name}")
        print(f"  async I/O: s3://{bucket}/{prefix}/")
        # Validate the endpoint contract before entering a potentially long
        # database-readiness wait. A typo that names the folding endpoint should
        # fail in seconds, not look like hours of MSA preparation.
        async_config = endpoint_async_config(sm, endpoint_name)
        failure_path = async_config.get("OutputConfig", {}).get("S3FailurePath")
        if failure_path:
            print(f"  failure reporting: {failure_path}")
        else:
            # Prove the fallback before any readiness wait. Otherwise a legacy
            # endpoint with neither a failure destination nor readable logs can
            # spend hours looking like it is still preparing after a terminal
            # serving failure.
            try:
                _cloudwatch_server_error(
                    logs, endpoint_name, int(time.time() * 1000)
                )
            except Exception as exc:
                raise RuntimeError(
                    "This older endpoint has no S3FailurePath and CloudWatch logs "
                    "are unavailable, so terminal async failures would be invisible. "
                    "Deploy with src/deploy_msa_endpoint.py or grant "
                    "logs:FilterLogEvents."
                ) from exc
            print(
                "  WARNING: endpoint has no S3FailurePath; using CloudWatch terminal-"
                "failure detection. New endpoints created by this repo include one."
            )
        if not args.no_wait_for_ready:
            wait_until_ready(
                sm=sm,
                logs=logs,
                runtime=runtime,
                s3=s3,
                endpoint_name=endpoint_name,
                bucket=bucket,
                prefix=prefix,
                timeout_seconds=args.readiness_timeout_seconds,
                poll_seconds=args.poll_seconds,
                readiness=args.readiness,
                # Deployment already performs the canary. For an existing endpoint,
                # a ready log marker avoids queueing a redundant probe behind another
                # user's long search; this client's first real request is itself
                # validated and retried end to end.
                final_probe=False,
            )

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[{i}/{len(chunks)}] searching {len(chunk)} sequence(s)...")
        chunk_started = time.monotonic()
        input_uri = upload_jobs(s3, bucket, prefix,
                                chunk_to_jobs(chunk, i), f"msa_request_{i:04d}")
        result = None
        for attempt in range(1, args.max_invocation_attempts + 1):
            submitted_ms = int(time.time() * 1000)
            response = invoke_async(
                runtime_client=runtime,
                endpoint_name=endpoint_name,
                input_s3_uri=input_uri,
                invocation_timeout_seconds=MAX_INVOCATION_TIMEOUT,
                request_ttl_seconds=args.request_ttl_seconds,
            )
            inference_id = response.get("InferenceId", "<not returned>")
            print(
                f"[{i}/{len(chunks)}] submitted attempt "
                f"{attempt}/{args.max_invocation_attempts}; inference id "
                f"{inference_id}"
            )
            try:
                result = wait_for_result(
                    s3_client=s3,
                    output_s3_uri=response["OutputLocation"],
                    failure_s3_uri=response.get("FailureLocation"),
                    poll_seconds=args.poll_seconds,
                    max_wait_seconds=args.max_wait_seconds,
                    terminal_failure_check=terminal_failure_checker(
                        sm,
                        logs,
                        endpoint_name,
                        submitted_ms,
                        # Correlated logs are a second line of defence if
                        # SageMaker accepts a request but never writes either S3
                        # object. If this unusual response omitted its inference
                        # id, do not use an uncorrelated endpoint-wide error when a
                        # per-request S3 failure destination is available.
                        check_logs=(
                            inference_id != "<not returned>"
                            or not response.get("FailureLocation")
                        ),
                        inference_id=(
                            inference_id if inference_id != "<not returned>" else None
                        ),
                        logs_required=not response.get("FailureLocation"),
                    ),
                )
                break
            except AsyncInferenceFailure as exc:
                detail = (exc.body or str(exc)).strip()
                if (
                    _is_transient_invocation_failure(detail)
                    and attempt < args.max_invocation_attempts
                ):
                    print(
                        f"[{i}/{len(chunks)}] transient serving failure: "
                        f"{detail[:500]}"
                    )
                    print(
                        f"[{i}/{len(chunks)}] retrying in "
                        f"{args.invocation_retry_seconds}s; completed chunks are safe."
                    )
                    time.sleep(args.invocation_retry_seconds)
                    continue
                raise RuntimeError(
                    f"Invocation {i} attempt {attempt} failed: {detail[:2000]}"
                ) from exc
        if result is None:
            raise RuntimeError(
                f"Invocation {i} exhausted {args.max_invocation_attempts} attempts."
            )
        msas = extract_msas(result)
        absent = [s for s in chunk if s not in msas]
        unexpected = [s for s in msas if s not in chunk]
        if absent or unexpected:
            details = []
            if absent:
                details.append(
                    f"missing {len(absent)} requested sequence(s), e.g. "
                    f"{absent[0][:24]}..."
                )
            if unexpected:
                details.append(
                    f"included {len(unexpected)} unrequested sequence(s), e.g. "
                    f"{unexpected[0][:24]}..."
                )
            raise RuntimeError(
                f"Invocation {i} returned an invalid response: {'; '.join(details)}. "
                "Nothing is written for this chunk; completed chunks are kept and "
                "will be skipped on re-run."
            )
        # Write and checkpoint per chunk so an interrupted run resumes cleanly.
        seq_to_dir.update(write_msa_dirs(msas, out_dir))
        # Refresh the output CSV with whatever is complete so far, so an
        # interrupted run still leaves something usable rather than nothing.
        _, ready = write_rows_with_msas(rows, fieldnames, seq_to_dir, out_csv,
                                        relative_to=csv_root, skip_incomplete=True)
        elapsed = time.monotonic() - chunk_started
        print(f"[{i}/{len(chunks)}] wrote {len(msas)} MSA(s) in {elapsed / 60:.1f} min "
              f"— {ready}/{len(rows)} row(s) now complete")

    print(f"\nMSAs for {len(seq_to_dir)} sequence(s) under {out_dir}")

    # Strict this time: at the end a missing sequence is a real error, not a
    # row that has simply not been reached yet.
    written, ready = write_rows_with_msas(rows, fieldnames, seq_to_dir, out_csv,
                                          relative_to=csv_root)
    # Prove the generated bundle is exactly what notebook 2 can consume, including
    # positional chain/MSA matching and relative-path resolution.
    output_jobs = csv_to_jobs(written)
    report = validate_msas(output_jobs, base_dir=csv_root)
    sendability = validate_notebook_sendability(output_jobs, base_dir=csv_root)
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"\nWrote notebook-ready CSV: {written} ({ready} row(s))")
    print(f"Validated {report['n_chains']} chain MSA reference(s).")
    print(
        "Largest single-job request after inlining: "
        f"{sendability['largest_single_job_payload_bytes'] / 1e6:.1f} MB."
    )
    if sendability["excluded_jobs"]:
        print(
            f"WARNING: notebook 2 will exclude {len(sendability['excluded_jobs'])} "
            "row(s) that exceed its 2048-residue or 25 MB Marketplace request limit:"
        )
        for item in sendability["excluded_jobs"]:
            print(
                f"  - {item['name']}: {item['total_residues']} residues, "
                f"{item['payload_bytes'] / 1e6:.1f} MB ({item['reason']})"
            )
        print("The MSA bundle is valid; split or remove those rows if they must be folded.")
    print("Point INPUT_PATH in 2-invoke-endpoint.ipynb at that file, and keep it "
          "alongside the MSA folder.")
    output_locks.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; completed MSA chunks are safe to resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
