import json
import re
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# S3 error codes worth retrying while polling for an async result (transient).
_RETRYABLE_S3 = {"SlowDown", "RequestTimeout", "ServiceUnavailable", "InternalError", "Throttling", "ThrottlingException"}


class AsyncInferenceFailure(RuntimeError):
    def __init__(self, failure_s3_uri, body):
        super().__init__(f"Async inference failed; failure object: {failure_s3_uri}")
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


def plan_batches(jobs, batch_size=4, max_residues=2048, sort_by_size=True, limit_batches=None):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    selected = [job for job in jobs if total_residues(job) <= max_residues]
    excluded = [job for job in jobs if total_residues(job) > max_residues]
    if sort_by_size:
        selected.sort(key=lambda job: (total_residues(job), str(job.get("name", ""))))

    batches = [selected[i : i + batch_size] for i in range(0, len(selected), batch_size)]
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
    invocation_timeout_seconds=3600,
    request_ttl_seconds=21600,  # 6h — SageMaker drops requests that wait longer in the queue
    output_path_extension=None,
):
    if invocation_timeout_seconds > 3600:
        raise ValueError("SageMaker async InvocationTimeoutSeconds max is 3600.")

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
    max_wait_seconds=4200,
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

    deadline = time.time() + max_wait_seconds
    start = time.time()
    while time.time() < deadline:
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

        # One heartbeat per minute, not every poll, to keep notebook output tidy.
        elapsed = int(time.time() - start)
        if elapsed > 0 and elapsed % 60 < poll_seconds:
            print(f"  ...still running ({elapsed // 60} min)")
        time.sleep(poll_seconds)

    raise TimeoutError(f"No async result appeared within {max_wait_seconds}s.")


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
