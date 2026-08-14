#!/usr/bin/env python3
"""Deploy, inspect, wait for, and delete an AnthroFold MSA endpoint.

The MSA model package has an intentionally unusual startup: SageMaker reports
the endpoint ``InService`` while the container is still copying and preparing
roughly 1 TB of database files. During that window
``/ping`` returns 200, but inference returns 503.  This client therefore treats
SageMaker health and application readiness as two separate states.

Typical use::

    python src/msa_endpoint.py deploy --role-arn arn:aws:iam::123:role/MyRole
    python src/msa_endpoint.py status
    python src/msa_endpoint.py wait
    python src/msa_endpoint.py delete

``deploy`` writes ``msa_endpoint_state.json`` as soon as an endpoint request is
created and writes ``msa_endpoint_name.txt`` only after the MSA service is truly
ready.  Both files live at the repository root by default, independent of the
caller's working directory.
"""

import argparse
import getpass
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_FILE = REPO_ROOT / "msa_endpoint_state.json"
DEFAULT_SIDECAR = REPO_ROOT / "msa_endpoint_name.txt"
DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:us-east-1:038462780959:"
    "model-package/anthrofold-msa-search/2"
)
DEFAULT_INSTANCE_TYPE = "ml.g5.12xlarge"
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_READY_TIMEOUT = 4 * 60 * 60
READY_MARKER = "AnthroFold MSA Search is ready"
FAILED_MARKER = "Failed to initialize MSA service"
TRANSIENT_INVOCATION_MARKERS = (
    "server error (0)",
    "could not get a response",
    "connection reset",
    "connection refused",
    "internalfailure",
    "internal server error",
    "modelerror",
    "service unavailable",
    "temporarily unavailable",
    "throttl",
    "timed out",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
)
CAPACITY_MARKERS = (
    "InsufficientInstanceCapacity",
    "Unable to provision requested ML compute capacity",
    "does not have sufficient capacity",
)
STARTUP_MILESTONES = (
    "ColabFold DB sync attempt",
    "Verified pinned ColabFold GPU databases",
    "Warming mmseqs2-GPU database servers",
    "gpuservers warm",
    FAILED_MARKER,
    READY_MARKER,
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_slug():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _safe_name(value, limit=63):
    value = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-").lower()
    value = re.sub(r"-+", "-", value)
    if not value:
        value = "anthrofold-msa"
    return value[:limit].rstrip("-")


def _attempt_name(base, attempt):
    suffix = "" if attempt == 1 else f"-r{attempt}"
    return _safe_name(base, 63 - len(suffix)) + suffix


def _atomic_write_text(path, text):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_state(path, state):
    state = dict(state)
    state["updated_at"] = _utc_now()
    _atomic_write_text(path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def _read_state(path, required=False):
    path = Path(path).resolve()
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Endpoint state file not found: {path}. Pass --endpoint-name or "
                "run the deploy command first."
            )
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read endpoint state file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Endpoint state file must contain a JSON object: {path}")
    return value


def _clients(region):
    import boto3

    session = boto3.Session(region_name=region)
    return {
        "session": session,
        "sm": session.client("sagemaker"),
        "runtime": session.client("sagemaker-runtime"),
        "s3": session.client("s3"),
        "logs": session.client("logs"),
        "sts": session.client("sts"),
    }


def _error_code(exc):
    return getattr(exc, "response", {}).get("Error", {}).get("Code", "")


def _not_found(exc):
    code = _error_code(exc)
    message = str(exc)
    return code in {"ResourceNotFound", "ResourceNotFoundException"} or "Could not find" in message


def _is_capacity_failure(reason):
    return any(marker.lower() in (reason or "").lower() for marker in CAPACITY_MARKERS)


def _load_endpoint_name(args, state=None):
    state = state or {}
    explicit = getattr(args, "endpoint_name", None)
    if explicit:
        return explicit.strip()
    if state.get("endpoint_name"):
        return str(state["endpoint_name"]).strip()
    sidecar = Path(args.sidecar).resolve()
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    raise ValueError(
        "Endpoint name is not set. Pass --endpoint-name, run deploy first, or "
        f"write it to {sidecar}."
    )


def _describe_endpoint(sm, endpoint_name):
    try:
        return sm.describe_endpoint(EndpointName=endpoint_name)
    except Exception as exc:
        if _not_found(exc):
            return None
        raise


def _model_from_config(sm, config_name):
    if not config_name:
        return None
    try:
        config = sm.describe_endpoint_config(EndpointConfigName=config_name)
    except Exception as exc:
        if _not_found(exc):
            return None
        raise
    variants = config.get("ProductionVariants") or []
    return variants[0].get("ModelName") if variants else None


def _named_resource_exists(describe_call, **kwargs):
    """Return whether an exact SageMaker resource exists; preserve real errors."""
    try:
        describe_call(**kwargs)
        return True
    except Exception as exc:
        if _not_found(exc):
            return False
        raise


def _ensure_bucket(s3, bucket, region, create_if_missing=True):
    """Validate the async I/O bucket and create the default bucket if absent."""
    from botocore.exceptions import ClientError

    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = _error_code(exc)
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchBucket", "NotFound"} or status == 404:
            if not create_if_missing:
                print(
                    f"S3 bucket s3://{bucket} does not exist; deployment would "
                    "create it (dry-run left it unchanged)."
                )
                return
            kwargs = {"Bucket": bucket}
            if region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
            print(f"Creating S3 bucket s3://{bucket} in {region}.")
            s3.create_bucket(**kwargs)
        else:
            raise RuntimeError(
                f"Cannot access S3 bucket s3://{bucket}: {exc}. The person running "
                "the client needs to write requests and read async results."
            ) from exc

    try:
        location = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint")
        actual_region = location or "us-east-1"
        if actual_region != region:
            raise ValueError(
                f"S3 bucket s3://{bucket} is in {actual_region}, but the endpoint is "
                f"in {region}. Use a same-region bucket."
            )
    except ClientError as exc:
        if _error_code(exc) not in {"AccessDenied", "403"}:
            raise
        print("WARNING: cannot verify the bucket region (s3:GetBucketLocation denied).")


def _validate_package(sm, package_arn, instance_type):
    desc = sm.describe_model_package(ModelPackageName=package_arn)
    if desc.get("ModelPackageStatus") != "Completed":
        raise RuntimeError(
            f"Model package is not deployable: status={desc.get('ModelPackageStatus')!r}"
        )
    spec = desc.get("InferenceSpecification") or {}
    supported = (
        spec.get("SupportedRealtimeInferenceInstanceTypes")
        or spec.get("SupportedTransformInstanceTypes")
        or []
    )
    if supported and instance_type not in supported:
        raise ValueError(
            f"{instance_type} is not allowed by this model package. Allowed: "
            + ", ".join(supported)
        )
    return desc, supported


def _cloudwatch_events(logs, endpoint_name, phrase, start_ms, limit=20):
    """Return matching events, None when the caller cannot read endpoint logs."""
    from botocore.exceptions import ClientError

    group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    phrases = phrase if isinstance(phrase, (list, tuple)) else (phrase,)
    try:
        return logs.filter_log_events(
            logGroupName=group,
            filterPattern=" ".join(json.dumps(value) for value in phrases),
            startTime=max(0, int(start_ms)),
            limit=limit,
        ).get("events", [])
    except ClientError as exc:
        if _error_code(exc) in {
            "ResourceNotFoundException",
            "AccessDeniedException",
            "AccessDenied",
            "UnrecognizedClientException",
            "Throttling",
            "ThrottlingException",
            "ServiceUnavailableException",
        }:
            return None
        raise


def _latest_startup_milestone(logs, endpoint_name, start_ms):
    latest = None
    logs_accessible = False
    for phrase in STARTUP_MILESTONES:
        events = _cloudwatch_events(logs, endpoint_name, phrase, start_ms, limit=10)
        if events is None:
            continue
        logs_accessible = True
        for event in events:
            if latest is None or event.get("timestamp", 0) > latest.get("timestamp", 0):
                latest = event
    return logs_accessible, latest


def _application_log_state(logs, endpoint_name, start_ms):
    """Return (logs_accessible, ready_event, failed_event) with two cheap filters."""
    ready_events = _cloudwatch_events(
        logs, endpoint_name, READY_MARKER, start_ms, limit=1
    )
    failed_events = _cloudwatch_events(
        logs, endpoint_name, FAILED_MARKER, start_ms, limit=1
    )
    accessible = ready_events is not None or failed_events is not None
    ready = max(ready_events or [], key=lambda e: e.get("timestamp", 0), default=None)
    failed = max(failed_events or [], key=lambda e: e.get("timestamp", 0), default=None)
    return accessible, ready, failed


def _s3_uri_parts(uri):
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    bucket, sep, key = uri[5:].partition("/")
    if not bucket or not sep or not key:
        raise ValueError(f"Expected an S3 object URI, got {uri!r}")
    return bucket, key


def _get_s3_text(s3, uri):
    from botocore.exceptions import ClientError

    if not uri:
        return None
    bucket, key = _s3_uri_parts(uri)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _error_code(exc) in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    return obj["Body"].read().decode("utf-8", errors="replace")


def _is_transient_invocation_failure(detail):
    lower = (detail or "").lower()
    return any(marker in lower for marker in TRANSIENT_INVOCATION_MARKERS)


def _terminal_serving_failure(
    sm, logs, endpoint_name, start_ms, inference_id=None
):
    """Return a terminal endpoint/handoff diagnostic, including legacy endpoints."""
    desc = _describe_endpoint(sm, endpoint_name)
    if desc is None:
        return f"Endpoint no longer exists: {endpoint_name}"
    status = desc.get("EndpointStatus")
    if status in {"Failed", "Deleting", "OutOfService"}:
        return (
            f"Endpoint entered {status}: "
            f"{desc.get('FailureReason', 'no failure reason returned')}"
        )
    events = _cloudwatch_events(
        logs,
        endpoint_name,
        (
            ("Received server error", inference_id)
            if inference_id
            else "Received server error"
        ),
        start_ms,
        limit=1,
    )
    if events:
        return events[0].get("message", "").strip()
    return None


def _readiness_probe(
    runtime,
    s3,
    endpoint_name,
    bucket,
    prefix,
    sm=None,
    logs=None,
    poll_seconds=5,
    timeout=300,
    max_attempts=3,
):
    """Invoke a tiny query and return (ready, detail).

    The package gates ``/invocations`` with HTTP 503 until database preparation is
    complete.  Once ready, a four-residue query is the smallest useful end-to-end
    check and normally returns a query-only A3M quickly.
    """
    sequence = "ACDE"
    payload = [{
        "name": "readiness_probe",
        "sequences": [{
            "proteinChain": {
                "sequence": sequence,
                "count": 1,
                "modifications": [],
            }
        }],
        "covalent_bonds": [],
    }]
    last_failure = None
    for attempt in range(1, max_attempts + 1):
        submitted_ms = int(time.time() * 1000)
        probe_id = uuid.uuid4().hex
        key = f"{prefix.strip('/')}/readiness/{probe_id}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
        )
        response = runtime.invoke_endpoint_async(
            EndpointName=endpoint_name,
            InputLocation=f"s3://{bucket}/{key}",
            ContentType="application/json",
            InvocationTimeoutSeconds=min(timeout, 3600),
            RequestTTLSeconds=min(timeout, 21600),
            S3OutputPathExtension=f"readiness/{probe_id}",
        )
        inference_id = response.get("InferenceId", "<not returned>")
        print(
            f"End-to-end readiness probe {attempt}/{max_attempts} submitted "
            f"(inference id: {inference_id})."
        )
        if not response.get("FailureLocation"):
            print(
                "WARNING: this endpoint has no async failure destination; a failed "
                "probe will use endpoint/CloudWatch diagnostics plus a bounded "
                "timeout. Endpoints created by this client always configure one."
            )

        deadline = time.monotonic() + timeout
        started = time.monotonic()
        reported_minute = 0
        while time.monotonic() < deadline:
            failure = _get_s3_text(s3, response.get("FailureLocation"))
            if failure is not None:
                lower = failure.lower()
                if (
                    "msa service not ready" in lower
                    or "status code: 503" in lower
                    or "(503)" in lower
                    or " 503" in lower
                ):
                    return False, failure.strip()[:500]
                last_failure = failure.strip()[:1000]
                if _is_transient_invocation_failure(last_failure) and attempt < max_attempts:
                    print(
                        f"Transient readiness-probe failure; retrying "
                        f"({last_failure[:300]})."
                    )
                    time.sleep(max(1, poll_seconds))
                    break
                raise RuntimeError(
                    "The readiness probe failed after reaching the serving layer: "
                    f"{last_failure}"
                )

            output = _get_s3_text(s3, response.get("OutputLocation"))
            if output is not None:
                try:
                    result = json.loads(output)
                    chain = result[0]["sequences"][0]["proteinChain"]
                    if chain.get("sequence") != sequence or "unpairedMsa" not in chain:
                        raise ValueError(
                            "response did not contain the probe sequence and inline MSA"
                        )
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"The MSA readiness probe returned an unexpected response: {exc}"
                    ) from exc
                return True, "tiny async MSA probe succeeded"
            if sm is not None and logs is not None:
                diagnostic = _terminal_serving_failure(
                    sm,
                    logs,
                    endpoint_name,
                    submitted_ms,
                    inference_id=inference_id,
                )
                if diagnostic:
                    last_failure = diagnostic[:1000]
                    if (
                        _is_transient_invocation_failure(last_failure)
                        and attempt < max_attempts
                    ):
                        print(
                            "Transient readiness-probe handoff failure; retrying "
                            f"({last_failure[:300]})."
                        )
                        time.sleep(max(1, poll_seconds))
                        break
                    raise RuntimeError(
                        "The readiness probe failed before producing an S3 object: "
                        f"{last_failure}"
                    )
            elapsed_minute = int((time.monotonic() - started) / 60)
            if elapsed_minute > reported_minute:
                reported_minute = elapsed_minute
                print(
                    f"Readiness probe {attempt}/{max_attempts} is still running "
                    f"({elapsed_minute} min)."
                )
            time.sleep(max(1, poll_seconds))
        else:
            last_failure = (
                f"probe {inference_id} produced neither a success nor failure object "
                f"within {timeout}s"
            )
            if attempt < max_attempts:
                print(f"{last_failure}; retrying.")

    raise TimeoutError(
        f"End-to-end readiness failed after {max_attempts} attempts: {last_failure}"
    )


def wait_until_ready(
    *,
    sm,
    logs,
    runtime,
    s3,
    endpoint_name,
    bucket=None,
    prefix="anthrofold-msa",
    timeout_seconds=DEFAULT_READY_TIMEOUT,
    poll_seconds=60,
    readiness="auto",
    final_probe=True,
    state=None,
    state_path=None,
):
    """Wait for endpoint allocation and true application readiness."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    last_status = None
    last_progress_minute = -1
    log_start_ms = None
    logs_known_accessible = None

    print(
        "Waiting for two stages: SageMaker capacity, then preparation of the "
        "~1 TB search database. InService alone is not ready."
    )
    while time.monotonic() < deadline:
        desc = _describe_endpoint(sm, endpoint_name)
        if desc is None:
            raise RuntimeError(f"Endpoint no longer exists: {endpoint_name}")
        if log_start_ms is None:
            created = desc.get("CreationTime")
            log_start_ms = (
                max(0, int(created.timestamp() * 1000) - 5000)
                if created is not None
                else int(time.time() * 1000)
            )
        status = desc.get("EndpointStatus")
        if status != last_status:
            print(f"[{int((time.monotonic() - started) / 60)} min] SageMaker: {status}")
            last_status = status
        if status == "Failed":
            raise RuntimeError(
                f"Endpoint failed: {desc.get('FailureReason', 'no failure reason returned')}"
            )
        if status in {"Deleting", "OutOfService"}:
            raise RuntimeError(f"Endpoint entered terminal state {status}")

        if status == "InService":
            if readiness in {"auto", "logs"}:
                accessible, ready_event, failed_event = _application_log_state(
                    logs, endpoint_name, log_start_ms
                )
                logs_known_accessible = accessible
                if ready_event:
                    message = ready_event.get("message", "").strip().splitlines()[0]
                    if not final_probe:
                        print(f"Application ready: {message}")
                        return {"method": "cloudwatch", "message": message}
                    if not bucket:
                        raise ValueError(
                            "A bucket is required for the final end-to-end readiness "
                            "probe. Pass --bucket or deploy with this client."
                        )
                    print(f"Database ready: {message}")
                    ready, detail = _readiness_probe(
                        runtime,
                        s3,
                        endpoint_name,
                        bucket,
                        prefix,
                        sm=sm,
                        logs=logs,
                        poll_seconds=min(5, poll_seconds),
                    )
                    if ready:
                        print(f"Application ready: {detail}")
                        return {
                            "method": "cloudwatch+async_probe",
                            "message": f"{message}; {detail}",
                        }
                if failed_event:
                    message = failed_event.get("message", "").strip().splitlines()[0]
                    raise RuntimeError(f"Container initialization failed: {message}")

            use_probe = readiness == "probe" or (
                readiness == "auto" and logs_known_accessible is False
            )
            if use_probe:
                if not bucket:
                    raise ValueError(
                        "A bucket is required for readiness probing when CloudWatch "
                        "logs are unavailable. Pass --bucket or deploy with this client."
                    )
                ready, detail = _readiness_probe(
                    runtime,
                    s3,
                    endpoint_name,
                    bucket,
                    prefix,
                    sm=sm,
                    logs=logs,
                    poll_seconds=min(5, poll_seconds),
                )
                if ready:
                    print(f"Application ready: {detail}")
                    return {"method": "async_probe", "message": detail}

        elapsed_minute = int((time.monotonic() - started) / 60)
        if elapsed_minute != last_progress_minute and elapsed_minute % 5 == 0:
            if status == "InService":
                print(
                    f"[{elapsed_minute} min] endpoint is healthy; database preparation "
                    "is still in progress"
                )
            elif status in {"Creating", "Updating", "SystemUpdating"}:
                print(
                    f"[{elapsed_minute} min] SageMaker is still provisioning "
                    f"{endpoint_name}; no application instance is available yet"
                )
            last_progress_minute = elapsed_minute
        if state is not None and state_path is not None:
            state["sagemaker_status"] = status
            state["application_ready"] = False
            _write_state(state_path, state)
        time.sleep(max(1, poll_seconds))

    raise TimeoutError(
        f"Endpoint did not become application-ready within {timeout_seconds}s. It was "
        "left running; use the status/wait commands to inspect or continue waiting."
    )


def _wait_endpoint_deleted(sm, endpoint_name, timeout_seconds=3600, poll_seconds=15):
    started = time.monotonic()
    deadline = time.monotonic() + timeout_seconds
    reported_minute = 0
    while time.monotonic() < deadline:
        if _describe_endpoint(sm, endpoint_name) is None:
            return
        elapsed_minute = int((time.monotonic() - started) / 60)
        if elapsed_minute > reported_minute:
            reported_minute = elapsed_minute
            print(f"Endpoint deletion is still running ({elapsed_minute} min).")
        time.sleep(max(1, poll_seconds))
    raise TimeoutError(f"Endpoint was not deleted within {timeout_seconds}s: {endpoint_name}")


def _delete_if_present(call, label, **kwargs):
    try:
        call(**kwargs)
        print(f"Deleted {label}.")
        return True
    except Exception as exc:
        if _not_found(exc):
            print(f"{label} was already absent.")
            return False
        raise


def _cleanup_resources(sm, endpoint_name=None, config_name=None, model_name=None, wait=True):
    if endpoint_name and _describe_endpoint(sm, endpoint_name) is not None:
        delete_submit_deadline = time.monotonic() + 3600
        while True:
            try:
                sm.delete_endpoint(EndpointName=endpoint_name)
                print(f"Deleting endpoint {endpoint_name}.")
                break
            except Exception as exc:
                # SageMaker refuses deletion while CreateEndpoint is still resolving.
                if "Creating" in str(exc) or "Cannot update in-progress" in str(exc):
                    if time.monotonic() >= delete_submit_deadline:
                        raise TimeoutError(
                            f"Endpoint {endpoint_name} remained in an in-progress "
                            "state for an hour; cleanup stopped instead of waiting "
                            "indefinitely. Run the delete command again later."
                        ) from exc
                    print("Endpoint is still Creating; waiting until SageMaker resolves it.")
                    time.sleep(30)
                    continue
                if _not_found(exc):
                    break
                raise
        if wait:
            _wait_endpoint_deleted(sm, endpoint_name)

    if config_name:
        _delete_if_present(
            sm.delete_endpoint_config,
            f"endpoint config {config_name}",
            EndpointConfigName=config_name,
        )
    if model_name:
        _delete_if_present(sm.delete_model, f"model {model_name}", ModelName=model_name)


def _deploy(args):
    arn_parts = args.model_package_arn.split(":", 5)
    package_region = arn_parts[3] if len(arn_parts) == 6 else ""
    if package_region and package_region != args.region:
        raise ValueError(
            f"Model package is in {package_region}, but --region is {args.region}. "
            "Deploy the endpoint in the package's region."
        )
    clients = _clients(args.region)
    sm, s3, sts = clients["sm"], clients["s3"], clients["sts"]
    account_id = sts.get_caller_identity()["Account"]
    role_arn = args.role_arn or os.environ.get("ANTHROFOLD_EXECUTION_ROLE_ARN")
    if not role_arn:
        raise ValueError(
            "Execution role is required. Pass --role-arn or set "
            "ANTHROFOLD_EXECUTION_ROLE_ARN. The deployer also needs iam:PassRole."
        )

    state_path = Path(args.state_file).resolve()
    sidecar = Path(args.sidecar).resolve()
    previous_state = _read_state(state_path)
    live_previous = []
    previous_endpoints = set()
    if previous_state.get("endpoint_name"):
        state_endpoint = str(previous_state["endpoint_name"]).strip()
        if state_endpoint:
            previous_endpoints.add(state_endpoint)
    if sidecar.exists():
        sidecar_endpoint = sidecar.read_text(encoding="utf-8").strip()
        if sidecar_endpoint:
            previous_endpoints.add(sidecar_endpoint)
    previous_config = previous_state.get("endpoint_config_name")
    previous_model = previous_state.get("model_name")
    for previous_endpoint in sorted(previous_endpoints):
        if _describe_endpoint(sm, previous_endpoint) is not None:
            live_previous.append(f"endpoint {previous_endpoint}")
    if previous_config and _named_resource_exists(
        sm.describe_endpoint_config, EndpointConfigName=previous_config
    ):
        live_previous.append(f"endpoint config {previous_config}")
    if previous_model and _named_resource_exists(
        sm.describe_model, ModelName=previous_model
    ):
        live_previous.append(f"model {previous_model}")
    if live_previous:
        detail = ", ".join(live_previous)
        if args.dry_run:
            print(
                f"WARNING: {state_path} already tracks live resources ({detail}); "
                "a real deploy would stop before overwriting that teardown record."
            )
        else:
            raise RuntimeError(
                f"Refusing to overwrite {state_path}; it still tracks {detail}. "
                "Reuse the endpoint, run src/teardown_msa_endpoint.py, or pass a "
                "different --state-file and --sidecar for a deliberate second deployment."
            )

    _, supported = _validate_package(sm, args.model_package_arn, args.instance_type)
    bucket = args.bucket or f"sagemaker-{args.region}-{account_id}"
    if not bucket.startswith("sagemaker-"):
        print(
            "WARNING: this package's standard execution role is scoped to sagemaker-* "
            "buckets. A different bucket works only if the role was extended."
        )
    output_prefix = (args.output_prefix or "anthrofold-msa/async-output").strip("/")
    _ensure_bucket(s3, bucket, args.region, create_if_missing=not args.dry_run)

    base = args.name or _safe_name(
        f"anthrofold-msa-{getpass.getuser()}-{_timestamp_slug()}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    model_name = _safe_name(f"{base}-model")
    state = {
        "schema": "anthrofold-msa-endpoint/1",
        "region": args.region,
        "model_package_arn": args.model_package_arn,
        "execution_role_arn": role_arn,
        "instance_type": args.instance_type,
        "bucket": bucket,
        "output_prefix": output_prefix,
        "model_name": model_name,
        "application_ready": False,
        "created_at": _utc_now(),
    }

    print(f"Account:       {account_id}")
    print(f"Package:       {args.model_package_arn}")
    print(f"Role:          {role_arn}")
    print(f"Instance:      {args.instance_type}")
    if supported:
        print(f"Allowed sizes: {', '.join(supported)}")
    print(f"Async output:  s3://{bucket}/{output_prefix}/<endpoint>/success/")
    print(f"Async failure: s3://{bucket}/{output_prefix}/<endpoint>/failure/")
    print("Network isolation: disabled (required for S3 database sync)")
    print("Max concurrent invocations per instance: 1")

    if args.dry_run:
        state["dry_run"] = True
        state["planned_endpoint_name"] = _attempt_name(base, 1)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    _write_state(state_path, state)
    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"ModelPackageName": args.model_package_arn},
            ExecutionRoleArn=role_arn,
            EnableNetworkIsolation=False,
        )
        print(f"Created model {model_name}.")
    except Exception:
        # Keep the state file: it tells delete/status exactly what was planned.
        raise

    for attempt in range(1, args.max_attempts + 1):
        endpoint_name = _attempt_name(base, attempt)
        config_name = _safe_name(f"{endpoint_name}-async")
        attempt_output = f"{output_prefix}/{endpoint_name}"
        success_uri = f"s3://{bucket}/{attempt_output}/success/"
        failure_uri = f"s3://{bucket}/{attempt_output}/failure/"
        state.update({
            "attempt": attempt,
            "max_attempts": args.max_attempts,
            "endpoint_name": endpoint_name,
            "endpoint_config_name": config_name,
            "s3_output_uri": success_uri,
            "s3_failure_uri": failure_uri,
            "sagemaker_status": "Creating",
        })
        _write_state(state_path, state)

        print(f"\nCapacity attempt {attempt}/{args.max_attempts}: {endpoint_name}")
        try:
            sm.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[{
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": args.instance_type,
                    "InitialVariantWeight": 1.0,
                    "ModelDataDownloadTimeoutInSeconds": 3600,
                    "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
                }],
                AsyncInferenceConfig={
                    "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1},
                    "OutputConfig": {
                        "S3OutputPath": success_uri,
                        "S3FailurePath": failure_uri,
                    },
                },
            )
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
        except Exception:
            _cleanup_resources(sm, config_name=config_name)
            raise

        if args.no_wait:
            print(f"Deployment submitted. Continue with: python src/msa_endpoint.py wait")
            return 0

        try:
            result = wait_until_ready(
                sm=sm,
                logs=clients["logs"],
                runtime=clients["runtime"],
                s3=s3,
                endpoint_name=endpoint_name,
                bucket=bucket,
                prefix=f"anthrofold-msa/{endpoint_name}",
                timeout_seconds=args.ready_timeout_seconds,
                poll_seconds=args.poll_seconds,
                readiness=args.readiness,
                final_probe=True,
                state=state,
                state_path=state_path,
            )
        except RuntimeError as exc:
            desc = _describe_endpoint(sm, endpoint_name)
            reason = (desc or {}).get("FailureReason", str(exc))
            if _is_capacity_failure(reason):
                state["sagemaker_status"] = "Failed"
                state["failure_reason"] = reason
                _write_state(state_path, state)
                print(f"Capacity unavailable: {reason}")
                _cleanup_resources(
                    sm,
                    endpoint_name=endpoint_name,
                    config_name=config_name,
                    wait=True,
                )
                if attempt == args.max_attempts:
                    break
                print(f"Retrying in {args.capacity_backoff_seconds}s.")
                time.sleep(args.capacity_backoff_seconds)
                continue
            state["failure_reason"] = reason
            _write_state(state_path, state)
            print(
                "Non-capacity failure. The endpoint is left recorded in the state "
                "file for inspection or exact teardown."
            )
            raise
        except (KeyboardInterrupt, TimeoutError):
            print(
                f"\nEndpoint left running: {endpoint_name}\n"
                "Resume with `python src/msa_endpoint.py wait`, or remove it with "
                "`python src/msa_endpoint.py delete`."
            )
            raise

        state["sagemaker_status"] = "InService"
        state["application_ready"] = True
        state["readiness"] = result
        state["ready_at"] = _utc_now()
        _write_state(state_path, state)
        _atomic_write_text(sidecar, endpoint_name + "\n")
        print(f"\nREADY: {endpoint_name}")
        print(f"Wrote endpoint name to {sidecar}")
        print(
            "Generate MSAs with `python src/msa_client.py --input <input.csv>`; "
            "the client will find this endpoint and the default sagemaker-* bucket."
        )
        return 0

    _cleanup_resources(sm, model_name=model_name)
    raise RuntimeError(
        f"No {args.instance_type} capacity after {args.max_attempts} attempts. "
        "All failed endpoint/config resources were cleaned up."
    )


def _status(args):
    state_path = Path(args.state_file).resolve()
    state = _read_state(state_path)
    endpoint_name = _load_endpoint_name(args, state)
    clients = _clients(args.region)
    desc = _describe_endpoint(clients["sm"], endpoint_name)
    if desc is None:
        print(f"Endpoint does not exist: {endpoint_name}")
        return 1
    status = desc.get("EndpointStatus")
    print(f"Endpoint:   {endpoint_name}")
    print(f"SageMaker:  {status}")
    if desc.get("FailureReason"):
        print(f"Failure:    {desc['FailureReason']}")
    created = desc.get("CreationTime")
    start_ms = int(created.timestamp() * 1000) if created else 0
    accessible, latest = _latest_startup_milestone(
        clients["logs"], endpoint_name, start_ms
    )
    if latest:
        message = latest.get("message", "").strip().splitlines()[0]
        print(f"Application: {'ready' if READY_MARKER in message else 'preparing'}")
        print(f"Milestone:  {message}")
    elif accessible:
        print("Application: starting (no database milestone yet)")
    elif state.get("endpoint_name") == endpoint_name and state.get("application_ready"):
        print("Application: ready (from local lifecycle state)")
    else:
        print("Application: unknown (CloudWatch Logs unavailable)")
        print("Run `python src/msa_endpoint.py wait` for a permissions-compatible check.")
    return 0


def _wait(args):
    state_path = Path(args.state_file).resolve()
    stored_state = _read_state(state_path)
    endpoint_name = _load_endpoint_name(args, stored_state)
    # An explicitly selected external endpoint must not inherit another
    # deployment's config/model/bucket into its lifecycle record.
    state = (
        stored_state
        if not stored_state or stored_state.get("endpoint_name") == endpoint_name
        else {}
    )
    clients = _clients(args.region)
    account_id = clients["sts"].get_caller_identity()["Account"]
    bucket = args.bucket or state.get("bucket") or f"sagemaker-{args.region}-{account_id}"
    result = wait_until_ready(
        sm=clients["sm"],
        logs=clients["logs"],
        runtime=clients["runtime"],
        s3=clients["s3"],
        endpoint_name=endpoint_name,
        bucket=bucket,
        prefix=f"anthrofold-msa/{endpoint_name}",
        timeout_seconds=args.ready_timeout_seconds,
        poll_seconds=args.poll_seconds,
        readiness=args.readiness,
        final_probe=True,
        state=state,
        state_path=state_path,
    )
    desc = _describe_endpoint(clients["sm"], endpoint_name) or {}
    config_name = desc.get("EndpointConfigName")
    state.update({
        "schema": "anthrofold-msa-endpoint/1",
        "endpoint_name": endpoint_name,
        "endpoint_config_name": config_name,
        "model_name": _model_from_config(clients["sm"], config_name),
        "region": args.region,
        "bucket": bucket,
        "sagemaker_status": "InService",
        "application_ready": True,
        "readiness": result,
        "ready_at": _utc_now(),
    })
    _write_state(state_path, state)
    _atomic_write_text(args.sidecar, endpoint_name + "\n")
    print(f"READY: {endpoint_name}")
    return 0


def _delete(args):
    state_path = Path(args.state_file).resolve()
    state = _read_state(state_path)
    try:
        endpoint_name = _load_endpoint_name(args, state)
    except ValueError:
        # A process can be interrupted after CreateModel but before the first
        # endpoint attempt. The state record is still enough to clean that
        # partial deployment even though no endpoint name exists yet.
        if state.get("endpoint_config_name") or state.get("model_name"):
            endpoint_name = None
        else:
            raise
    state_matches_endpoint = state.get("endpoint_name") == endpoint_name
    clients = _clients(args.region)
    sm = clients["sm"]
    desc = _describe_endpoint(sm, endpoint_name) if endpoint_name else None
    if desc:
        config_name = desc.get("EndpointConfigName")
        model_name = _model_from_config(sm, config_name)
        # State is only a fallback when it describes this exact endpoint. Never
        # let an explicit --endpoint-name borrow another deployment's resources.
        if state_matches_endpoint:
            config_name = config_name or state.get("endpoint_config_name")
            model_name = model_name or state.get("model_name")
    elif state_matches_endpoint:
        # The endpoint may already be gone while its dependent config/model remain.
        config_name = state.get("endpoint_config_name")
        model_name = state.get("model_name") or _model_from_config(sm, config_name)
    else:
        config_name = None
        model_name = None

    print("Exact SageMaker resources selected for deletion:")
    print(f"  endpoint:        {endpoint_name or '<not created>'}")
    print(f"  endpoint config: {config_name or '<not found>'}")
    print(f"  model:           {model_name or '<not found>'}")
    sys.stdout.flush()
    if desc is None and not config_name and not model_name:
        raise RuntimeError(
            "No endpoint, endpoint config, or model matched the supplied lifecycle "
            "state/name; nothing was deleted."
        )
    if not args.yes:
        if not sys.stdin.isatty():
            raise RuntimeError("Refusing non-interactive deletion without --yes.")
        answer = input("Delete these resources? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled; nothing was deleted.")
            return 1

    _cleanup_resources(
        sm,
        endpoint_name=endpoint_name,
        config_name=None if args.keep_config else config_name,
        model_name=None if args.keep_model else model_name,
        wait=True,
    )
    sidecar = Path(args.sidecar).resolve()
    if (
        endpoint_name
        and sidecar.exists()
        and sidecar.read_text(encoding="utf-8").strip() == endpoint_name
    ):
        sidecar.unlink()
        print(f"Removed local endpoint sidecar {sidecar}.")
    if state_matches_endpoint or not state:
        state.update({
            "endpoint_name": endpoint_name,
            "endpoint_config_name": config_name,
            "model_name": model_name,
            "sagemaker_status": "Deleted",
            "application_ready": False,
            "deleted_at": _utc_now(),
        })
        _write_state(state_path, state)
        print("Teardown complete. The state JSON was retained as an audit record.")
    else:
        print(
            "Teardown complete. The lifecycle state file belongs to a different "
            "endpoint and was left unchanged."
        )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="msa_endpoint",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help="SageMaker region.")
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help=f"Lifecycle state JSON (default: {DEFAULT_STATE_FILE}).",
    )
    parser.add_argument(
        "--sidecar",
        default=str(DEFAULT_SIDECAR),
        help=f"Ready endpoint-name file (default: {DEFAULT_SIDECAR}).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    deploy = sub.add_parser("deploy", help="Create an async MSA endpoint and wait for true readiness.")
    deploy.add_argument("--model-package-arn", default=DEFAULT_MODEL_PACKAGE_ARN)
    deploy.add_argument("--role-arn", help="SageMaker execution role; or set ANTHROFOLD_EXECUTION_ROLE_ARN.")
    deploy.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    deploy.add_argument("--name", help="Base endpoint name. A timestamped name is generated by default.")
    deploy.add_argument("--bucket", help="Async I/O bucket. Default: sagemaker-<region>-<account>.")
    deploy.add_argument("--output-prefix", default="anthrofold-msa/async-output")
    deploy.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    deploy.add_argument("--capacity-backoff-seconds", type=int, default=120)
    deploy.add_argument("--poll-seconds", type=int, default=60)
    deploy.add_argument("--ready-timeout-seconds", type=int, default=DEFAULT_READY_TIMEOUT)
    deploy.add_argument("--readiness", choices=("auto", "logs", "probe"), default="auto")
    deploy.add_argument("--no-wait", action="store_true", help="Return after CreateEndpoint; use wait later.")
    deploy.add_argument("--dry-run", action="store_true", help="Validate and print the plan without creating resources.")
    deploy.set_defaults(func=_deploy)

    status = sub.add_parser("status", help="Show allocation and application-readiness states.")
    status.add_argument("--endpoint-name")
    status.set_defaults(func=_status)

    wait = sub.add_parser("wait", help="Wait for database preparation and write the endpoint sidecar.")
    wait.add_argument("--endpoint-name")
    wait.add_argument("--bucket", help="Needed for probe fallback if CloudWatch Logs is unavailable.")
    wait.add_argument("--poll-seconds", type=int, default=60)
    wait.add_argument("--ready-timeout-seconds", type=int, default=DEFAULT_READY_TIMEOUT)
    wait.add_argument("--readiness", choices=("auto", "logs", "probe"), default="auto")
    wait.set_defaults(func=_wait)

    delete = sub.add_parser("delete", help="Delete the exact endpoint, config, and model.")
    delete.add_argument("--endpoint-name")
    delete.add_argument("--yes", action="store_true", help="Skip the interactive confirmation.")
    delete.add_argument("--keep-config", action="store_true")
    delete.add_argument("--keep-model", action="store_true")
    delete.set_defaults(func=_delete)
    return parser


def _default_command_argv(command, argv):
    """Insert a wrapper's subcommand without breaking top-level options."""
    argv = list(argv)
    global_with_value = {"--region", "--state-file", "--sidecar"}
    before, after = [], []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in global_with_value:
            if index + 1 >= len(argv):
                # Let argparse produce its standard missing-value diagnostic.
                before.append(token)
                index += 1
                continue
            before.extend((token, argv[index + 1]))
            index += 2
            continue
        if any(token.startswith(option + "=") for option in global_with_value):
            before.append(token)
        else:
            after.append(token)
        index += 1
    return [*before, command, *after]


def main(argv=None, default_command=None):
    if argv is None:
        argv = sys.argv[1:]
    if default_command is not None:
        argv = _default_command_argv(default_command, argv)
    args = build_parser().parse_args(argv)
    try:
        if getattr(args, "max_attempts", 1) < 1:
            raise ValueError("--max-attempts must be at least 1")
        if getattr(args, "poll_seconds", 1) < 1:
            raise ValueError("--poll-seconds must be at least 1")
        if getattr(args, "capacity_backoff_seconds", 0) < 0:
            raise ValueError("--capacity-backoff-seconds cannot be negative")
        if getattr(args, "ready_timeout_seconds", 1) < 1:
            raise ValueError("--ready-timeout-seconds must be at least 1")
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
