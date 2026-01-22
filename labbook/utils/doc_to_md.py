#!/usr/bin/env python3
import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import error, request
from urllib.parse import urlparse
from dotenv import load_dotenv, find_dotenv

API_BASE_DEFAULT = "https://mineru.net/api/v4"
SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".html"}
TERMINAL_STATES = {"done", "failed"}
POLLING_STATES = {"waiting-file", "pending", "running", "converting"}
UPLOAD_RETRIES = 2
UPLOAD_RETRY_DELAY = 3
DOWNLOAD_RETRIES = 2
DOWNLOAD_RETRY_DELAY = 3
POLL_RETRIES = 2
POLL_RETRY_DELAY = 3


def chunked(items: List[Path], size: int) -> Iterable[List[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def sanitize_data_id(text: str, max_len: int = 128) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    trimmed = safe[: max_len - len(digest) - 1]
    return f"{trimmed}-{digest}"


def stable_data_id(rel_path: Path) -> str:
    rel_text = rel_path.as_posix()
    digest = hashlib.sha256(rel_text.encode("utf-8")).hexdigest()[:12]
    return sanitize_data_id(f"{rel_text}-{digest}")


def http_json(
    method: str, url: str, token: str, payload: Optional[dict] = None, timeout: int = 60
) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {method} {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed for {method} {url}: {exc}") from exc


def upload_file(url: str, file_path: Path, timeout: int = 120) -> None:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"Invalid upload URL: {url}")

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout)
    size = file_path.stat().st_size

    try:
        conn.putrequest("PUT", path)
        conn.putheader("Content-Length", str(size))
        conn.endheaders()

        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                conn.send(chunk)

        resp = conn.getresponse()
        _ = resp.read()
        if resp.status not in {200, 201, 204}:
            raise RuntimeError(f"Upload failed with status {resp.status} for {file_path}")
    finally:
        conn.close()


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    req = request.Request(url, method="GET")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} downloading {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc


def run_with_retries(action, retries: int, delay: int, label: str):
    attempt = 0
    while True:
        try:
            return action()
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                raise
            print(f"{label} failed: {exc}. Retrying in {delay}s ({attempt}/{retries})...")
            time.sleep(delay)


def find_markdown_file(root: Path, expected_stem: str) -> Optional[Path]:
    md_files = sorted(root.rglob("*.md"))
    if not md_files:
        return None
    for md_file in md_files:
        if md_file.stem == expected_stem or md_file.stem.lower() == expected_stem.lower():
            return md_file
    for md_file in md_files:
        if md_file.name.lower() in {"markdown.md", "main.md", "output.md"}:
            return md_file
    return md_files[0]


def find_json_file(root: Path, expected_stem: str) -> Optional[Path]:
    # First try to find _content_list.json specifically
    content_list_files = sorted(root.rglob("*_content_list.json"))
    if content_list_files:
        return content_list_files[0]

    # Fallback to generic search
    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        return None
    for json_file in json_files:
        if json_file.stem == expected_stem or json_file.stem.lower() == expected_stem.lower():
            return json_file
    for json_file in json_files:
        if "content" in json_file.name.lower() or "result" in json_file.name.lower():
            return json_file
    return json_files[0]


def collect_files(input_dir: Path, output_dir: Path) -> List[Path]:
    files = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
    return sorted(files)


def request_upload_urls(
    api_base: str,
    token: str,
    files: List[Path],
    data_ids: Dict[Path, str],
    model_version: str,
) -> Tuple[str, List[str]]:
    payload = {
        "files": [{"name": path.name, "data_id": data_ids[path]} for path in files],
        "model_version": model_version,
    }
    url = f"{api_base}/file-urls/batch"
    response = http_json("POST", url, token, payload=payload)
    if response.get("code") != 0:
        raise RuntimeError(f"Upload URL request failed: {response}")
    data = response.get("data") or {}
    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls") or data.get("files")
    if not batch_id or not file_urls:
        raise RuntimeError(f"Missing batch_id or file URLs in response: {response}")
    if len(file_urls) != len(files):
        raise RuntimeError(
            f"Upload URL count mismatch: expected {len(files)} got {len(file_urls)}"
        )
    return batch_id, file_urls


def poll_batch_results(
    api_base: str,
    token: str,
    batch_id: str,
    expected_count: int,
    poll_interval: int,
    timeout_seconds: int,
) -> List[dict]:
    start = time.monotonic()
    url = f"{api_base}/extract-results/batch/{batch_id}"
    while True:
        response = run_with_retries(
            lambda: http_json("GET", url, token),
            retries=POLL_RETRIES,
            delay=POLL_RETRY_DELAY,
            label=f"Polling batch {batch_id}",
        )
        if response.get("code") != 0:
            raise RuntimeError(f"Batch result request failed: {response}")
        data = response.get("data") or {}
        results = data.get("extract_result") or []
        if len(results) >= expected_count and all(
            item.get("state") in TERMINAL_STATES for item in results
        ):
            return results

        elapsed = time.monotonic() - start
        if timeout_seconds and elapsed > timeout_seconds:
            raise RuntimeError(f"Timed out waiting for batch {batch_id} results")
        time.sleep(poll_interval)


def save_markdown_from_zip(
    zip_url: str, output_path: Path, expected_stem: str, save_json: bool = False, save_zip: bool = False
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        zip_path = temp_dir_path / "result.zip"
        run_with_retries(
            lambda: download_file(zip_url, zip_path),
            retries=DOWNLOAD_RETRIES,
            delay=DOWNLOAD_RETRY_DELAY,
            label=f"Download {expected_stem}",
        )

        # Save zip file if requested
        if save_zip:
            zip_output_path = output_path.with_suffix(".zip")
            zip_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(zip_path, zip_output_path)
            print(f"Wrote ZIP {zip_output_path}")

        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall(temp_dir_path / "extract")

        md_file = find_markdown_file(temp_dir_path / "extract", expected_stem)
        if not md_file:
            raise RuntimeError(f"No markdown file found in result for {expected_stem}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(md_file, output_path)

        if save_json:
            json_file = find_json_file(temp_dir_path / "extract", expected_stem)
            if json_file:
                json_output_path = output_path.with_suffix(".json")
                shutil.copyfile(json_file, json_output_path)
                print(f"Wrote JSON {json_output_path}")
            else:
                print(f"Warning: No JSON file found in result for {expected_stem}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch convert documents to Markdown using MinerU API."
    )
    parser.add_argument("input_dir", help="Directory containing source documents.")
    parser.add_argument("output_dir", help="Directory to write Markdown results.")
    parser.add_argument(
        "--token",
        help="MinerU token; defaults to MINERU_TOKEN environment variable.",
    )
    parser.add_argument(
        "--api-base",
        default=API_BASE_DEFAULT,
        help=f"MinerU API base URL. Default: {API_BASE_DEFAULT}",
    )
    parser.add_argument(
        "--model-version",
        default="pipeline",
        choices=["pipeline", "vlm"],
        help="MinerU model version; default is pipeline.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of files per batch (max 200).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown outputs. Default is to skip.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save JSON output in addition to Markdown. JSON will be saved in same path with .json extension.",
    )
    parser.add_argument(
        "--save-zip",
        action="store_true",
        help="Save original ZIP file from MinerU API. ZIP will be saved in same path with .zip extension.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Polling interval in seconds for batch results.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=1800,
        help="Max seconds to wait for a batch before timing out.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    token = args.token or os.getenv("MINERU_TOKEN")
    if not token:
        print("Missing MinerU token. Provide --token or set MINERU_TOKEN.", file=sys.stderr)
        return 2

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    files = collect_files(input_dir, output_dir)
    if not files:
        print("No supported documents found.", file=sys.stderr)
        return 1

    if args.batch_size > 200 or args.batch_size <= 0:
        print("Batch size must be between 1 and 200.", file=sys.stderr)
        return 2

    if not args.overwrite:
        skipped = []
        remaining = []
        for path in files:
            rel_path = path.relative_to(input_dir)
            output_path = (output_dir / rel_path).with_suffix(".md")
            md_exists = output_path.exists()
            json_exists = output_path.with_suffix(".json").exists()
            zip_exists = output_path.with_suffix(".zip").exists()

            # Check all requested outputs
            outputs_exist = md_exists
            if args.save_json:
                outputs_exist = outputs_exist and json_exists
            if args.save_zip:
                outputs_exist = outputs_exist and zip_exists

            if outputs_exist:
                skipped.append(path)
            else:
                remaining.append(path)

        if skipped:
            suffix = "outputs" if (args.save_json or args.save_zip) else "markdown outputs"
            print(f"Skipping {len(skipped)} files with existing {suffix}.")
        files = remaining

    if not files:
        print("No documents to process after applying skip/overwrite rules.")
        return 0

    data_ids = {path: stable_data_id(path.relative_to(input_dir)) for path in files}
    data_id_to_path = {data_ids[path]: path for path in files}
    file_name_to_paths: Dict[str, List[Path]] = {}
    for path in files:
        file_name_to_paths.setdefault(path.name, []).append(path)

    print(f"Found {len(files)} documents. Uploading in batches of {args.batch_size}...")

    for batch in chunked(files, args.batch_size):
        print(f"\nRequesting upload URLs for {len(batch)} files...")
        try:
            batch_id, upload_urls = request_upload_urls(
                args.api_base, token, batch, data_ids, args.model_version
            )
        except Exception as exc:
            print(f"Batch upload URL request failed, skipping batch: {exc}")
            continue

        successful_uploads = []
        for file_path, upload_url in zip(batch, upload_urls):
            print(f"Uploading {file_path}...")
            try:
                run_with_retries(
                    lambda: upload_file(upload_url, file_path),
                    retries=UPLOAD_RETRIES,
                    delay=UPLOAD_RETRY_DELAY,
                    label=f"Upload {file_path}",
                )
                successful_uploads.append(file_path)
            except Exception as exc:
                print(f"Upload failed for {file_path}: {exc}")

        if not successful_uploads:
            print("No files uploaded successfully in this batch, skipping results.")
            continue

        print(f"Polling results for batch {batch_id}...")
        try:
            results = poll_batch_results(
                args.api_base,
                token,
                batch_id,
                expected_count=len(successful_uploads),
                poll_interval=args.poll_interval,
                timeout_seconds=args.poll_timeout,
            )
        except Exception as exc:
            print(f"Batch result polling failed, skipping batch: {exc}")
            continue

        for result in results:
            state = result.get("state")
            file_name = result.get("file_name")
            data_id = result.get("data_id")
            zip_url = result.get("full_zip_url")

            source_path = None
            if data_id and data_id in data_id_to_path:
                source_path = data_id_to_path[data_id]
            elif file_name and len(file_name_to_paths.get(file_name, [])) == 1:
                source_path = file_name_to_paths[file_name][0]

            if not source_path:
                print(f"Skipping result (unmatched source): {result}")
                continue

            rel_path = source_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            output_path = output_path.with_suffix(".md")

            if state != "done":
                print(f"Failed to convert {source_path} (state={state}).")
                continue
            if not zip_url:
                print(f"Missing result zip URL for {source_path}.")
                continue

            print(f"Downloading markdown for {source_path}...")
            try:
                save_markdown_from_zip(zip_url, output_path, source_path.stem, args.save_json, args.save_zip)
            except Exception as exc:
                print(f"Download failed for {source_path}: {exc}")
                continue
            print(f"Wrote {output_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
