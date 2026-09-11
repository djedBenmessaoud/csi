#!/usr/bin/env python3
"""
Fireworks AI API key tester (DeepSeek).

Verifies a Fireworks AI API key across four use cases:
  1. Simple request        (single chat completion, non-streaming)
  2. Streaming request     (single chat completion, streaming)
  3. Multiple requests     (several sequential chat completions, non-streaming)
  4. Multiple requests     (several concurrent chat completions, streaming)

Usage:
    export FIREWORKS_API_KEY=fw_...
    python3 test_fireworks_key.py

Set FIREWORKS_MODEL to pin a specific model; otherwise the script picks a
DeepSeek model that is actually deployed for the account.

The key is read only from the environment (or an interactive prompt as a
fallback) - it is never written to disk or committed anywhere.
"""

import os
import sys
import json
import time
import getpass
import concurrent.futures

import requests

BASE_URL = "https://api.fireworks.ai/inference/v1"
TIMEOUT = 60

# Tried in order when the account's model listing gives us nothing usable.
# Non-reasoning V3 variants first: they answer directly, so the tests stay cheap.
FALLBACK_MODELS = [
    "accounts/fireworks/models/deepseek-v3p1",
    "accounts/fireworks/models/deepseek-v3-0324",
    "accounts/fireworks/models/deepseek-v3",
    "accounts/fireworks/models/deepseek-r1-0528",
    "accounts/fireworks/models/deepseek-r1",
]


def get_api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if key:
        return key.strip()
    if sys.stdin.isatty():
        return getpass.getpass("Enter FIREWORKS_API_KEY: ").strip()
    print("ERROR: FIREWORKS_API_KEY not set and no TTY to prompt.", file=sys.stderr)
    sys.exit(1)


def headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _rank(model_id: str) -> tuple:
    """Prefer non-reasoning V3 variants over R1 so replies are short and direct."""
    return (0 if "-v3" in model_id else 1, model_id)


def check_key_valid(api_key: str) -> tuple:
    """Cheap check: list models. Confirms the key authenticates and shows what it can reach."""
    print("\n=== 0. Key validity check (GET /models) ===")
    try:
        resp = requests.get(f"{BASE_URL}/models", headers=headers(api_key), timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"FAIL: request error: {e}")
        return False, []

    if resp.status_code == 401:
        print("FAIL: status 401 Unauthorized - key is invalid or revoked.")
        print(f"Body: {resp.text[:300]}")
        return False, []
    if resp.status_code != 200:
        print(f"FAIL: unexpected status {resp.status_code}")
        print(f"Body: {resp.text[:300]}")
        return False, []

    data = resp.json().get("data", [])
    ids = [m.get("id", "") for m in data]
    deepseek = sorted([m for m in ids if "deepseek" in m.lower()], key=_rank)
    print(f"PASS: status 200, key authenticates. {len(ids)} models listed.")
    if deepseek:
        print(f"DeepSeek models listed for this account ({len(deepseek)}):")
        for m in deepseek:
            print(f"  - {m}")
    else:
        print("No DeepSeek models in the listing; will probe known serverless IDs.")
    return True, deepseek


def resolve_model(api_key: str, listed_deepseek: list) -> str:
    """Pick a DeepSeek model that actually answers, probing candidates with a 1-token call."""
    print("\n=== Model selection ===")
    pinned = os.environ.get("FIREWORKS_MODEL")

    candidates = []
    for m in ([pinned] if pinned else []) + listed_deepseek + FALLBACK_MODELS:
        if m and m not in candidates:
            candidates.append(m)

    for model in candidates:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "temperature": 0,
        }
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions", headers=headers(api_key), json=payload, timeout=TIMEOUT
            )
        except requests.RequestException as e:
            print(f"  probe {model}: request error: {e}")
            continue

        if resp.status_code == 200:
            print(f"  probe {model}: OK -> using this model")
            return model
        try:
            msg = resp.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            msg = resp.text[:120]
        print(f"  probe {model}: status {resp.status_code} - {msg}")

    print("FAIL: no usable DeepSeek model found.")
    return ""


def _extract(message: dict) -> str:
    """Reasoning models may put the reply in reasoning_content when max_tokens is tight."""
    return message.get("content") or message.get("reasoning_content") or ""


def _consume_stream(resp) -> tuple:
    """Read an SSE chat-completion stream. Returns (chunk_count, text)."""
    chunks = 0
    text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data.strip() == "[DONE]":
            break
        obj = json.loads(data)
        delta = obj.get("choices", [{}])[0].get("delta", {})
        piece = _extract(delta)
        if piece:
            text += piece
            chunks += 1
    return chunks, text


def test_simple_request(api_key: str, model: str) -> bool:
    print("\n=== 1. Simple request (single, non-streaming) ===")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": False,
    }
    try:
        t0 = time.time()
        resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers(api_key), json=payload, timeout=TIMEOUT)
        elapsed = time.time() - t0
    except requests.RequestException as e:
        print(f"FAIL: request error: {e}")
        return False

    if resp.status_code != 200:
        print(f"FAIL: status {resp.status_code} - {resp.text[:300]}")
        return False

    data = resp.json()
    content = _extract(data["choices"][0]["message"])
    usage = data.get("usage", {})
    print(f"PASS: status 200 in {elapsed:.2f}s - reply: {content.strip()!r}")
    print(f"      tokens: {usage.get('prompt_tokens')} prompt / {usage.get('completion_tokens')} completion")
    return True


def test_streaming_request(api_key: str, model: str) -> bool:
    print("\n=== 2. Streaming request (single) ===")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5, separated by spaces."}],
        "max_tokens": 96,
        "temperature": 0,
        "stream": True,
    }
    try:
        t0 = time.time()
        resp = requests.post(
            f"{BASE_URL}/chat/completions", headers=headers(api_key), json=payload, timeout=TIMEOUT, stream=True
        )
    except requests.RequestException as e:
        print(f"FAIL: request error: {e}")
        return False

    if resp.status_code != 200:
        print(f"FAIL: status {resp.status_code} - {resp.text[:300]}")
        return False

    try:
        chunks, text = _consume_stream(resp)
    except (requests.RequestException, ValueError) as e:
        print(f"FAIL: stream read error: {e}")
        return False

    elapsed = time.time() - t0
    if chunks == 0:
        print("FAIL: no streamed chunks received.")
        return False
    print(f"PASS: status 200 in {elapsed:.2f}s - {chunks} chunks streamed")
    print(f"      text: {text.strip()!r}")
    return True


def test_multiple_non_streaming(api_key: str, model: str, n: int = 3) -> bool:
    print(f"\n=== 3. Multiple requests, non-streaming (n={n}, sequential) ===")
    all_ok = True
    for i in range(n):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"Reply with exactly the number {i + 1} and nothing else."}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
        }
        try:
            t0 = time.time()
            resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers(api_key), json=payload, timeout=TIMEOUT)
            elapsed = time.time() - t0
        except requests.RequestException as e:
            print(f"  [{i + 1}/{n}] FAIL: request error: {e}")
            all_ok = False
            continue

        if resp.status_code != 200:
            print(f"  [{i + 1}/{n}] FAIL: status {resp.status_code} - {resp.text[:200]}")
            all_ok = False
            continue

        content = _extract(resp.json()["choices"][0]["message"])
        print(f"  [{i + 1}/{n}] PASS: status 200 in {elapsed:.2f}s - reply: {content.strip()!r}")

    print("PASS: all sub-requests succeeded" if all_ok else "FAIL: one or more sub-requests failed")
    return all_ok


def _stream_one(api_key: str, model: str, index: int, n: int) -> bool:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Say the word 'request-{index + 1}' three times."}],
        "max_tokens": 96,
        "temperature": 0,
        "stream": True,
    }
    try:
        t0 = time.time()
        resp = requests.post(
            f"{BASE_URL}/chat/completions", headers=headers(api_key), json=payload, timeout=TIMEOUT, stream=True
        )
    except requests.RequestException as e:
        print(f"  [{index + 1}/{n}] FAIL: request error: {e}")
        return False

    if resp.status_code != 200:
        print(f"  [{index + 1}/{n}] FAIL: status {resp.status_code} - {resp.text[:200]}")
        return False

    try:
        chunks, text = _consume_stream(resp)
    except (requests.RequestException, ValueError) as e:
        print(f"  [{index + 1}/{n}] FAIL: stream read error: {e}")
        return False

    elapsed = time.time() - t0
    if chunks == 0:
        print(f"  [{index + 1}/{n}] FAIL: no streamed chunks received.")
        return False
    print(f"  [{index + 1}/{n}] PASS: status 200 in {elapsed:.2f}s - {chunks} chunks - {text.strip()[:60]!r}")
    return True


def test_multiple_streaming(api_key: str, model: str, n: int = 3) -> bool:
    print(f"\n=== 4. Multiple requests, streaming (n={n}, concurrent) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda i: _stream_one(api_key, model, i, n), range(n)))
    all_ok = all(results)
    print("PASS: all concurrent streams succeeded" if all_ok else "FAIL: one or more concurrent streams failed")
    return all_ok


def print_summary(results: dict, model: str = ""):
    print("\n" + "=" * 50)
    print("SUMMARY" + (f"  (model: {model})" if model else ""))
    print("=" * 50)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL':4}  {name}")
    print("=" * 50)


def main():
    api_key = get_api_key()

    results = {}
    valid, listed_deepseek = check_key_valid(api_key)
    results["key_valid"] = valid
    if not valid:
        print("\nKey failed basic validity check - skipping remaining tests.")
        print_summary(results)
        sys.exit(1)

    model = resolve_model(api_key, listed_deepseek)
    results["model_available"] = bool(model)
    if not model:
        print("\nNo usable DeepSeek model - skipping remaining tests.")
        print_summary(results)
        sys.exit(1)

    results["simple_request"] = test_simple_request(api_key, model)
    results["streaming_request"] = test_streaming_request(api_key, model)
    results["multiple_non_streaming"] = test_multiple_non_streaming(api_key, model)
    results["multiple_streaming"] = test_multiple_streaming(api_key, model)

    print_summary(results, model)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
