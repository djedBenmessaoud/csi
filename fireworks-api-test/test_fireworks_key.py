#!/usr/bin/env python3
"""
Fireworks AI API key tester.

Verifies a Fireworks AI API key across four use cases:
  1. Simple request        (single chat completion, non-streaming)
  2. Streaming request     (single chat completion, streaming)
  3. Multiple requests     (several sequential chat completions, non-streaming)
  4. Multiple requests     (several concurrent chat completions, streaming)

Usage:
    export FIREWORKS_API_KEY=fw_...
    python3 test_fireworks_key.py

The key is read only from the environment (or an interactive prompt as a
fallback) — it is never written to disk or committed anywhere.
"""

import os
import sys
import time
import getpass
import concurrent.futures

import requests

BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
TIMEOUT = 30


def get_api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if key:
        return key.strip()
    if sys.stdin.isatty():
        return getpass.getpass("Enter FIREWORKS_API_KEY: ").strip()
    print("ERROR: FIREWORKS_API_KEY not set and no TTY to prompt.", file=sys.stderr)
    sys.exit(1)


def headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def check_key_valid(api_key: str) -> bool:
    """Cheap check: list models. Confirms the key authenticates at all."""
    print("\n=== 0. Key validity check (GET /models) ===")
    try:
        resp = requests.get(f"{BASE_URL}/models", headers=headers(api_key), timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"FAIL: request error: {e}")
        return False

    if resp.status_code == 200:
        print(f"PASS: status {resp.status_code}, key authenticates.")
        return True
    elif resp.status_code == 401:
        print(f"FAIL: status 401 Unauthorized — key is invalid or revoked.")
        print(f"Body: {resp.text[:300]}")
        return False
    else:
        print(f"FAIL: unexpected status {resp.status_code}")
        print(f"Body: {resp.text[:300]}")
        return False


def test_simple_request(api_key: str) -> bool:
    print("\n=== 1. Simple request (single, non-streaming) ===")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
        "max_tokens": 10,
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
        print(f"FAIL: status {resp.status_code} — {resp.text[:300]}")
        return False

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    print(f"PASS: status 200 in {elapsed:.2f}s — reply: {content!r}")
    return True


def test_streaming_request(api_key: str) -> bool:
    print("\n=== 2. Streaming request (single) ===")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 5, one number per word."}],
        "max_tokens": 40,
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
        print(f"FAIL: status {resp.status_code} — {resp.text[:300]}")
        return False

    chunks = 0
    text = ""
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            import json
            obj = json.loads(data)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                text += piece
                chunks += 1
    except requests.RequestException as e:
        print(f"FAIL: stream read error: {e}")
        return False

    elapsed = time.time() - t0
    if chunks == 0:
        print("FAIL: no streamed chunks received.")
        return False
    print(f"PASS: status 200 in {elapsed:.2f}s — {chunks} chunks, text: {text!r}")
    return True


def test_multiple_non_streaming(api_key: str, n: int = 3) -> bool:
    print(f"\n=== 3. Multiple requests, non-streaming (n={n}, sequential) ===")
    all_ok = True
    for i in range(n):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": f"Reply with exactly the number {i + 1} and nothing else."}],
            "max_tokens": 10,
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
            print(f"  [{i + 1}/{n}] FAIL: status {resp.status_code} — {resp.text[:200]}")
            all_ok = False
            continue

        content = resp.json()["choices"][0]["message"]["content"]
        print(f"  [{i + 1}/{n}] PASS: status 200 in {elapsed:.2f}s — reply: {content!r}")

    print("PASS: all sub-requests succeeded" if all_ok else "FAIL: one or more sub-requests failed")
    return all_ok


def _stream_one(api_key: str, index: int, n: int) -> bool:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Say the word 'request-{index}' three times."}],
        "max_tokens": 30,
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
        print(f"  [{index + 1}/{n}] FAIL: status {resp.status_code} — {resp.text[:200]}")
        return False

    chunks = 0
    text = ""
    import json
    try:
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
            piece = delta.get("content", "")
            if piece:
                text += piece
                chunks += 1
    except requests.RequestException as e:
        print(f"  [{index + 1}/{n}] FAIL: stream read error: {e}")
        return False

    elapsed = time.time() - t0
    if chunks == 0:
        print(f"  [{index + 1}/{n}] FAIL: no streamed chunks received.")
        return False
    print(f"  [{index + 1}/{n}] PASS: status 200 in {elapsed:.2f}s — {chunks} chunks, text: {text!r}")
    return True


def test_multiple_streaming(api_key: str, n: int = 3) -> bool:
    print(f"\n=== 4. Multiple requests, streaming (n={n}, concurrent) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda i: _stream_one(api_key, i, n), range(n)))
    all_ok = all(results)
    print("PASS: all concurrent streams succeeded" if all_ok else "FAIL: one or more concurrent streams failed")
    return all_ok


def main():
    api_key = get_api_key()

    results = {}
    results["key_valid"] = check_key_valid(api_key)
    if not results["key_valid"]:
        print("\nKey failed basic validity check — skipping remaining tests.")
        print_summary(results)
        sys.exit(1)

    results["simple_request"] = test_simple_request(api_key)
    results["streaming_request"] = test_streaming_request(api_key)
    results["multiple_non_streaming"] = test_multiple_non_streaming(api_key)
    results["multiple_streaming"] = test_multiple_streaming(api_key)

    print_summary(results)
    sys.exit(0 if all(results.values()) else 1)


def print_summary(results: dict):
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL':4}  {name}")
    print("=" * 50)


if __name__ == "__main__":
    main()
