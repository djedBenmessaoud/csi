# Fireworks AI API key test

Checks whether a Fireworks AI API key works, across four use cases:

1. Simple request — single chat completion, non-streaming
2. Streaming request — single chat completion, streaming
3. Multiple requests — several sequential chat completions, non-streaming
4. Multiple requests — several concurrent chat completions, streaming

A cheap `GET /models` call runs first to confirm the key authenticates at
all before spending tokens on the rest.

## Usage

```bash
export FIREWORKS_API_KEY=fw_...
python3 test_fireworks_key.py
```

If `FIREWORKS_API_KEY` isn't set and you're in an interactive terminal,
the script will prompt for it instead (input hidden, not echoed).

The key is only ever read from the environment or the prompt — it is
never written to disk or committed to the repo. Rotate/revoke the key
after testing as needed.
