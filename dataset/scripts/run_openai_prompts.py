#!/usr/bin/env python3
"""Run BOUND-Handoff prompts with the OpenAI API.

Standard-library only; no OpenAI SDK required.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


URL = "[URL]"


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def done_ids(path):
    if not Path(path).exists():
        return set()
    out = set()
    for row in read_jsonl(path):
        out.add(row.get("prompt_id"))
    return out


def extract_text(resp):
    parts = []
    for item in resp.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text"):
                parts.append(content.get("text", ""))
    if parts:
        return "\n".join(parts).strip()
    return resp.get("output_text", "")


def call_openai(prompt, model, max_tokens, timeout, retries):
    key = os.environ.get("MODEL_KEY")
    if not key:
        raise RuntimeError("MODEL_KEY is not set on this server.")

    body = json.dumps(
        {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": "minimal"},
            "text": {"verbosity": "low"},
        }
    ).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = "HTTP {}: {}".format(exc.code, exc.read().decode("utf-8", errors="replace"))
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except Exception as exc:
            last_error = repr(exc)

        if attempt < retries:
            time.sleep(2 ** attempt)

    raise RuntimeError(last_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="data/bound_handoff_prompts.jsonl")
    parser.add_argument("--output", default="data/model_outputs_openai_gpt5mini.jsonl")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=1500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and Path(args.output).exists():
        Path(args.output).unlink()

    seen = set() if args.overwrite else done_ids(args.output)
    prompts = [p for p in read_jsonl(args.prompts) if p["prompt_id"] not in seen]
    if args.limit:
        prompts = prompts[: args.limit]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print("Running {} prompts with {}".format(len(prompts), args.model))
    with open(args.output, "a") as out:
        for i, prompt_record in enumerate(prompts, 1):
            print("[{}/{}] {}".format(i, len(prompts), prompt_record["prompt_id"]), flush=True)
            response = call_openai(
                prompt_record["prompt"],
                args.model,
                args.max_output_tokens,
                args.timeout,
                args.retries,
            )
            row = {
                key: prompt_record[key]
                for key in ["prompt_id", "scenario_id", "domain", "handoff_surface", "topology", "condition"]
            }
            row.update(
                {
                    "model": args.model,
                    "output": extract_text(response),
                    "response_id": response.get("id"),
                    "status": response.get("status"),
                    "incomplete_details": response.get("incomplete_details"),
                    "created_at": int(time.time()),
                }
            )
            out.write(json.dumps(row) + "\n")
            out.flush()
    print("Wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

