"""Diagnose why DeepSeek structured JSON generation fails in Kosmos.

Runs two small checks against the configured provider and prints the raw
result/exception so we can see whether the API call itself fails or the JSON
parsing fails.

Usage:
    .venv/bin/python scripts/debug_deepseek_structured.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from kosmos.core.llm import get_client  # noqa: E402


def main():
    client = get_client(reset=True)
    print("provider:", type(client).__name__)
    print("model   :", getattr(client, "model", None))
    print("base_url:", getattr(client, "base_url", None))

    # Check 1: plain generation still works?
    plain = client.generate("Reply with exactly one word: OK", max_tokens=16, temperature=0)
    print("\n[1] plain generate ->", str(getattr(plain, "content", plain))[:80])

    # Check 2: the exact structured call used by HypothesisGeneratorAgent.
    schema = {
        "hypotheses": [
            {
                "statement": "string (clear, testable hypothesis)",
                "rationale": "string (scientific justification)",
                "confidence_score": "float 0.0-1.0",
                "testability_score": "float 0.0-1.0 (preliminary estimate)",
                "suggested_experiment_types": [
                    "computational | data_analysis | literature_synthesis"
                ],
            }
        ]
    }
    prompt = (
        "Research Question: In the provided gold RNA dataset, does adding "
        "pseudo-labeled non-gold external samples improve binary "
        "classification of BMMC cell state?\n\n"
        "Domain: biology\n\n"
        "Generate exactly 1 testable hypothesis."
    )
    print("\n[2] calling generate_structured ...")
    try:
        result = client.generate_structured(
            prompt=prompt,
            schema=schema,
            max_tokens=1000,
            temperature=0.7,
        )
        print("[2] structured ok ->", json.dumps(result)[:500])
    except Exception as exc:
        print("[2] structured FAILED:", type(exc).__name__)
        print("    message:", str(exc)[:1000])
        # Fallback: call plain generate with the same JSON-only system prompt
        # so we can see what DeepSeek actually returns.
        system = (
            "\n\nYou must respond with valid JSON matching this schema:\n"
            + json.dumps(schema, indent=2)
            + "\n\nIMPORTANT: Return ONLY valid JSON, no additional text or explanations."
        )
        print("\n[3] raw generate with same system prompt ...")
        try:
            raw = client.generate(
                prompt=prompt,
                system=system,
                max_tokens=1000,
                temperature=0.7,
            )
            text = getattr(raw, "content", str(raw))
            print("[3] raw response first 2000 chars:\n", text[:2000])
            usage = getattr(raw, "usage", None)
            if usage is not None:
                print("\n[3] usage:", usage)
        except Exception as raw_exc:
            print("[3] raw generate also FAILED:", type(raw_exc).__name__)
            print("    message:", str(raw_exc)[:1000])


if __name__ == "__main__":
    main()
