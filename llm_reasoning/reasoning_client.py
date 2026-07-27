
import json
import os
import time

from dotenv import load_dotenv
load_dotenv()  # reads .env in the project root automatically

from llm_reasoning.prompt_template import SYSTEM_PROMPT, build_prompt

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL_FALLBACKS = [
    os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types
    from google.genai.errors import ServerError, ClientError

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    backoff_seconds = [10, 30, 60]
    last_error = None

    for model_name in GEMINI_MODEL_FALLBACKS:
        try:
            for wait in backoff_seconds:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            response_mime_type="application/json",
                        ),
                    )
                    return response.text
                except ServerError as e:
                    # transient overload -- worth waiting and retrying the same model
                    last_error = e
                    print(f"  {model_name} overloaded (503), retrying in {wait}s...")
                    time.sleep(wait)
        except ClientError as e:
            # 429 = daily/rate quota exhausted -- retrying the SAME model won't help,
            # it resets on a ~24h clock, not in seconds. Skip straight to the next model.
            last_error = e
            print(f"  {model_name} quota exhausted (429), moving to next model...")
            continue

        print(f"  Giving up on {model_name}, trying next fallback model...")

    raise last_error


def get_rationale(claim: dict, max_retries: int = 2) -> dict:
    prompt = build_prompt(claim)

    for attempt in range(max_retries + 1):
        try:
            if PROVIDER == "anthropic":
                raw = _call_anthropic(prompt)
            elif PROVIDER == "openai":
                raw = _call_openai(prompt)
            else:
                raw = _call_gemini(prompt)

            cleaned = _strip_code_fences(raw)
            parsed = json.loads(cleaned)

            required = {"discrepancy_detected", "flagged_reasons", "recommended_action", "rationale_text"}
            if not required.issubset(parsed.keys()):
                raise ValueError(f"Missing keys in LLM response: {parsed.keys()}")
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            if attempt == max_retries:
                # Fail safe: return a neutral fallback rather than crashing the pipeline
                return {
                    "discrepancy_detected": False,
                    "flagged_reasons": ["LLM response could not be parsed"],
                    "recommended_action": "REVIEW",
                    "rationale_text": f"Automated reasoning failed ({e}); needs manual review.",
                }