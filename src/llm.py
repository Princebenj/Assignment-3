"""
llm.py
======
Thin, defensive wrapper around a local Ollama server.

Three things this module is responsible for, beyond just calling the API:

1. **Getting JSON out of a language model reliably.**  Even with structured
   decoding enabled, models occasionally wrap output in markdown fences or
   prepend a sentence.  `extract_json` handles the realistic failure modes.
2. **Validating what came back.**  A response that parses is not necessarily a
   response that conforms.  `validate_record` checks required keys and
   enumerated values, and every call records whether validation passed - which
   is what turns "the prompt works well" into a number in the report.
3. **Archiving every exchange.**  Prompt, model, options, raw response and
   parse outcome are written to outputs/llm_logs/ as JSONL.  Without this the
   report's LLM sections are unreproducible.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from . import config


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """One call to one model, with everything needed to audit it later."""

    model: str
    prompt_name: str
    raw_text: str
    parsed: dict | None
    valid: bool
    validation_errors: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    run_index: int = 0
    image_id: str | None = None
    error: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Image encoding
# --------------------------------------------------------------------------
def save_array_as_png(arr: np.ndarray, path: str | Path) -> str:
    """Save an image array as PNG and return the path.

    This mirrors `save_array_as_png` from Lab 2/3: the Ollama Python client
    accepts images as file paths, which is the convention the module teaches,
    and it also leaves the exact bytes the model saw on disk - useful evidence
    when a description has to be re-checked later.
    """
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)     # Ollama is happiest with 3-channel PNGs
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)
    return str(path)


def array_to_base64_png(arr: np.ndarray) -> str:
    """Encode a numpy image as base64 PNG for the Ollama `images` field.

    Accepts float arrays in [0, 1] or uint8 arrays, grayscale or RGB.  We send
    the *preprocessed* grayscale image rather than the original RGB, so that
    the VLM sees exactly what the rest of the pipeline sees - otherwise a
    comparison between the VLM's description and the measured features would
    be comparing two different inputs.
    """
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 1)
        a = (a * 255).astype(np.uint8)
    img = Image.fromarray(a)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------------------------------------------------------------
# JSON handling
# --------------------------------------------------------------------------
def extract_json(text: str) -> dict | None:
    """Best-effort recovery of a JSON object from model output.

    Strategy, cheapest first:
      1. the whole string parses;
      2. strip ``` fences and retry;
      3. take the first balanced {...} span and parse that;
      4. give up and return None (the caller records a parse failure).

    We deliberately do NOT attempt aggressive repairs such as quoting bare keys
    or trimming trailing commas.  A record that needed heavy repair is not
    trustworthy, and silently fixing it would hide exactly the unreliability
    the report is supposed to measure.
    """
    if not text:
        return None

    candidates: list[str] = [text.strip()]

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    candidates.extend(f.strip() for f in fenced)

    span = _first_balanced_object(text)
    if span:
        candidates.append(span)

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return obj[0]
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _first_balanced_object(text: str) -> str | None:
    """Return the first brace-balanced {...} substring, ignoring braces in strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def validate_record(record: dict | None, schema: dict) -> tuple[bool, list[str]]:
    """Lightweight schema check: required keys, types, and enum membership.

    A deliberately small dependency-free validator - `jsonschema` is not
    guaranteed present in a fresh Colab runtime, and the schemas here are flat.
    """
    errors: list[str] = []
    if record is None:
        return False, ["response did not contain parseable JSON"]

    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in record:
            errors.append(f"missing required key: {key}")

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "array": list,
        "object": dict,
        "boolean": bool,
    }

    for key, spec in props.items():
        if key not in record:
            continue
        value = record[key]
        expected = spec.get("type")
        if expected in type_map and not isinstance(value, type_map[expected]):
            # Accept an integer-valued float for "integer" - models often emit 12.0
            if not (expected == "integer" and isinstance(value, float)
                    and float(value).is_integer()):
                errors.append(
                    f"{key}: expected {expected}, got {type(value).__name__}"
                )
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key}: {value!r} not in allowed values {spec['enum']}")

    return (len(errors) == 0), errors


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class OllamaClient:
    """Minimal client for the local Ollama HTTP API.

    Usage
    -----
    >>> client = OllamaClient()
    >>> client.available()
    True
    >>> resp = client.generate("llama3.1", "Say hi")
    """

    def __init__(self, host: str = None, timeout: int = None,
                 log_dir: Path = None):
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.timeout = timeout or config.LLM_TIMEOUT_S
        self.log_dir = Path(log_dir or config.LLM_LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # -- infrastructure ---------------------------------------------------
    def available(self) -> bool:
        """True if an Ollama server is reachable."""
        try:
            import requests

            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        import requests

        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def require(self, model: str) -> None:
        """Raise a helpful error if the server or model is missing."""
        if not self.available():
            raise RuntimeError(
                f"No Ollama server at {self.host}.\n"
                "Start one with `ollama serve`, or in Colab run the bootstrap "
                "cell in notebooks/00_setup.ipynb."
            )
        models = self.list_models()
        # Ollama reports tags like 'llama3.1:latest'; match on the base name.
        if not any(m.split(":")[0] == model.split(":")[0] for m in models):
            raise RuntimeError(
                f"Model {model!r} not pulled. Available: {models}\n"
                f"Run: ollama pull {model}"
            )

    # -- core call --------------------------------------------------------
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        images: Sequence[str] | None = None,
        image_paths: Sequence[str] | None = None,
        fmt: Any = None,
        temperature: float = None,
        seed: int | None = None,
        num_predict: int = 700,
    ) -> tuple[str, float]:
        """Single completion. Returns (text, latency_seconds).

        `fmt` may be the string "json" or a JSON-schema dict; both are passed
        straight to Ollama's `format` field, which constrains decoding so the
        output is guaranteed parseable.  We still run our own validator on top,
        because constrained decoding guarantees syntax, not semantics.

        Transport: the `ollama` Python client (`from ollama import chat`) is
        used when installed, which is the interface used throughout Labs 2-5.
        We fall back to the raw HTTP endpoint if the package is missing, so the
        code still runs in a bare environment.
        """
        try:
            return self._generate_via_client(
                model, prompt, system=system, image_paths=image_paths,
                fmt=fmt, temperature=temperature, seed=seed,
                num_predict=num_predict,
            )
        except ImportError:
            pass          # ollama package not installed -> use plain HTTP

        import requests

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": (
                    config.LLM_TEMPERATURE if temperature is None else temperature
                ),
                "num_predict": num_predict,
            },
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = list(images)
        if fmt is not None:
            payload["format"] = fmt
        if seed is not None:
            payload["options"]["seed"] = seed

        t0 = time.time()
        r = requests.post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        latency = time.time() - t0
        return r.json().get("response", ""), latency

    def _generate_via_client(
        self, model: str, prompt: str, *, system: str | None,
        image_paths: Sequence[str] | None, fmt: Any,
        temperature: float | None, seed: int | None, num_predict: int,
    ) -> tuple[str, float]:
        """Transport using the `ollama` Python package, as taught in the labs.

        Raises ImportError if the package is absent, which `generate` catches
        and handles by falling back to HTTP.
        """
        from ollama import chat        # ImportError -> caller falls back

        options: dict[str, Any] = {
            "temperature": (
                config.LLM_TEMPERATURE if temperature is None else temperature
            ),
            "num_predict": num_predict,
        }
        if seed is not None:
            options["seed"] = seed

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user_msg: dict[str, Any] = {"role": "user", "content": prompt}
        if image_paths:
            user_msg["images"] = list(image_paths)
        messages.append(user_msg)

        kwargs: dict[str, Any] = dict(model=model, messages=messages, options=options)
        if fmt is not None:
            kwargs["format"] = fmt

        t0 = time.time()
        response = chat(**kwargs)
        latency = time.time() - t0
        return response["message"]["content"], latency

    # -- structured call with validation and logging ----------------------
    def structured(
        self,
        model: str,
        prompt: str,
        *,
        prompt_name: str,
        schema: dict | None = None,
        system: str | None = None,
        image: np.ndarray | None = None,
        image_id: str | None = None,
        use_format: bool = True,
        run_index: int = 0,
        temperature: float = None,
        seed: int | None = None,
    ) -> LLMResponse:
        """Call the model, parse, validate, log. Never raises on model error.

        Returns an LLMResponse whose `.valid` flag drives the reliability
        statistics reported in Task 1.
        """
        # Save the exact image sent to the model, then pass its path (lab
        # convention). The base64 form is also computed for the HTTP fallback.
        image_paths = images_b64 = None
        if image is not None:
            img_dir = self.log_dir / "sent_images"
            path = save_array_as_png(
                image, img_dir / f"{prompt_name}_{image_id or 'img'}_{run_index}.png"
            )
            image_paths = [path]
            images_b64 = [array_to_base64_png(image)]
        fmt = None
        if use_format and schema is not None:
            fmt = schema          # Ollama >= 0.5 accepts a JSON schema here
        elif use_format:
            fmt = "json"

        try:
            raw, latency = self.generate(
                model, prompt, system=system, images=images_b64,
                image_paths=image_paths, fmt=fmt,
                temperature=temperature, seed=seed,
            )
            error = None
        except Exception as exc:                      # network, OOM, timeout...
            raw, latency, error = "", 0.0, f"{type(exc).__name__}: {exc}"

        parsed = extract_json(raw) if raw else None
        if schema is not None:
            valid, errs = validate_record(parsed, schema)
        else:
            valid, errs = (parsed is not None), []

        resp = LLMResponse(
            model=model, prompt_name=prompt_name, raw_text=raw, parsed=parsed,
            valid=valid, validation_errors=errs, latency_s=round(latency, 2),
            run_index=run_index, image_id=image_id, error=error,
        )
        self._log(resp)
        return resp

    def repeat(self, n: int = None, **kwargs) -> list[LLMResponse]:
        """Run the same call n times - the run-to-run variability experiment.

        No seed is passed, so we observe the behaviour a user would actually
        get from default sampling.
        """
        n = n or config.N_REPEAT_RUNS
        return [self.structured(run_index=i, **kwargs) for i in range(n)]

    # -- logging ----------------------------------------------------------
    def _log(self, resp: LLMResponse) -> None:
        path = self.log_dir / f"{resp.prompt_name}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(resp.to_json(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Analysis helpers for the report
# --------------------------------------------------------------------------
def response_agreement(responses: Sequence[LLMResponse], keys: Sequence[str]) -> dict:
    """Quantify run-to-run stability across repeated calls.

    For each key: how many distinct values appeared across runs, and whether
    every run agreed.  This is the evidence for "repeated runs are not
    identical" that the brief asks for - stated as a measurement rather than
    an anecdote.
    """
    out: dict[str, Any] = {"n_runs": len(responses)}
    for key in keys:
        values = []
        for r in responses:
            if r.parsed and key in r.parsed:
                v = r.parsed[key]
                values.append(json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else str(v))
        uniq = sorted(set(values))
        out[f"{key}__n_unique"] = len(uniq)
        out[f"{key}__identical"] = (len(uniq) <= 1)
        out[f"{key}__values"] = uniq
    texts = [r.raw_text.strip() for r in responses if r.raw_text]
    out["raw_text_identical"] = (len(set(texts)) <= 1) if texts else None
    return out


def validity_rate(responses: Sequence[LLMResponse]) -> float:
    """Fraction of responses that parsed AND conformed to the schema."""
    if not responses:
        return float("nan")
    return sum(r.valid for r in responses) / len(responses)
