from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from summarize_pdfs.config import AppConfig, Settings
from summarize_pdfs.pipeline.prompts import SYSTEM_JSON


@dataclass
class ArkaLLMClient:
    task: str = "pdf"
    skill: str = "summarize_pdfs"


LLMClient = AsyncOpenAI | ArkaLLMClient


def arka_llm_available() -> bool:
    try:
        from arka.llm.fallback import provider_available, provider_specs

        return any(provider_available(spec.slug) for spec in provider_specs())
    except ImportError:
        pass
    try:
        result = subprocess.run(
            [sys.executable, "-m", "arka.llm.cli", "active-model"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def llm_is_configured(config: AppConfig, settings: Settings) -> bool:
    provider = config.llm.provider.lower()
    if provider == "openai":
        return bool(
            settings.openai_api_key
            or config.llm.base_url
            or settings.openai_base_url
        )
    if provider == "arka":
        return arka_llm_available()
    return False


def make_llm_client(config: AppConfig, settings: Settings) -> LLMClient:
    provider = config.llm.provider.lower()
    if provider == "arka":
        return ArkaLLMClient(task=config.llm.task, skill=config.llm.skill)

    kwargs: dict[str, Any] = {}
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    base_url = config.llm.base_url or settings.openai_base_url
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def _cache_key(prompt: str, model: str, *, provider: str = "openai", system: str = "") -> str:
    raw = f"{provider}:{model}:{system}:{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_cache(cache_dir: Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_cache(cache_dir: Path, key: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps(payload, indent=2))


def _parse_json_text(content: str) -> dict:
    text = content.strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # Some models wrap JSON in prose; grab the outermost object if present.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            return json.loads(repair_json(text))
        except Exception:
            return {}


def _arka_complete_sync(client: ArkaLLMClient, system: str, user: str, temperature: float) -> str:
    try:
        from arka.llm.fallback import llm_complete

        text = llm_complete(system, user, temperature, task=client.task, skill=client.skill)
    except ImportError:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "arka.llm.cli",
                "complete",
                "-s",
                system,
                "-u",
                user,
                "-t",
                str(temperature),
                "--task",
                client.task,
                "--skill",
                client.skill,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Arka LLM completion failed")
        text = result.stdout
    if text.startswith("[LLM error:"):
        raise RuntimeError(text)
    return text


async def _complete_text(
    client: LLMClient,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
) -> str:
    if isinstance(client, ArkaLLMClient):
        return await asyncio.to_thread(
            _arka_complete_sync,
            client,
            system,
            user,
            temperature,
        )

    response = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
async def chat_json(
    client: LLMClient,
    *,
    model: str,
    prompt: str,
    temperature: float = 0.1,
    cache_dir: Path | None = None,
    system_prompt: str | None = None,
) -> dict:
    provider = "arka" if isinstance(client, ArkaLLMClient) else "openai"
    system = system_prompt or SYSTEM_JSON
    key = _cache_key(prompt, model, provider=provider, system=system)
    if cache_dir:
        cached = _read_cache(cache_dir, key)
        if cached is not None:
            return cached

    if isinstance(client, AsyncOpenAI):
        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
    else:
        content = await _complete_text(
            client,
            model=model,
            system=system,
            user=prompt,
            temperature=temperature,
        )
        data = _parse_json_text(content)

    if cache_dir:
        _write_cache(cache_dir, key, data)
    return data


async def chat_text(
    client: LLMClient,
    *,
    model: str,
    prompt: str,
    temperature: float = 0.1,
    cache_dir: Path | None = None,
) -> str:
    provider = "arka" if isinstance(client, ArkaLLMClient) else "openai"
    system = "You are a precise technical study assistant. Quote source text verbatim when asked."
    key = _cache_key(prompt, model, provider=provider, system=system)
    if cache_dir:
        cached = _read_cache(cache_dir, key)
        if cached is not None and "text" in cached:
            return cached["text"]

    text = await _complete_text(
        client,
        model=model,
        system="You are a precise technical study assistant. Quote source text verbatim when asked.",
        user=prompt,
        temperature=temperature,
    )
    if cache_dir:
        _write_cache(cache_dir, key, {"text": text})
    return text


async def map_concurrent(
    coros: list,
    *,
    limit: int,
) -> list:
    semaphore = asyncio.Semaphore(limit)
    results: list = [None] * len(coros)

    async def run(idx: int, coro):
        async with semaphore:
            results[idx] = await coro

    await asyncio.gather(*(run(i, c) for i, c in enumerate(coros)))
    return results
