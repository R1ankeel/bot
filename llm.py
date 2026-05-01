"""
llm.py — обёртки над xAI API (Grok).

Интерфейс намеренно совместим со старым Ollama-вариантом:
  - ollama_chat(**kwargs)   → dict с ["message"]["content"]
  - ollama_analyze(**kwargs) → то же самое, для фоновых задач
  - unload_model / preload_model / restart_ollama → заглушки (не нужны для API)

Остальной код (responder, worker, analysis, daily_verdict, main) не меняется.
"""

import asyncio
import time

import httpx

from config import MODEL, MODEL_ANALYSIS, XAI_API_KEY

XAI_URL = "https://api.x.ai/v1/chat/completions"


# ── Внутренний HTTP-клиент ────────────────────────────────────────────

# Один переиспользуемый клиент на весь процесс — быстрее чем создавать каждый раз
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=60,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
        )
    return _http_client


# ── Конвертация формата ────────────────────────────────────────────────
# Ollama принимал: model, messages, options={temperature, repeat_penalty, num_ctx}
# xAI принимает:  model, messages, temperature, max_tokens
# Возвращаем dict в формате Ollama: {"message": {"content": "..."}}

def _build_payload(model: str, messages: list[dict], options: dict) -> dict:
    """Собирает payload для xAI API (Grok)."""
    payload: dict = {
        "model": model,
        "messages": messages,
    }

    # Температура
    if "temperature" in options:
        payload["temperature"] = options["temperature"]

    # ── Борьба с зацикливанием на странных словах пользователя ─────────────────
    # Grok-4 очень чувствителен к repetition_penalty.
    # presence_penalty на модели grok-4-1-fast-non-reasoning-latest НЕ поддерживается → вызывает ошибку 400.

    repetition_penalty = options.get("repeat_penalty") or options.get("repetition_penalty")
    if repetition_penalty is not None:
        payload["repetition_penalty"] = float(repetition_penalty)
    else:
        # Агрессивное значение специально против твоей проблемы с повторением слов
        payload["repetition_penalty"] = 1.38

    # presence_penalty добавляем ТОЛЬКО если он явно передан в options
    # (по умолчанию НЕ ставим, чтобы не было ошибки 400)
    presence_penalty = options.get("presence_penalty")
    if presence_penalty is not None:
        payload["presence_penalty"] = float(presence_penalty)

    # Лимит токенов
    payload["max_tokens"] = options.get("max_tokens", 350)

    return payload


def _wrap_response(content: str) -> dict:
    """Оборачивает ответ в формат совместимый с Ollama."""
    return {"message": {"content": content, "role": "assistant"}}


def _extract_chat_completion_content(data: dict) -> str:
    """Достаёт текст из разных вариантов ответа Chat Completions.

    Иногда API/модель может вернуть слегка отличающийся JSON или пустой content.
    Раньше это падало почти без текста ошибки, из-за чего фоновые задачи было сложно чинить.
    """
    try:
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text") or item.get("content") or ""))
                    else:
                        parts.append(str(item))
                return "".join(parts).strip()
    except Exception:
        pass

    # На всякий случай пробуем Responses-like fallback.
    for item in data.get("output", []) if isinstance(data, dict) else []:
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") in ("output_text", "text"):
                    return block.get("text", "")

    raise ValueError(f"Не смог найти текст ответа xAI. JSON keys={list(data.keys()) if isinstance(data, dict) else type(data)}")


# ── Основные функции ───────────────────────────────────────────────────

async def ollama_chat(
    model: str = None,
    messages: list[dict] = None,
    options: dict = None,
    **_kwargs,   # поглощаем think= и другие Ollama-специфичные параметры
) -> dict:
    """Основной LLM-вызов. Интерфейс совместим с Ollama.

    При 503/502/429 делает до 3 попыток с экспоненциальным backoff (1s, 3s, 9s).
    """
    model    = model    or MODEL
    messages = messages or []
    options  = options  or {}

    payload = _build_payload(model, messages, options)

    _RETRY_STATUSES = {429, 502, 503, 504}
    _MAX_RETRIES    = 3

    for attempt in range(_MAX_RETRIES):
        t0 = time.time()
        try:
            client = _get_client()
            resp = await client.post(XAI_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = _extract_chat_completion_content(data)
            dt = time.time() - t0
            print(f"  [xAI] {model} — {dt:.1f}с, {len(content)} симв")
            return _wrap_response(content)

        except httpx.HTTPStatusError as e:
            dt = time.time() - t0
            status = e.response.status_code
            print(f"  [xAI] HTTP {status} за {dt:.1f}с (попытка {attempt+1}/{_MAX_RETRIES}): {e.response.text[:120]}")
            if status in _RETRY_STATUSES and attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"  [xAI] ожидаю {wait}с перед повтором...")
                await asyncio.sleep(wait)
                continue
            raise

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ReadError, httpx.TransportError) as e:
            dt = time.time() - t0
            print(f"  [xAI] сетевая ошибка за {dt:.1f}с (попытка {attempt+1}/{_MAX_RETRIES}): {type(e).__name__}: {e!r}")
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  [xAI] ожидаю {wait}с перед повтором...")
                await asyncio.sleep(wait)
                continue
            raise

        except Exception as e:
            dt = time.time() - t0
            print(f"  [xAI] ошибка за {dt:.1f}с: {type(e).__name__}: {e!r}")
            raise


async def ollama_analyze(
    model: str = None,
    messages: list[dict] = None,
    options: dict = None,
    **_kwargs,
) -> dict:
    """LLM-вызов для фоновой аналитики (факты, репутация, дневник).

    Используем ту же модель что и для чата — MODEL_ANALYSIS из config.
    num_ctx игнорируется (у API нет этого параметра).
    """
    model    = model    or MODEL_ANALYSIS
    messages = messages or []
    options  = options  or {}

    # Для аналитики температуру держим низкой если не задана явно
    options.setdefault("temperature", 0.2)
    # Факты могут давать длинные списки — берём запас
    options.setdefault("max_tokens", 1500)

    return await ollama_chat(model=model, messages=messages, options=options)


# ── Заглушки swap-логики (были нужны для Ollama, API не требует) ──────

async def unload_model(model: str):
    """Заглушка — в API-режиме выгружать нечего."""
    pass


async def preload_model(model: str):
    """Заглушка — в API-режиме прогревать нечего."""
    pass


async def restart_ollama():
    """Заглушка — Ollama больше не используется."""
    pass


# ── Поиск через нативный web_search Grok (Responses API) ──────────────
# Chat Completions /v1/chat/completions не поддерживает web_search —
# нужен отдельный endpoint /v1/responses.

XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"


async def ollama_chat_with_web_search(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> dict:
    """Вызов Grok с web_search через Responses API (/v1/responses).

    messages — стандартный список [{"role": "system/user/assistant", ...}].
    Системное сообщение автоматически переносится в поле `instructions`.
    """
    model = model or MODEL

    # Responses API: системный промпт — в `instructions`, остальное — в `input`
    instructions = ""
    input_messages = []
    for msg in messages:
        if msg["role"] == "system":
            instructions = msg["content"]
        else:
            input_messages.append(msg)

    payload: dict = {
        "model": model,
        "input": input_messages,
        "tools": [{"type": "web_search"}],
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if instructions:
        payload["instructions"] = instructions

    _RETRY_STATUSES = {429, 502, 503, 504}
    _MAX_RETRIES    = 3

    for attempt in range(_MAX_RETRIES):
        t0 = time.time()
        try:
            client = _get_client()
            resp = await client.post(XAI_RESPONSES_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Responses API возвращает output[] — ищем блок с текстом
            content = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") == "output_text":
                            content = block.get("text", "")
                            break
                if content:
                    break

            if not content:
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

            dt = time.time() - t0
            print(f"  [xAI web_search] {model} — {dt:.1f}с, {len(content)} симв")
            return _wrap_response(content)

        except httpx.HTTPStatusError as e:
            dt = time.time() - t0
            status = e.response.status_code
            print(f"  [xAI web_search] HTTP {status} за {dt:.1f}с (попытка {attempt+1}/{_MAX_RETRIES}): {e.response.text[:120]}")
            if status in _RETRY_STATUSES and attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  [xAI web_search] ожидаю {wait}с перед повтором...")
                await asyncio.sleep(wait)
                continue
            raise

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            dt = time.time() - t0
            print(f"  [xAI web_search] сетевая ошибка за {dt:.1f}с (попытка {attempt+1}/{_MAX_RETRIES}): {e}")
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                await asyncio.sleep(wait)
                continue
            raise

        except Exception as e:
            dt = time.time() - t0
            print(f"  [xAI web_search] ошибка за {dt:.1f}с: {e}")
            raise
