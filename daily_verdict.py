"""
daily_verdict.py — ежедневный вердикт за день для общего чата.
"""

from datetime import datetime

from config import DAILY_VERDICT_FALLBACK, DAILY_VERDICT_PROMPT, MODEL
from database import get_global_messages_for_date, has_daily_report, save_daily_report
from llm import ollama_chat
from utils import clean_response


def get_date_key(now_dt: datetime | None = None) -> str:
    now_dt = now_dt or datetime.now()
    return now_dt.strftime("%Y-%m-%d")


def is_due(now_dt: datetime | None = None) -> bool:
    now_dt = now_dt or datetime.now()
    return now_dt.hour >= 22


async def generate_daily_verdict(report_date: str) -> str:
    messages = get_global_messages_for_date(report_date, limit=180)
    if not messages:
        save_daily_report(report_date, DAILY_VERDICT_FALLBACK)
        return DAILY_VERDICT_FALLBACK

    msg_text = "\n".join(f"[{m['username']}]: {m['text']}" for m in messages)
    try:
        resp = await ollama_chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": DAILY_VERDICT_PROMPT},
                {"role": "user", "content": f"Дата: {report_date}\nСообщения дня:\n{msg_text}"},
            ],
            options={"temperature": 0.9, "repeat_penalty": 1.4, "num_ctx": 8192},
        )
        verdict = clean_response(resp["message"]["content"]) or DAILY_VERDICT_FALLBACK
    except Exception as e:
        print(f"[ВЕРДИКТ ДНЯ] ошибка генерации: {e}")
        verdict = DAILY_VERDICT_FALLBACK

    save_daily_report(report_date, verdict)
    return verdict


def should_generate_today(now_dt: datetime | None = None) -> tuple[bool, str]:
    report_date = get_date_key(now_dt)
    return is_due(now_dt) and not has_daily_report(report_date), report_date
