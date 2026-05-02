"""
responder.py — генерация ответов: основной, поиск, приветствие,
мнение о пользователе, совместимость и сводка дневника.
"""

import random
import re
import time

from config import (
    CHARACTER_NAME,
    CHAT_TEMPERATURE_BASE,
    CHAT_TEMPERATURE_RETRY,
    FALLBACK_RESPONSE,
    GREETING_FALLBACK_TEMPLATE,
    GREETING_STYLES,
    GREETING_TEMPERATURE,
    MAX_HISTORY,
    MODEL,
    MODEL_DIARY_SUMMARY,
    MOODS,
    SEARCH_ERROR_TEMPLATE,
    SEARCH_TEMPERATURE,
)
from database import get_all_facts_summary, get_facts, get_history, get_history_hybrid, get_traits, save_message, update_profile_realtime
from llm import ollama_chat, ollama_chat_with_web_search
from prompt_builder import (
    PROMPT_BUDGET_CHAT,
    PROMPT_BUDGET_CONTEXT,
    PROMPT_BUDGET_LIGHT,
    _estimate_tokens,
    append_cross_chat_context,
    build_mentioned_people_block,
    build_person_context_block,
    build_recent_chat_block,
    build_relation_question_instruction,
    extract_target_username,
    get_recent_bot_replies,
    get_system_prompt,
    is_gossip_question,
    is_relation_question,
)
from reputation import LEVELS, get_progress_cap, get_reputation
from utils import clean_response, ensure_nick, strip_urls, fix_username_in_response, deduplicate_username_in_response, normalize_strict_nick_prefix, has_name_trigger, SELF_IDENTITY_MODE_MARKER, get_search_kind
from state import buffer_add, should_include_facts as state_include_facts, check_and_block_patterns, update_user_emotion, reset_user_runtime_state
import state
from jealousy import maybe_ignore_lover_message


USER_NICK_GREETING_FALLBACKS = [
    "{username}, ты так и не придумал себе ник, да? у тебя имя звучит как тестовый аккаунт.",
    "{username}, я смотрю, фантазия взяла выходной и оставила тебе заводской ник.",
    "{username}, это ник или серийный номер, который забыли поменять после регистрации?",
    "{username}, ну да, зачем ник придумывать, когда можно жить как временный файл.",
    "{username}, выглядит так, будто ты зашёл в чат прямиком из меню регистрации.",
]

NICK_JOKE_GREETING_FALLBACKS = [
    "{username}, ник у тебя звучит так, будто за ним уже должна идти какая-то странная история.",
    "{username}, с таким ником ты либо легенда, либо очень подозрительный сайд-квест.",
    "{username}, у тебя ник как будто сам просит, чтобы над ним немного поиздевались.",
    "{username}, ник мощный, конечно. ощущение, будто ты либо босс, либо мем.",
    "{username}, у тебя такой ник, что хочется сначала пошутить, а потом уже здороваться.",
]


def _is_default_user_nick(username: str) -> bool:
    return re.fullmatch(r"user\d+", username.strip(), flags=re.IGNORECASE) is not None


# ── Детектор повторов ─────────────────────────────────────────────────

def _extract_reply_core(text: str) -> str:
    text = re.sub(r'^[^,]+,\s*', '', text)
    text = re.sub(r'[^\w\s]', '', text).strip().lower()
    return text


def _is_repetitive(new_answer: str, username: str, threshold: float = 0.55) -> bool:
    recent = get_recent_bot_replies(username, 5)
    if not recent:
        return False
    new_core = _extract_reply_core(new_answer)
    new_words = set(new_core.split())
    if not new_words:
        return False
    for old in recent:
        old_words = set(_extract_reply_core(old).split())
        if not old_words:
            continue
        overlap = len(new_words & old_words) / max(len(new_words), len(old_words))
        if overlap >= threshold:
            return True
    return False


def _get_last_assistant_message(history: list[dict]) -> str:
    for msg in reversed(history):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


_FOLLOWUP_SHORT_REPLIES: frozenset[str] = frozenset({
    "да", "нет", "ага", "неа", "угу", "понятно",
    "хорошо", "норм", "нормально", "отлично",
    "плохо", "так себе", "не очень", "сойдет", "пойдет", "сойдёт", "пойдёт",
    "ок", "окей", "ладно", "ясно", "именно", "возможно", "наверное",
    "конечно", "естественно", "точно", "скорее да", "скорее нет",
    "ну да", "ну нет", "ну ладно", "ну ок",
})

# Ссылка на уже сказанное / ответ на вопрос (подстроки, нижний регистр).
_FOLLOWUP_REFERENCE_MARKERS: tuple[str, ...] = (
    "об этом",
    "про это",
    "про то",
    "насчёт",
    "на счёт",
    "в том плане",
    "в этом плане",
    "как ты и",
    "ты же",
    "одно и то же",
    "то же самое",
    "именно так",
    "как раз",
    "согласен",
    "согласна",
    "не согласен",
    "типа того",
    "примерно так",
    "до этого",
    "как сказать",
    "трудно сказать",
    "сложно сказать",
    "мне кажется",
    "по-моему",
    "наверное так",
    "видимо так",
    "на этот счёт",
    "на тот счёт",
    "из-за этого",
    "именно про",
    "тот самый",
    "та самая",
    "это же",
    "то же",
    "так что да",
    "так что нет",
    "раньше ",
    "ещё тогда",
    "еще тогда",
    "по твоему",
    "по-твоему",
)


def _looks_like_followup_answer(text: str, username: str = "") -> bool:
    t = (text or "").strip().lower()
    if not t or "?" in t:
        return False

    t_plain = t.rstrip(".,!?…").strip()
    if t_plain in _FOLLOWUP_SHORT_REPLIES or t in _FOLLOWUP_SHORT_REPLIES:
        return True

    if any(m in t for m in _FOLLOWUP_REFERENCE_MARKERS):
        # Не цепляем длинные монологи, где маркер случайно встретился.
        if len(t) <= 100 and len(t.split()) <= 16:
            return True

    has_thread_context = bool(
        username
        and (
            state.get_pending_dialog(username)
            or state.get_dialog_state(username)
        )
    )
    if has_thread_context and (len(t) <= 40 or len(t.split()) <= 6):
        return True

    return False


def _extract_open_question(text: str) -> str:
    """Извлекает последний вопрос из ответа бота, если он есть.

    Нужно чтобы сохранять pending_dialog — то, что бот спросил.
    """
    if "?" not in text:
        return ""
    # Ищем последнее предложение с вопросительным знаком
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for sent in reversed(sentences):
        if "?" in sent:
            return sent.strip()
    return ""


# ── Вспомогательное ───────────────────────────────────────────────────

def _needs_recent_chat_context(user_message: str) -> bool:
    t = (user_message or "").lower()
    markers = (
        "кто", "что", "где", "когда", "кому", "кого",
        "сказал", "сказала", "писал", "писала",
        "выше", "до этого", "только что", "сейчас",
        "про кого", "о ком", "что было", "что происходит",
        # реакции на общий чат
        "ахах", "ору", "жесть", "кринж", "ну вы", "вы тут", "опять",
        "понеслась", "мда",
    )
    return "?" in t or any(m in t for m in markers)


def _estimate_msg_tokens(messages: list[dict]) -> int:
    return sum(max(1, len((m.get("content") or "")) // 3) for m in messages)


def _hard_trim_system_prompt(prompt: str, max_chars: int = 4200) -> str:
    if len(prompt) <= max_chars:
        return prompt

    kept = prompt[:max_chars].rstrip()
    cut = kept.rfind("\n\n")
    if cut > 2500:
        kept = kept[:cut].rstrip()

    return kept + "\n\n[Часть второстепенного контекста убрана ради краткости.]"


def _trim_history_to_budget(history: list[dict], budget_tokens: int = 550) -> list[dict]:
    if not history:
        return history

    kept = list(history)

    while kept and _estimate_msg_tokens(kept) > budget_tokens:
        if len(kept) <= 4:
            break

        sep_idx = next(
            (
                i
                for i, m in enumerate(kept)
                if m.get("role") == "system" and "опущены" in (m.get("content") or "")
            ),
            None,
        )
        if sep_idx is not None:
            kept.pop(sep_idx)
            continue

        kept.pop(0)

    return kept


def _looks_like_weather_hallucination(text: str) -> bool:
    t = (text or "").lower()
    weather_words = (
        "градус", "°", "+", "дожд", "снег", "пасмур", "солнеч",
        "облач", "ветер", "тепло", "холодно", "жара", "мороз",
        "прогноз", "температур",
    )
    has_weather_word = any(w in t for w in weather_words)
    has_number = bool(re.search(r"[-+]?\d{1,2}\s*(?:°|градус|c|с)?", t))
    return has_weather_word and has_number

def _postprocess_answer(text: str, username: str) -> str:
    """Финальная постобработка ответа.

    ВАЖНО: порядок шагов критичен. Ранее `ensure_nick()` вызывался первым и мог
    добавлять ник поверх уже существующего (или частично искажённого) префикса,
    после чего последующие "починки" портили строку: "Скуф, ск uf, ...".

    Безопасная цепочка:
    1) normalize_strict_nick_prefix — если ник УЖЕ корректно в начале, приводим
       разделитель к `Ник, ...` (без fuzzy).
    2) fix_username_in_response — при необходимости чинит ТОЛЬКО префикс по строгим
       условиям (высокий порог, проверки длины/первых букв). НЕ трогает тело.
    3) deduplicate_username_in_response — удаляет все повторные упоминания ника,
       кроме первого (учитывает короткие ники и ники с пробелами).
    4) normalize_strict_nick_prefix — ещё раз стабилизируем формат после вырезаний.
    5) ensure_nick — только если ника всё ещё нет в начале.
    """
    t = (text or "").strip()
    t = normalize_strict_nick_prefix(t, username)
    t = fix_username_in_response(t, username)
    t = deduplicate_username_in_response(t, username)   # ← должен быть здесь
    t = normalize_strict_nick_prefix(t, username)       # стабилизация
    t = ensure_nick(t, username)
    return (t or "").strip()


async def _generate_answer_with_prompt(username: str, history: list[dict], system_prompt: str) -> str:
    history = _trim_history_to_budget(history, budget_tokens=550)
    total_est = _estimate_tokens(system_prompt) + _estimate_msg_tokens(history)
    print(
        f"  [TOKENS] system≈{_estimate_tokens(system_prompt)}, "
        f"history≈{_estimate_msg_tokens(history)}, total≈{total_est}"
    )

    rep = get_reputation(username)
    is_hate = rep["level"] >= 8

    answer = None
    for attempt in range(2):
        if is_hate:
            temp = 0.95 if attempt == 0 else 1.05
            repeat_penalty = 1.6
        else:
            temp = CHAT_TEMPERATURE_BASE if attempt == 0 else CHAT_TEMPERATURE_RETRY
            repeat_penalty = 1.5

        t0 = time.time()
        resp = await ollama_chat(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}, *history],
            options={
                "temperature": temp,
                "repeat_penalty": repeat_penalty,
                "num_ctx": 8192,
            },
        )
        dt = time.time() - t0
        answer = clean_response(resp["message"]["content"], username)
        print(f"  [xAI] ответ: {len(answer)} символов")
        if not _is_repetitive(answer, username):
            break
        print(f"  [АНТИПОВТОР] попытка {attempt+1}: повтор, перегенерирую (temp={temp:.2f})...")
    return answer or FALLBACK_RESPONSE


async def _get_cross_user_answer(
    username: str, user_message: str, history: list[dict], mood: str, target: str
) -> str:
    extra_blocks = [
        build_person_context_block(target, include_rep=True, include_facts=True, include_traits=True),
        build_mentioned_people_block(user_message, username),
    ]
    base_prompt = get_system_prompt(
        username,
        include_facts_flag=state_include_facts(username),
        allow_style_hint=False,
        extra_blocks=extra_blocks,
        include_story=True,
        budget_tokens=PROMPT_BUDGET_CONTEXT,
    )
    if mood and mood in MOODS:
        base_prompt += f"\n\nТЕКУЩЕЕ НАСТРОЕНИЕ: {MOODS[mood]}"
    base_prompt = append_cross_chat_context(base_prompt, user_message, username)
    base_prompt = _hard_trim_system_prompt(base_prompt, max_chars=4200)
    relation_instruction = build_relation_question_instruction(username, target, user_message)
    answer = await _generate_answer_with_prompt(
        username,
        history + [{"role": "user", "content": relation_instruction}],
        base_prompt,
    )
    return _postprocess_answer(answer, username)


def _truncate_text_soft(text: str, max_len: int = 780) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    cut = text.rfind(" ", 0, max_len)
    if cut == -1:
        cut = max_len
    trimmed = text[:cut].rstrip(" ,;:-")
    return trimmed + "…"


def _normalize_diary_entry_text(raw: str) -> str:
    text = re.sub(r"^запись в дневник:\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -•\n\t")


def _fallback_daily_diary_digest(entries: list[dict], max_len: int = 780) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        raw = _normalize_diary_entry_text(str(entry.get("entry", "")))
        if not raw:
            continue
        norm = raw.lower()
        if norm in seen:
            continue
        seen.add(norm)
        parts.append(raw)
    if not parts:
        return "за сегодня дневник пока пуст"
    digest = " ".join(parts)
    return _truncate_text_soft(digest, max_len=max_len)


def _looks_like_diary_prompt_leak(text: str) -> bool:
    t = text.lower()
    markers = (
        "записи за день", "начни строго", "максимум 4-6", "сделай одну живую выжимку",
        "текущее настроение", "стиль этого ответа", "ты не ии", "правила общения",
        "душа компании", "формат:", "──", "══", "системный промпт",
    )
    if any(marker in t for marker in markers):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = sum(1 for line in lines if line.startswith(("-", "•")))
    return bullet_lines >= 2


def _looks_like_diary_dump(answer: str, entry_texts: list[str]) -> bool:
    if _looks_like_diary_prompt_leak(answer):
        return True
    normalized_answer = _normalize_diary_entry_text(answer).lower()
    if not normalized_answer:
        return True

    exact_hits = 0
    for entry in entry_texts:
        entry_norm = _normalize_diary_entry_text(entry).lower()
        if entry_norm and entry_norm in normalized_answer:
            exact_hits += 1
    if exact_hits >= 2:
        return True

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if sum(1 for line in lines if line.startswith(("-", "•"))) >= 2:
        return True
    return False


def _fallback_daily_diary_summary(entries: list[dict], max_len: int = 780) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        raw = _normalize_diary_entry_text(str(entry.get("entry", "")))
        if not raw:
            continue
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(raw)

    if not parts:
        return "за сегодня дневник пока пуст"

    fragments: list[str] = []
    for raw in parts:
        piece = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)[0].strip()
        piece = piece.rstrip(" .")
        if piece:
            fragments.append(piece)

    if not fragments:
        return "за сегодня дневник пока пуст"

    openers = {
        1: "за сегодня на планете вайб был довольно цельный:",
        2: "за сегодня на планете день собрался такой:",
        3: "за сегодня на планете день был рваный, но в целом такой:",
    }
    opener = openers.get(min(len(fragments), 3), "за сегодня на планете крутился такой общий вайб:")

    text = opener
    connectors = ["сначала", "потом", "дальше", "под конец"]
    for idx, fragment in enumerate(fragments):
        fragment = fragment[0].lower() + fragment[1:] if fragment else fragment
        chunk = fragment if idx == 0 else f"{connectors[min(idx, len(connectors) - 1)]} {fragment}"
        candidate = f"{text} {chunk}."
        if len(candidate) > max_len:
            break
        text = candidate

    if len(parts) > 1 and "в целом" not in text.lower():
        closing = " в целом сутки вышли живыми и не пустыми."
        if len(text) + len(closing) <= max_len:
            text += closing

    return _truncate_text_soft(text, max_len=max_len)


# ── Публичные функции ─────────────────────────────────────────────────

async def get_response(username: str, user_message: str, mood: str = "") -> str:
    is_self_identity = SELF_IDENTITY_MODE_MARKER in (user_message or "")
    save_message(username, "user", f"Ник: {username}\nСообщение: {user_message}")

    # Realtime profile update — немедленно обновляем профиль если юзер
    # сказал что-то типа «меня зовут Саша», «мне 22 года», «я из Питера»
    update_profile_realtime(username, user_message)

    # Emotion tracking — обновляем тон диалога для current-thread block
    update_user_emotion(username, user_message)

    # Иногда ревнивая Анька просто игнорит любимчика.
    if maybe_ignore_lover_message(username, user_message):
        print(f"  [РЕВНОСТЬ] {username}: намеренно без ответа")
        if not is_self_identity:
            state.update_dialog_state(username, user_message, None)
        return ""

    # Гибридная история: последние ходы + якорные первые реплики (компактный бюджет)
    history = get_history_hybrid(username, recent_turns=5, anchor_turns=1)
    # В SELF_IDENTITY-режиме вопрос про саму Аньку; нельзя вытаскивать
    # случайный ник из структурированной карточки как target_user.
    target_user = "" if is_self_identity else extract_target_username(user_message, asker=username)
    # Для fallback followup detection — берём обычную историю для последнего сообщения
    history_plain = get_history(username, limit=4)
    history_before_user = history_plain[:-1] if history_plain and history_plain[-1]["role"] == "user" else history_plain
    last_assistant = _get_last_assistant_message(history_before_user)
    is_followup_answer = last_assistant.strip().endswith("?") and _looks_like_followup_answer(
        user_message, username
    )

    try:
        is_about_someone = (
            is_relation_question(user_message)
            or is_gossip_question(user_message)
            # Fallback: если нашли target и вопрос адресован боту — тоже сплетница
            or (target_user and "?" in user_message and has_name_trigger(user_message))
        )
        if (not is_self_identity) and target_user and is_about_someone:
            print(f"  [СПЛЕТНИЦА] специальный ответ про {target_user} для {username}")
            answer = await _get_cross_user_answer(username, user_message, history, mood, target_user)
            save_message(username, "assistant", answer)
            buffer_add(CHARACTER_NAME, answer)
            state.update_dialog_state(username, user_message, answer)
            return answer

        extra_blocks = []

        # ── Follow-up: бот помнит что сам спросил ─────────────────
        # Проверяем pending_dialog (сохранённый вопрос бота) — точнее чем
        # просто "последний ответ заканчивался на ?"
        pending = state.get_pending_dialog(username)
        if pending and _looks_like_followup_answer(user_message, username):
            topic_hint = f" на тему «{pending['topic']}»" if pending.get("topic") else ""
            extra_blocks.append(
                f"Пользователь отвечает на твой вопрос{topic_hint}: «{pending['question']}». "
                "Это продолжение уже начатого тобой диалога. "
                "Не спрашивай зачем он пишет, не начинай разговор заново. "
                "Ответь естественно как на продолжение беседы."
            )
            state.clear_pending_dialog(username)
        elif is_followup_answer:
            # Старый fallback — на случай если pending_dialog не был установлен
            extra_blocks.append(
                "Пользователь, вероятно, отвечает на твой предыдущий вопрос. "
                "Это продолжение уже начатого тобой диалога. "
                "Не спрашивай, зачем он пишет, и не начинай разговор заново. "
                "Ответь естественно как на продолжение беседы."
            )

        if is_self_identity:
            extra_blocks.append(
                "── SELF_IDENTITY режим ──\n"
                "Пользователь передал сторонний вопрос про НейроАньку. "
                "Отвечай про себя от первого лица. "
                "Не подключай контекст других пользователей и не сплетничай о них. "
                "Не превращай адресата из исходной фразы в объект ответа.\n"
                "────────────────────────"
            )
        else:
            mentioned_block = build_mentioned_people_block(user_message, username)
            if mentioned_block:
                extra_blocks.append(mentioned_block)

            if _needs_recent_chat_context(user_message):
                recent_chat_block = build_recent_chat_block(limit=4, query=user_message)
                if recent_chat_block:
                    extra_blocks.append(recent_chat_block)

        search_kind = get_search_kind(user_message)
        if search_kind == "blocked_weather":
            extra_blocks.append(
                "Пользователь спрашивает погоду или актуальный прогноз. "
                "НЕ используй поиск. НЕ называй температуру, осадки, ветер, прогноз и любые конкретные погодные данные. "
                "Ты не знаешь текущую погоду. Ответь как персонаж: уклонись, пошути, предложи выглянуть в окно или открыть погоду самому. "
                "Главное: никаких чисел и утверждений о реальной погоде."
            )
        elif search_kind == "blocked_science":
            extra_blocks.append(
                "Пользователь спрашивает сложную научную/учебную/энциклопедическую тему. "
                "НЕ используй поиск. НЕ изображай эксперта и НЕ давай уверенное учебное объяснение. "
                "Ответь как обычная девчонка из чата: можно признаться, что тебе это неинтересно/сложно, "
                "пошутить или дать очень бытовую метафору без претензии на точность."
            )

        mood_prompt = get_system_prompt(
            username,
            include_facts_flag=state_include_facts(username),
            allow_style_hint=True,
            extra_blocks=extra_blocks,
            include_story=True,
            query=user_message,   # для salience ranking фактов
            budget_tokens=PROMPT_BUDGET_CHAT,
        )
        if is_self_identity:
            mood_prompt += (
                "\n\nВАЖНО — SELF_IDENTITY:\n"
                "Тебя обсуждают в третьем лице. Ответь от первого лица: кто ты / какая ты / норм ли ты. "
                "Не отвечай про других людей из сообщения. Не используй recent chat context. "
                "Можно быть язвительной, живой и короткой, но смысл должен быть про НейроАньку. "
                "Начало ответа — ник автора сообщения."
            )
        if mood and mood in MOODS:
            mood_prompt += f"\n\nТЕКУЩЕЕ НАСТРОЕНИЕ: {MOODS[mood]}"
        if not is_self_identity:
            mood_prompt = append_cross_chat_context(mood_prompt, user_message, username)

        mood_prompt = _hard_trim_system_prompt(mood_prompt, max_chars=4200)

        # ── Автопоиск ─────────────────────────────────────────────
        # ИСПРАВЛЕНИЕ: раньше результат поиска заменял системный промпт персонажа
        # (mood_prompt = strip_urls(search_result)) — это убивало характер Аньки.
        # Теперь: поиск выполняется с промптом персонажа и возвращается напрямую.
        # Это корректно, потому что ollama_chat_with_web_search уже получает
        # mood_prompt как system, и ответ уже отформатирован в стиле персонажа.
        if search_kind in ("culture", "factual"):
            print(f"  → [автопоиск:{search_kind}] Grok web_search: «{user_message}»")

            if search_kind == "culture":
                search_instruction = (
                    f"Пользователь написал: {user_message}\n\n"
                    "Пользователь спрашивает про музыку/песни/фильмы/игры/аниме/мемы или другой культурный контент. "
                    "Используй web_search, чтобы не выдумывать названия, факты и строки. "
                    "Если речь о песне — НЕ давай полный текст и длинные цитаты; "
                    "можно максимум очень короткий фрагмент, а лучше передай вайб своими словами. "
                    "Если просят «напой» — не выдавай фейковый официальный текст: "
                    "можешь дать короткую авторскую стилизацию и честно отметить, что это не дословная цитата. "
                    "Ответь коротко, в стиле НейроАньки. "
                    "Не говори 'я нашла в интернете' или 'поискал'. "
                    "ВАЖНО: не вставляй ссылки, URL и адреса сайтов в ответ."
                )
            else:
                search_instruction = (
                    f"Пользователь написал: {user_message}\n\n"
                    "Это фактический вопрос про новости/даты/актуальные события/цены/расписание/результаты. "
                    "Используй web_search инструмент, найди нужные данные и дай ответ в моём стиле — "
                    "коротко, по-свойски, как НейроАнька. "
                    "Не говори 'я нашла в интернете' или 'поискал'. Просто ответь естественно. "
                    "ВАЖНО: не вставляй ссылки, URL и адреса сайтов в ответ."
                )

            search_messages = [
                {"role": "system", "content": mood_prompt},
                {
                    "role": "user",
                    "content": search_instruction
                }
            ]

            try:
                resp = await ollama_chat_with_web_search(
                    messages=search_messages,
                    temperature=0.68,
                    max_tokens=850,
                )
                # Поиск прошёл успешно — ответ уже сгенерирован с CHARACTER промптом.
                # Возвращаем его напрямую, не гоним через _generate_answer_with_prompt ещё раз.
                search_answer = _postprocess_answer(
                    clean_response(strip_urls(resp["message"]["content"]), username),
                    username,
                )
                if search_answer and search_answer.strip():
                    save_message(username, "assistant", search_answer)
                    buffer_add(CHARACTER_NAME, search_answer)
                    check_and_block_patterns(username, search_answer)
                    print(f"  → [автопоиск] ответ возвращён напрямую ({len(search_answer)} симв)")
                    _update_pending_dialog(username, search_answer)
                    if not is_self_identity:
                        state.update_dialog_state(username, user_message, search_answer)
                    return search_answer
            except Exception as e:
                print(f"  → [автопоиск] ошибка Grok web_search: {e}")
                # Поиск упал — продолжаем без него (обычная генерация ниже)

        answer = await _generate_answer_with_prompt(username, history, mood_prompt)
        if not answer or not answer.strip():
            answer = FALLBACK_RESPONSE
        answer = _postprocess_answer(answer, username)
        if search_kind == "blocked_weather" and _looks_like_weather_hallucination(answer):
            answer = (
                f"{username}, я не погодный виджет. "
                "открой прогноз сам, а то я сейчас уверенно навру и ещё обижусь 😏"
            )

        # Обновляем паттерн-блокировки и pending dialog
        check_and_block_patterns(username, answer)
        _update_pending_dialog(username, answer)

        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        if not is_self_identity:
            state.update_dialog_state(username, user_message, answer)
        return answer

    except Exception as e:
        print(f"Ошибка ollama: {e}")
        return f"{username}, {FALLBACK_RESPONSE}"


def _update_pending_dialog(username: str, bot_answer: str):
    """Если в ответе бота есть вопрос — сохраняем его как pending."""
    from state import set_pending_dialog
    question = _extract_open_question(bot_answer)
    if question:
        set_pending_dialog(username, question)


async def get_forced_search_response(username: str, query: str, mood: str = "") -> str:
    """!найди — использует нативный web_search Grok."""
    print(f"  → [!найди] Ищу через Grok: «{query}»")

    save_message(username, "user", f"Ник: {username}\nКоманда: !найди {query}")

    system_prompt = get_system_prompt(username, budget_tokens=PROMPT_BUDGET_LIGHT)
    if mood and mood in MOODS:
        system_prompt += f"\n\nТЕКУЩЕЕ НАСТРОЕНИЕ: {MOODS[mood]}"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Пользователь попросил найти информацию: {query}\n"
                f"Используй web_search и дай точный, по-свойски ответ в стиле Аньки. "
                f"Никогда не говори 'загугли', 'я нашла в интернете', 'вот что я нашла' — просто дай ответ. "
                f"ВАЖНО: не вставляй ссылки, URL и адреса сайтов в ответ. "
                f"Начни строго с ника {username}."
            ),
        },
    ]

    try:
        resp = await ollama_chat_with_web_search(
            messages=messages,
            temperature=0.65,
            max_tokens=900,
        )
        raw_content = strip_urls(resp["message"]["content"])
        answer = _postprocess_answer(clean_response(raw_content, username), username)
        print(f"  → [!найди] Ответ готов ({len(answer)} симв)")
        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer

    except Exception as e:
        print(f"  → [!найди] Ошибка: {e}")
        return SEARCH_ERROR_TEMPLATE.format(username=username)


async def get_self_opinion_response(username: str, mood: str = "") -> str:
    save_message(username, "user", f"Ник: {username}\nКоманда: !мнение\nПользователь просит коротко описать моё мнение о нём.")
    history = get_history(username, limit=MAX_HISTORY)

    prompt = get_system_prompt(
        username,
        include_facts_flag=True,
        allow_style_hint=False,
        include_story=True,
        budget_tokens=PROMPT_BUDGET_CHAT,
    )
    prompt += (
        "\n\nТебя попросили описать своё текущее мнение о пользователе. "
        "Опирайся на уровень отношений, факты, ярлыки и общий вайб общения. "
        "Не перечисляй всё списком и не отвечай сухо. "
        "Начни строго с ника пользователя. Максимум 2 предложения."
    )
    if mood and mood in MOODS:
        prompt += f"\n\nТЕКУЩЕЕ НАСТРОЕНИЕ: {MOODS[mood]}"

    try:
        answer = await _generate_answer_with_prompt(username, history, prompt)
        answer = _postprocess_answer(answer, username)
        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer
    except Exception as e:
        print(f"Ошибка !мнение: {e}")
        return f"{username}, ты у меня пока вызываешь слишком много мыслей, чтобы уложить это нормально 😏"


async def get_compatibility_response(asker: str, nick1: str, nick2: str, mood: str = "") -> str:
    if nick1.lower() == nick2.lower():
        answer = f"{asker}, ну у {nick1} с {nick2} совместимость идеальная, это literally один и тот же человек."
        save_message(asker, "user", f"Ник: {asker}\nКоманда: !совместимость {nick1} {nick2}")
        save_message(asker, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer

    save_message(asker, "user", f"Ник: {asker}\nКоманда: !совместимость {nick1} {nick2}\nПользователь просит оценить совместимость двух людей.")
    history = get_history(asker, limit=MAX_HISTORY)

    extra_blocks = [
        build_person_context_block(nick1, include_rep=True, include_facts=True, include_traits=True),
        build_person_context_block(nick2, include_rep=True, include_facts=True, include_traits=True),
    ]
    prompt = get_system_prompt(
        asker,
        include_facts_flag=False,
        allow_style_hint=False,
        extra_blocks=extra_blocks,
        include_story=False,
        budget_tokens=PROMPT_BUDGET_CONTEXT,
    )
    prompt += (
        f"\n\nТебя попросили оценить совместимость {nick1} и {nick2}. "
        "Смотри на их вайб, ярлыки, факты и на то, как они ощущаются в этом чате. "
        "Это может быть совместимость как пары, дуэта, друзей или хаотичного союза. "
        "Дай короткий вердикт и одну причину. Не выдумывай факты. "
        f"Начни строго с ника {asker}. Максимум 2 предложения."
    )
    if mood and mood in MOODS:
        prompt += f"\n\nТЕКУЩЕЕ НАСТРОЕНИЕ: {MOODS[mood]}"

    try:
        answer = await _generate_answer_with_prompt(asker, history, prompt)
        answer = _postprocess_answer(answer, asker)
        save_message(asker, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer
    except Exception as e:
        print(f"Ошибка !совместимость: {e}")
        return f"{asker}, у них вайб либо взорвётся, либо сработается, и это уже звучит как нормальная совместимость 😏"


async def get_find_pair_response(username: str, mood: str = "") -> str:
    """Шутливо подбирает пользователю пару из людей, о которых есть память."""
    save_message(username, "user", f"Ник: {username}\nКоманда: !найдипару")

    # Данные запрашивающего
    my_facts = get_facts(username, limit=12)
    my_traits = get_traits(username, limit=8)

    if not my_facts and not my_traits:
        answer = (
            f"{username}, я о тебе пока почти ничего не знаю — "
            f"пообщайся со мной побольше, тогда найду тебе пару."
        )
        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer

    # Все пользователи с данными
    all_facts_map = get_all_facts_summary()  # {username: [facts]}

    # Кандидаты: все кроме самого юзера, у кого есть хоть какие-то данные
    candidates: list[dict] = []
    for cand_name, cand_facts in all_facts_map.items():
        if cand_name.lower() == username.lower():
            continue
        cand_traits = get_traits(cand_name, limit=6)
        if not cand_facts and not cand_traits:
            continue
        candidates.append({
            "username": cand_name,
            "facts": cand_facts[:10],
            "traits": cand_traits,
        })

    if not candidates:
        answer = (
            f"{username}, пока не с кем сравнивать — "
            f"в чате ещё мало людей с историей. Возвращайся позже."
        )
        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer

    # Формируем блок с данными по кандидатам (топ 15 по количеству данных)
    candidates.sort(key=lambda c: len(c["facts"]) + len(c["traits"]), reverse=True)
    top_candidates = candidates[:15]

    cand_block = ""
    for c in top_candidates:
        facts_str = "; ".join(c["facts"]) if c["facts"] else "нет данных"
        traits_str = "; ".join(c["traits"]) if c["traits"] else "нет ярлыков"
        cand_block += f"\n[{c['username']}]\nФакты: {facts_str}\nЯрлыки: {traits_str}\n"

    my_facts_str = "; ".join(my_facts) if my_facts else "нет данных"
    my_traits_str = "; ".join(my_traits) if my_traits else "нет ярлыков"

    mood_desc = MOODS.get(mood, "")
    mood_line = f"Твоё настроение сейчас: {mood_desc}. " if mood_desc else ""

    user_prompt = (
        f"Ник: {username}\n"
        f"Данные о {username}:\n"
        f"Факты: {my_facts_str}\n"
        f"Ярлыки: {my_traits_str}\n\n"
        f"Кандидаты в пару:\n{cand_block}\n"
        f"{mood_line}"
        f"Выбери одного человека из списка кандидатов как идеальную пару для {username}. "
        f"Обоснование должно быть конкретным — используй реальные факты из данных обоих людей, "
        f"найди смешное, неожиданное или абсурдное совпадение / противоречие. "
        f"Тон — твой: ироничный, немного язвительный, но с теплотой. "
        f"Можно одну короткую дополнительную шутку в конце. "
        f"2–3 предложения. Начни с ника {username}."
    )

    try:
        resp = await ollama_chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": get_system_prompt(username, budget_tokens=PROMPT_BUDGET_LIGHT)},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.95, "repeat_penalty": 1.3, "max_tokens": 400},
        )
        result = clean_response(resp["message"]["content"], username)
        result = _postprocess_answer(result, username)
        answer = result or f"{username}, звёзды не определились. Видимо, ты слишком сложная личность."
        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer
    except Exception as e:
        print(f"[НАЙДИПАРУ] ошибка: {e}")
        answer = f"{username}, что-то пошло не так. Попробуй ещё раз."
        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer


ZODIAC_SIGNS = {
    "овен": "Овен",
    "телец": "Телец",
    "близнецы": "Близнецы",
    "рак": "Рак",
    "лев": "Лев",
    "дева": "Дева",
    "весы": "Весы",
    "скорпион": "Скорпион",
    "стрелец": "Стрелец",
    "козерог": "Козерог",
    "водолей": "Водолей",
    "рыбы": "Рыбы",
}


async def get_horoscope_response(username: str, sign_raw: str, mood: str = "") -> str:
    """Генерирует персональный гороскоп для пользователя по его памяти."""
    sign_key = (sign_raw or "").lower().strip(".,!?;: ")
    sign = ZODIAC_SIGNS.get(sign_key)

    save_message(username, "user", f"Ник: {username}\nКоманда: !гороскоп {sign_raw}")

    if not sign:
        answer = (
            f"{username}, не знаю такого знака. Попробуй: овен, телец, близнецы, "
            f"рак, лев, дева, весы, скорпион, стрелец, козерог, водолей, рыбы."
        )
        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer

    facts = get_facts(username, limit=10)
    traits = get_traits(username, limit=8)

    facts_block = "\n".join(f"• {f}" for f in facts) if facts else "нет данных"
    traits_block = "\n".join(f"• {t}" for t in traits) if traits else "нет данных"

    mood_desc = MOODS.get(mood, "")
    mood_line = f"Твоё сейчас настроение: {mood_desc}. " if mood_desc else ""

    system = get_system_prompt(username, budget_tokens=PROMPT_BUDGET_LIGHT)
    user_prompt = (
        f"Ник: {username}\n"
        f"Знак зодиака: {sign}\n\n"
        f"Что я знаю об этом человеке:\n"
        f"Факты:\n{facts_block}\n"
        f"Ярлыки:\n{traits_block}\n\n"
        f"{mood_line}"
        f"Составь персональный гороскоп для {username} на сегодня. "
        f"Используй факты и ярлыки — вплети их в предсказание конкретно, не абстрактно. "
        f"Тон — твой обычный: саркастично, с иронией, местами нежно, но без пафоса. "
        f"Не пиши как настоящий астролог — пиши как ты. "
        f"2–4 предложения. Начни с ника {username}."
    )

    try:
        resp = await ollama_chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.92, "repeat_penalty": 1.3, "max_tokens": 350},
        )
        result = clean_response(resp["message"]["content"], username)
        result = _postprocess_answer(result, username)
        answer = result or f"{username}, звёзды сегодня молчат. Загадочно, да."
        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer
    except Exception as e:
        print(f"[ГОРОСКОП] ошибка: {e}")
        answer = f"{username}, астральный канал временно недоступен, попробуй позже."
        save_message(username, "assistant", answer)
        buffer_add(CHARACTER_NAME, answer)
        return answer


async def get_daily_diary_summary(username: str, entries: list[dict], mood: str = "") -> str:
    """Делает цельную выжимку по всем записям дневника за текущие сутки."""
    if not entries:
        return f"{username}, за сегодня дневник пока пуст"

    entry_texts: list[str] = []
    for entry in entries:
        raw = _normalize_diary_entry_text(str(entry.get("entry", "")))
        if raw:
            entry_texts.append(raw)

    if not entry_texts:
        return f"{username}, за сегодня дневник пока пуст"

    import state as _state
    from config import SYSTEM_PROMPTS as _SYSTEM_PROMPTS

    diary_lines = "\n".join(f"- {text}" for text in entry_texts)
    compact_system_prompt = (
        _SYSTEM_PROMPTS[_state.current_mode]
        + "\n\n"
        "Тебя попросили написать живую сводку того, что происходило сегодня на планете. "
        "Это не аналитика, не список и не разбор данных. "
        "Пиши от себя, своим голосом, как будто рассказываешь в общий чат что было за день. "
        "Не цитируй записи дословно и не перечисляй их по одной. "
        "ВАЖНО: уложи весь текст ровно в 800 символов или меньше — это жёсткое ограничение чата. "
        "Если материала много — сжимай, выбирай главное, отбрасывай детали. "
        "Не обрывай мысль на полуслове: заверши последнее предложение целиком. "
        f"Начни строго с ника {username}."
    )
    if mood and mood in MOODS:
        compact_system_prompt += f" Твоё текущее настроение: {MOODS[mood]}"

    user_prompt = (
        "Записи из дневника планеты за сегодня:\n"
        f"{diary_lines}\n\n"
        "Напиши одну цельную живую сводку дня своим языком. "
        "Уложись в 800 символов — сжимай если нужно, но не обрывай на полуслове."
    )

    try:
        for attempt in range(3):
            resp = await ollama_chat(
                model=MODEL_DIARY_SUMMARY,
                messages=[
                    {"role": "system", "content": compact_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": 0.55 + attempt * 0.1,
                    "repeat_penalty": 1.3,
                    "max_tokens": 450,
                    "num_ctx": 8192,
                },
            )
            answer = _postprocess_answer(clean_response(resp["message"]["content"], username), username)
            answer_body = re.sub(rf'^\s*{re.escape(username)}\s*,\s*', '', answer, flags=re.IGNORECASE)
            if not answer_body.strip():
                continue
            if _looks_like_diary_dump(answer_body, entry_texts):
                print(f"[!дневник] попытка {attempt + 1}: ответ похож на дамп записей, перегенерирую")
                continue
            if len(answer) > 820:
                print(f"[!дневник] попытка {attempt + 1}: ответ {len(answer)} симв > 820, прошу сжать...")
                user_prompt = (
                    "Записи из дневника планеты за сегодня:\n"
                    f"{diary_lines}\n\n"
                    f"Напиши сводку дня. СТРОГО не более 800 символов — "
                    f"твой предыдущий ответ был {len(answer)} символов, это слишком много. "
                    "Сожми: убери детали, оставь главное, закончи мысль целиком."
                )
                continue
            return answer
    except Exception as e:
        print(f"Ошибка ollama (!дневник, model={MODEL_DIARY_SUMMARY}): {e}")

    fallback = _fallback_daily_diary_summary(entries, max_len=760)
    return _postprocess_answer(fallback or "за сегодня дневник пока пуст", username)


async def get_greeting(username: str, mood: str = "") -> str:
    style = random.choice(GREETING_STYLES)
    mood_hint = f"\nТЕКУЩЕЕ НАСТРОЕНИЕ: {MOODS[mood]}" if mood and mood in MOODS else ""
    is_default_user = _is_default_user_nick(username)
    try:
        if is_default_user:
            user_prompt = (
                f"Ник: {username}\n"
                f"В чат только что зашёл {username}. "
                f"Сделай короткое приветствие в виде забавной шутки ИМЕННО ПРО ЕГО НИК. "
                f"Смысл шутки: человек даже нормальный ник не придумал и оставил дефолтный user с цифрами. "
                f"Шути мягко-ехидно, без злобы. Можно обыграть, что это серийный номер, тестовый аккаунт, заводская настройка или временный файл. "
                f"Начни строго с ника {username}. "
                f"Максимум 1–2 предложения."
            )
        else:
            user_prompt = (
                f"Ник: {username}\n"
                f"В чат только что зашёл {username}. {style} "
                f"Но само приветствие построй как короткую забавную шутку или подколку ИМЕННО ПРО ЕГО НИК. "
                f"Нужно зацепиться за звучание, смысл, ассоциацию или известную отсылку из ника. "
                f"Пример вайба: если ник Шрёдингер — можно пошутить 'где кота потерял?'. "
                f"Шути легко, по-доброму и без злобы, будто это свой человек в чате. "
                f"Если в нике нет очевидной ассоциации, всё равно сделай мягкую шутку про то, как этот ник ощущается. "
                f"Начни строго с ника {username}. "
                f"Не спрашивай 'зачем пишешь' и не веди себя так, будто он навязался. "
                f"Лучше без вопроса или максимум с одним коротким вопросом. "
                f"Максимум 1–2 предложения."
            )

        resp = await ollama_chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": get_system_prompt(username, budget_tokens=PROMPT_BUDGET_LIGHT) + mood_hint},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": GREETING_TEMPERATURE, "repeat_penalty": 1.5, "num_ctx": 8192},
        )
        return clean_response(resp["message"]["content"], username)
    except Exception as e:
        print(f"Ошибка ollama (приветствие): {e}")
        if is_default_user:
            return random.choice(USER_NICK_GREETING_FALLBACKS).format(username=username)
        return random.choice(NICK_JOKE_GREETING_FALLBACKS).format(username=username)


# ── Форматирование репутации (для !репутация) ─────────────────────────

def format_history_summary(username: str) -> str:
    """Краткая сводка последних взаимодействий для команды !история."""
    from database import get_history

    history = get_history(username, limit=30)
    if not history:
        return f"{username}, у нас пока нет истории общения. Первый раз что ли?"

    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    bot_msgs  = [m["content"] for m in history if m["role"] == "assistant"]

    if not user_msgs:
        return f"{username}, ты ещё ничего мне не писал."

    lines = [f"{username}, вот что помню из нашего общения:"]

    # Последние 4 сообщения пользователя (без служебного префикса)
    clean_user = []
    for msg in user_msgs[-4:]:
        clean = re.sub(r'^Ник:\s*\S+\s*\n(?:Сообщение|Команда):\s*', '', msg).strip()
        if clean:
            clean_user.append(clean[:90] + ("…" if len(clean) > 90 else ""))

    if clean_user:
        lines.append("Твои последние сообщения:")
        lines.extend(f"  • {m}" for m in clean_user)

    # Последние 2 ответа бота
    if bot_msgs:
        lines.append("Мои последние ответы:")
        for msg in bot_msgs[-2:]:
            clean = msg.strip()[:90] + ("…" if len(msg.strip()) > 90 else "")
            lines.append(f"  • {clean}")

    # Статистика
    lines.append(f"Всего сообщений в истории: {len(history)}")

    return "\n".join(lines)


def format_forget_response(username: str, target: str = "") -> str:
    """Обрабатывает команду !забудь.

    target может быть:
      ''          — показать справку
      'профиль'   — сбросить user_profile
      'факты'     — удалить user_facts + user_traits
      'всё'       — профиль + факты + трейты
      'ник'       — удалить display_name из профиля
      любое слово — удалить факты содержащие это слово
    """
    from database import (
        clear_facts, clear_traits, clear_user_profile,
        delete_facts_containing, update_user_profile_field,
    )

    t = target.strip().lower()

    if not t:
        return (
            f"{username}, чего именно забыть? Варианты:\n"
            "• !забудь профиль — сброс имени, возраста, города\n"
            "• !забудь факты — удалить всё что помню о тебе\n"
            "• !забудь всё — полный сброс\n"
            "• !забудь [слово] — удалить конкретный факт"
        )

    if t in ("профиль", "profile"):
        clear_user_profile(username)
        reset_user_runtime_state(username)   # сбрасываем in-memory состояние
        return f"{username}, профиль стёрт. Буду знакомиться заново 🙄"

    if t in ("факты", "память", "факт"):
        clear_facts(username)
        clear_traits(username)
        reset_user_runtime_state(username)
        return f"{username}, всё что помнила — удалила. Чистый лист 🙂"

    if t in ("всё", "все", "all"):
        clear_user_profile(username)
        clear_facts(username)
        clear_traits(username)
        reset_user_runtime_state(username)
        return f"{username}, полный сброс. Ты для меня снова незнакомец 🤷‍♀️"

    if t in ("ник", "имя", "nickname", "name"):
        # Теперь работает: update_user_profile_field поддерживает None
        update_user_profile_field(username, "display_name", None)
        update_user_profile_field(username, "real_name", None)
        return f"{username}, имя забыла. Буду звать по нику."

    # Иначе — ищем и удаляем конкретный факт
    deleted = delete_facts_containing(username, target)
    if deleted:
        return f"{username}, нашла и удалила {deleted} факт(а) про «{target}»."
    return f"{username}, не нашла ничего про «{target}» в своей памяти."


def format_user_profile_info(username: str) -> str:
    """Форматирует полный профиль пользователя для команды !профиль.

    Показывает: структурированный профиль + факты + ярлыки + репутация.
    """
    from database import get_user_profile, get_facts, get_traits_prioritized
    from reputation import LEVELS, get_reputation, get_progress_cap

    parts = [f"{username}, вот что я о тебе знаю:"]

    # ── Структурированный профиль ─────────────────────────────
    profile = get_user_profile(username)
    profile_lines = []
    if profile.get("display_name"):
        profile_lines.append(f"Называю тебя: {profile['display_name']}")
    if profile.get("real_name"):
        profile_lines.append(f"Имя: {profile['real_name']}")
    if profile.get("gender"):
        g = "девушка" if profile["gender"] == "female" else "парень"
        profile_lines.append(f"Пол: {g}")
    if profile.get("age"):
        profile_lines.append(f"Возраст: {profile['age']}")
    if profile.get("city"):
        profile_lines.append(f"Город: {profile['city']}")
    if profile.get("interests"):
        interests = profile["interests"]
        if isinstance(interests, list):
            profile_lines.append(f"Интересы: {', '.join(interests[:4])}")

    if profile_lines:
        parts.append("Профиль:\n" + "\n".join(f"  {l}" for l in profile_lines))
    else:
        parts.append("Профиль: пока пустой — расскажи немного о себе 😏")

    # ── Факты ─────────────────────────────────────────────────
    facts = get_facts(username, limit=6)
    if facts:
        parts.append("Что помню:\n" + "\n".join(f"  • {f}" for f in facts))

    # ── Ярлыки ────────────────────────────────────────────────
    traits = get_traits_prioritized(username, limit=4)
    if traits:
        parts.append("Как ты обычно:\n" + "\n".join(f"  • {t}" for t in traits))

    # ── Репутация ─────────────────────────────────────────────
    rep = get_reputation(username)
    level = rep["level"]
    progress = rep["progress"]
    cap = get_progress_cap(level)
    lover_mark = " 💕" if rep["is_lover"] else ""
    parts.append(
        f"Отношения: {LEVELS[level]}{lover_mark} "
        f"({progress}/{cap})"
    )

    return "\n\n".join(parts)


def format_reputation_info(username: str) -> str:
    rep = get_reputation(username)
    level = rep["level"]
    progress = rep["progress"]
    cap = get_progress_cap(level)
    percent = int(round((progress / cap) * 100)) if cap > 0 else 0
    next_up = LEVELS[level - 1] if level > 0 else "—"
    next_down = LEVELS[level + 1] if level < 9 else "—"
    lover_mark = " 💕" if rep["is_lover"] else ""
    return (
        f"{username}{lover_mark}\n"
        f"Уровень: {LEVELS[level]}\n"
        f"Прогресс: {progress}/{cap} ({percent}%)\n"
        f"↑ {next_up} | ↓ {next_down}"
    )
