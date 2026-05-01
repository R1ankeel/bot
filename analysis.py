"""
analysis.py — фоновый LLM-анализ:
извлечение фактов/ярлыков, дневник планеты, репутация.
"""

import asyncio
import random
import re
import time

from config import (
    CHARACTER_NAME, LEVEL_REACTION_TEMPERATURE, MODEL, MODEL_ANALYSIS,
    FACTS_EXTRACTION_PROMPT,
    FACTS_VALIDATION_PROMPT,
    PLANET_DIARY_PROMPT,
    TRAITS_EXTRACTION_PROMPT,
)
from database import (
    add_diary_entry,
    add_facts_bulk,
    add_traits_bulk,
    deduplicate_age_facts,
    deduplicate_city_facts,
    clean_conflicting_gender_facts,
    extract_and_update_profile_from_facts,
    get_last_user_messages,
    get_recent_global_messages,
    get_global_messages_after,
    get_unanalyzed_messages,
    mark_messages_analyzed,
)
from llm import ollama_analyze, ollama_chat
from prompt_builder import get_system_prompt
from reputation import (
    LEVELS,
    SENTIMENT_ANALYSIS_PROMPT,
    apply_reputation_delta,
    apply_spam_fatigue_to_delta,
    calculate_spam_fatigue,
    get_level_change_reaction,
    get_reputation,
)
from utils import clean_response, is_valid_trait, looks_like_raw_chat_fragment, parse_fact_lines
import state
from achievements import build_achievement_notification, check_new_achievements


# ── Извлечение фактов и ярлыков ───────────────────────────────────────

async def analyze_facts_from_chat():
    """
    Берёт необработанные сообщения, прогоняет двухэтапное LLM-извлечение
    фактов и ярлыков, сохраняет в БД.
    """
    messages = get_unanalyzed_messages(limit=50)
    if not messages:
        return

    msg_lines = [f"[{m['username']}]: {m['text']}" for m in messages]
    msg_ids   = [m["id"] for m in messages]
    msg_text  = "\n".join(msg_lines)

    try:
        # Шаг 1 — извлечение кандидатов
        resp = await ollama_analyze(
            messages=[{"role": "user", "content": FACTS_EXTRACTION_PROMPT.format(messages=msg_text)}],
            options={"temperature": 0.2},
        )
        raw_candidates = resp["message"]["content"].strip()
        print(f"[ФАКТЫ] Шаг1 сырой ответ: {raw_candidates[:200]!r}")

        facts: list[tuple[str, str]] = []
        candidates = parse_fact_lines(raw_candidates) if raw_candidates and raw_candidates.upper() != "НЕТ" else []
        print(f"[ФАКТЫ] Шаг1 кандидатов: {len(candidates)} → {candidates[:5]}")

        # Шаг 2 — валидация
        if candidates:
            candidate_text = "\n".join(f"{nick}: {fact}" for nick, fact in candidates)
            resp2 = await ollama_analyze(
                messages=[{"role": "user", "content": FACTS_VALIDATION_PROMPT.format(
                    messages=msg_text, facts=candidate_text
                )}],
                options={"temperature": 0.1},
            )
            raw_validated = resp2["message"]["content"].strip()
            print(f"[ФАКТЫ] Шаг2 сырой ответ: {raw_validated[:200]!r}")
            if raw_validated and raw_validated.upper() != "НЕТ":
                parsed = parse_fact_lines(raw_validated)
                facts = [
                    (nick, fact) for nick, fact in parsed
                    if not looks_like_raw_chat_fragment(fact, messages, threshold=0.95)
                ]
                filtered_out = len(parsed) - len(facts)
                if filtered_out:
                    print(f"[ФАКТЫ] Шаг2 отфильтровано looks_like: {filtered_out}")

        # Фильтруем факты о полях профиля если они уже зафиксированы в user_profile.
        # Без этого LLM цепляется за любую фразу в чате и перезаписывает пол/имя в таблице фактов,
        # что потом ломает build_gender_block несмотря на guard в extract_and_update_profile.
        from database import get_user_profile as _get_profile
        _name_fact_re = re.compile(
            r'(?:имя\s*[:：]|меня зовут|моё имя|мое имя)',
            re.IGNORECASE,
        )
        facts_filtered: list[tuple[str, str]] = []
        for _nick, _fact in facts:
            _prof = _get_profile(_nick)
            # Гендерный факт — пропускаем если пол уже зафиксирован
            if "пол:" in _fact.lower():
                _existing_gender = _prof.get("gender", "")
                if _existing_gender and _existing_gender not in ("unknown", ""):
                    print(f"[ФАКТЫ] Пропущен гендерный факт для {_nick} (профиль: {_existing_gender}): {_fact!r}")
                    continue
            # Факт об имени — пропускаем если имя уже зафиксировано
            if _name_fact_re.search(_fact):
                if _prof.get("real_name") or _prof.get("display_name"):
                    print(f"[ФАКТЫ] Пропущен имя-факт для {_nick} (профиль уже знает имя): {_fact!r}")
                    continue
            facts_filtered.append((_nick, _fact))
        facts = facts_filtered

        added_facts = add_facts_bulk(facts, source="chat") if facts else 0

        # Ярлыки — НЕ критичны. Если третий LLM-вызов или сохранение ярлыков упали,
        # факты всё равно уже можно сохранить, а batch нужно пометить обработанным.
        # Иначе одни и те же 50 сообщений будут бесконечно гоняться заново.
        traits: list[tuple[str, str]] = []
        self_declared: list[tuple[str, str]] = []
        observed: list[tuple[str, str]] = []
        added_sd = 0
        added_ob = 0

        try:
            resp3 = await ollama_analyze(
                messages=[{"role": "user", "content": TRAITS_EXTRACTION_PROMPT.format(messages=msg_text)}],
                options={"temperature": 0.2, "max_tokens": 600},
            )
            raw_traits = resp3["message"]["content"].strip()
            print(f"[ФАКТЫ] Ярлыки сырой ответ: {raw_traits[:200]!r}")

            if raw_traits and raw_traits.upper() != "НЕТ":
                traits = [
                    (nick, trait.strip())
                    for nick, trait in parse_fact_lines(raw_traits)
                    if is_valid_trait(trait) and not looks_like_raw_chat_fragment(trait, messages)
                ]

            # Разделяем ярлыки: self_declared (явное самоописание) vs observed (наблюдение бота)
            _self_markers = ("я ", "меня ", "мне ", "сам ", "сама ", "говорю", "считаю", "люблю", "хочу")
            for nick, trait in traits:
                is_self = any(
                    m["username"].lower() == nick.lower()
                    and any(marker in m["text"].lower() for marker in _self_markers)
                    for m in messages
                )
                if is_self:
                    self_declared.append((nick, trait))
                else:
                    observed.append((nick, trait))

            try:
                added_sd = add_traits_bulk(self_declared, source="self_declared") if self_declared else 0
                added_ob = add_traits_bulk(observed, source="observed") if observed else 0
            except Exception as te:
                print(f"[ФАКТЫ] Ошибка сохранения ярлыков, пропускаю ярлыки: {type(te).__name__}: {te!r}")
                self_declared = []
                observed = []
                traits = []
                added_sd = 0
                added_ob = 0

        except Exception as te:
            print(f"[ФАКТЫ] Ошибка анализа ярлыков, пропускаю ярлыки: {type(te).__name__}: {te!r}")
            traits = []
            self_declared = []
            observed = []
            added_sd = 0
            added_ob = 0

        added_traits = added_sd + added_ob

        print(
            f"[ФАКТЫ] Проанализировано {len(messages)} сообщений "
            f"→ фактов: {len(facts)} (новых {added_facts}) "
            f"→ ярлыков: {len(traits)} (self={added_sd} observed={added_ob})"
        )
        for nick, fact  in facts:  print(f"  • факт  {nick}: {fact}")
        for nick, trait in self_declared: print(f"  • ярлык [self] {nick}: {trait}")
        for nick, trait in observed:     print(f"  • ярлык [obs]  {nick}: {trait}")

        mark_messages_analyzed(msg_ids)
        state._last_facts_analysis = time.time()

        # Обновляем структурированный профиль из накопленных фактов
        # Делаем это для всех уников кому добавили хоть что-то
        unique_users = {nick for nick, _ in facts} | {nick for nick, _ in traits}
        for nick in unique_users:
            try:
                extract_and_update_profile_from_facts(nick)
                # Убираем конфликтующие старые факты о возрасте, городе и профиль-полях
                deduplicate_age_facts(nick)
                deduplicate_city_facts(nick)
                clean_conflicting_gender_facts(nick)
                from database import clean_profile_redundant_facts
                clean_profile_redundant_facts(nick)
            except Exception as pe:
                print(f"[ПРОФИЛЬ] Ошибка обновления профиля {nick}: {pe}")

    except Exception as e:
        print(f"Ошибка анализа фактов: {type(e).__name__}: {e!r}")


async def _analyze_facts_background():
    """Фоновый анализ фактов — с API запускается напрямую без ожидания."""
    state._facts_task_running = True
    try:
        print("[ФАКТЫ] Запускаю фоновый анализ...")
        await analyze_facts_from_chat()
    except Exception as e:
        print(f"[ФАКТЫ] Ошибка фонового анализа: {e}")
    finally:
        state._facts_task_running = False


# ── Дневник планеты ───────────────────────────────────────────────────

async def generate_planet_diary_entry() -> bool:
    recent = (
        get_global_messages_after(state._last_diary_created_at, limit=120)
        if state._last_diary_created_at
        else get_recent_global_messages(limit=80)
    )
    if len(recent) < state.DIARY_MIN_MESSAGES:
        return False

    msg_text = "\n".join(f"[{m['username']}]: {m['text']}" for m in recent)
    try:
        resp = await ollama_analyze(
            messages=[{"role": "user", "content": PLANET_DIARY_PROMPT.format(messages=msg_text)}],
            options={"temperature": 0.6},
        )
        entry = clean_response(resp["message"]["content"])
        if not entry:
            return False
        add_diary_entry(entry, source="chat")
        state._last_diary_created_at = recent[-1]["created_at"]
        print(f"[ДНЕВНИК] Новая запись: {entry[:120]}")
        return True
    except Exception as e:
        print(f"Ошибка генерации дневника: {e}")
        return False


async def _generate_diary_background():
    """Фоновая запись дневника — с API запускается напрямую."""
    state._diary_task_running = True
    try:
        print("[ДНЕВНИК] Запускаю фоновую запись...")
        await generate_planet_diary_entry()
    except Exception as e:
        print(f"[ДНЕВНИК] Ошибка фоновой записи: {e}")
    finally:
        state._diary_task_running = False


# ── Репутация ─────────────────────────────────────────────────────────

async def evaluate_reputation(username: str):
    messages = get_last_user_messages(username, limit=10)
    if not messages:
        return None

    rep = get_reputation(username)
    prompt = SENTIMENT_ANALYSIS_PROMPT.format(
        char_name=CHARACTER_NAME,
        current_level=LEVELS[rep["level"]],
        username=username,
        messages="\n".join(f"- {m}" for m in messages),
    )

    try:
        resp = await ollama_analyze(
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        raw = resp["message"]["content"].strip()
        match = re.search(r'([+-]?[0-3])', raw)
        delta = max(-3, min(3, int(match.group(1)))) if match else 0

        # Спам-усталость: если юзер спамит одним и тем же — режем или инвертируем дельту
        fatigue = calculate_spam_fatigue(messages)
        if fatigue > 0:
            original_delta = delta
            delta = apply_spam_fatigue_to_delta(delta, fatigue)
            from reputation import _FATIGUE_EFFECTS
            _, fatigue_desc = _FATIGUE_EFFECTS.get(fatigue, (0, ""))
            print(f"[РЕПУТАЦИЯ] {username}: спам-усталость {fatigue} ({fatigue_desc}), дельта {original_delta:+d} → {delta:+d}")
            # Обновляем спам-состояние в state для real-time хинта в промпте
            state.update_spam_fatigue(username, fatigue)
        else:
            state.update_spam_fatigue(username, 0)

        if delta != 0:
            change = apply_reputation_delta(username, delta, reason=f"оценка реакции по {len(messages)} сообщениям (усталость={fatigue})")
            reaction = get_level_change_reaction(change)
            if reaction:
                print(f"[РЕПУТАЦИЯ] Реакция: {reaction[:80]}")
                return reaction
        else:
            print(f"[РЕПУТАЦИЯ] {username}: нейтрально, без изменений")

    except Exception as e:
        print(f"Ошибка анализа репутации: {e}")

    return None


async def evaluate_reputation_background(username: str, page):
    """Фоновый анализ репутации. Реакция на смену уровня идёт через очередь."""
    try:
        print(f"[РЕПУТАЦИЯ] Фоновый анализ {username}...")
        reaction = await evaluate_reputation(username)

        if reaction:
            react_resp = await ollama_chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": get_system_prompt(username)},
                    {"role": "user", "content": (
                        f"Ник: {username}\n"
                        f"[ВНУТРЕННЕЕ СОБЫТИЕ] {reaction}\n"
                        f"Напиши короткую реакцию (1 предложение). Начни с ника {username}."
                    )},
                ],
                options={"temperature": LEVEL_REACTION_TEMPERATURE, "num_ctx": 8192},
            )
            react_text = clean_response(react_resp["message"]["content"], username)
            print(f"  [РЕАКЦИЯ] {react_text}")
            # Кладём в очередь чтобы соблюдать кулдаун платформы
            await state._reply_queue.put({"type": "text", "text": react_text})

        unlocked = check_new_achievements(username)
        notice = build_achievement_notification(username, unlocked)
        if notice:
            print(f"  [АЧИВКА] {notice}")
            await state._reply_queue.put({"type": "text", "text": notice})

    except Exception as e:
        print(f"[РЕПУТАЦИЯ] Ошибка фонового анализа: {e}")
