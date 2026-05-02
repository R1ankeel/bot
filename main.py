"""
main.py — точка входа: браузер, главный цикл, обработка команд.
"""

import asyncio
import collections
import os
import random
import re
import time
from datetime import datetime

from playwright.async_api import async_playwright

from config import (
    ADMIN_USER,
    AUTO_REPLY_TRIGGERS,
    CHARACTER_NAME,
    HISTORY_FILE,
    HOOK_COOLDOWN,
    HOOK_SILENCE,
    IGNORED_USERS,
    INJECTION_RESPONSES,
    LOG_NAME,
    LOOP_INTERVAL,
    MODE_NAMES,
    MOOD_CHANGE_INTERVAL,
    MODEL,
    MOODS,
    PRIVATE_TRIGGERS,
    SYSTEM_PROMPTS,
)
from database import (
    add_ignored,
    get_facts,
    get_global_message_count,
    get_global_messages_after,
    get_ignored_list,
    get_diary_entries_for_date,
    get_latest_diary_timestamp,
    get_recent_global_messages,
    get_traits,
    init_db,
    is_ignored,
    log_global_message,
    migrate_from_json,
    remove_ignored,
    seed_ignored,
)
from achievements import check_new_achievements
from daily_verdict import should_generate_today
from reputation import get_all_reputations

from analysis import _analyze_facts_background, _generate_diary_background
from browser import find_new_messages, parse_messages, send_response, send_text
from llm import ollama_chat
from prompt_builder import get_system_prompt
from responder import format_reputation_info, get_daily_diary_summary
from utils import (
    clean_response,
    detect_injection,
    has_name_trigger,
    pick_hook,
    looks_like_third_party_bot_mention,
    build_third_party_bot_mention_prompt,
)
from worker import _reply_worker
from jealousy import note_direct_interaction, note_direct_message_context, note_public_message, note_rival_message
import state


async def main():
    init_db()
    seed_ignored(IGNORED_USERS)
    state._last_diary_created_at = get_latest_diary_timestamp()

    if os.path.exists(HISTORY_FILE):
        migrate_from_json(HISTORY_FILE)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--window-size=1450,920"])
        page    = await browser.new_page()

        await page.goto("https://galaxy.mobstudio.ru/web/")
        print("Браузер открыт. Зайди на планету Альтушки.")
        print(f"Бот [{CHARACTER_NAME}] запущен.\n")

        last_seen_index: int | None = None
        last_seen_key: str | None = None
        processed_message_keys: collections.deque[str] = collections.deque(maxlen=300)
        processed_message_key_set: set[str] = set()

        def remember_processed_message_key(key: str):
            if not key or key in processed_message_key_set:
                return
            if len(processed_message_keys) >= processed_message_keys.maxlen:
                old_key = processed_message_keys.popleft()
                processed_message_key_set.discard(old_key)
            processed_message_keys.append(key)
            processed_message_key_set.add(key)

        last_chat_activity = time.time()
        last_hook_time     = 0.0
        last_group_comment_time = 0.0
        GROUP_COMMENT_COOLDOWN = 180  # не чаще одного встревания в 3 минуты
        greeted_users: set[str] = set()

        # Кулдаун команды !найдипару: не чаще 1 раза в час для каждого пользователя.
        find_pair_cooldown_seconds = 3600
        find_pair_last_used: dict[str, float] = {}

        current_mood   = random.choice(list(MOODS.keys()))
        last_mood_change = time.time()
        print(f"Начальное настроение: {current_mood}")
        print(f"Режим: {MODE_NAMES.get(state.current_mode, state.current_mode)}")

        state._reply_queue = asyncio.Queue()
        asyncio.create_task(_reply_worker(page))
        print("[ОЧЕРЕДЬ] Воркер запущен")

        try:
            while True:
                try:
                    try:
                        all_msgs = await asyncio.wait_for(parse_messages(page), timeout=8.0)
                    except asyncio.TimeoutError:
                        print("[ЦИКЛ] parse_messages подвис — пропускаю итерацию и пробую снова")
                        await page.wait_for_timeout(LOOP_INTERVAL)
                        continue

                    if all_msgs:
                        if last_seen_key is None:
                            for msg in all_msgs:
                                remember_processed_message_key(msg["key"])
                            last_seen_index = len(all_msgs) - 1
                            last_seen_key = all_msgs[last_seen_index]["key"]
                            new_msgs = []
                        else:
                            candidate_msgs, new_idx = find_new_messages(all_msgs, last_seen_index, last_seen_key)
                            last_seen_index = new_idx
                            last_seen_key = all_msgs[new_idx]["key"] if all_msgs else None

                            new_msgs = []
                            skipped_duplicates = 0
                            for msg in candidate_msgs:
                                msg_key = msg["key"]
                                if msg_key in processed_message_key_set:
                                    skipped_duplicates += 1
                                    continue
                                remember_processed_message_key(msg_key)
                                new_msgs.append(msg)

                            if skipped_duplicates:
                                print(f"[ДЕДУП] пропущено повторно увиденных сообщений: {skipped_duplicates}")

                        meaningful_new_msgs = [
                            msg for msg in new_msgs
                            if msg["username"].lower() != CHARACTER_NAME.lower() and not is_ignored(msg["username"])
                        ]
                        if meaningful_new_msgs:
                            last_chat_activity = time.time()

                        for msg in new_msgs:
                            username = msg["username"]
                            text = msg["text"]
                            if username.lower() == CHARACTER_NAME.lower():
                                continue
                            if is_ignored(username):
                                continue

                            log_global_message(username, text)
                            state.buffer_add(username, text)
                            state.track_group_interaction(username, text)
                            if username != "[ДЕЙСТВИЕ]":
                                check_new_achievements(username)

                            text_lower = text.lower()

                            # ── Авто-ответы на конкретные фразы ────────────────
                            _auto_replied = False
                            if username != "[ДЕЙСТВИЕ]":
                                for _pattern, _replies in AUTO_REPLY_TRIGGERS:
                                    if _pattern in text_lower:
                                        _reply = random.choice(_replies)
                                        print(f"[АВТО] '{_pattern}' от {username} → {_reply[:60]}")
                                        await state._reply_queue.put({"type": "text", "text": _reply})
                                        _auto_replied = True
                                        break  # один ответ на сообщение

                            if _auto_replied:
                                continue
                            # усиливаем контекст даже без его собственного сообщения.
                            if username != "[ДЕЙСТВИЕ]":
                                rival_payload = note_rival_message(username, text)
                                if rival_payload:
                                    print(f"[РЕВНОСТЬ] соперник {username} флиртует с любимчиком → очередь")
                                    await state._reply_queue.put({
                                        "type": "jealousy_rival",
                                        "username": username,
                                        "payload": rival_payload,
                                        "mood": current_mood,
                                    })

                            # ── Ревность: учитываем публичное сообщение любимчика ──
                            # ВАЖНО: если любимчик пишет самой Аньке или использует команду,
                            # это не должно считаться "общением с кем-то другим".
                            if username != "[ДЕЙСТВИЕ]" and not has_name_trigger(text) and not text_lower.startswith("!"):
                                jealousy_payload = note_public_message(username, text)
                                if jealousy_payload:
                                    print(f"[РЕВНОСТЬ] порог достигнут для {username}, stage={jealousy_payload['stage']}")
                                    await state._reply_queue.put({
                                        "type": "jealousy",
                                        "username": username,
                                        "payload": jealousy_payload,
                                        "mood": current_mood,
                                    })

                            # ── Действие без ника ──────────────────────────────
                            if username == "[ДЕЙСТВИЕ]":
                                if has_name_trigger(text):
                                    actor = text.split()[0].lstrip("^") if text.split() else "кто-то"
                                    if is_ignored(actor):
                                        print(f"[ИГНОР] действие от {actor} пропущено")
                                    else:
                                        print(f"Действие с ботом от [{actor}]: {text} → в очередь")
                                        action_lower = text.lower()
                                        extra_action_guard = ""
                                        if any(
                                            x in action_lower
                                            for x in (
                                                "плюнул",
                                                "плюнула",
                                                "плюёт",
                                                "плюет",
                                                "плюнула в лицо",
                                                "плюнул в лицо",
                                            )
                                        ):
                                            extra_action_guard = (
                                                "\n\nЭто уже повторяющееся действие с плевком. "
                                                "Не отвечай про слюну, плевки, мокроту, червей, слизней, пиявок, насекомых и грязные ёмкости. "
                                                "Реагируй на унизительность поступка: презрение, брезгливость, злость, сарказм. "
                                                "Лучше обыграй, что человек пытается казаться опасной, но выглядит жалко."
                                            )
                                        action_prompt = (
                                            f"[ДЕЙСТВИЕ В ЧАТЕ]\n"
                                            f"Пользователь {actor} совершил действие: {text}\n\n"
                                            "Отреагируй как Аня на само действие, а не описывай физиологию действия.\n"
                                            "Если действие грубое или унизительное — ответь зло, колко и коротко.\n"
                                            "ВАЖНО: не строй ответ по шаблону «Твоя слюна/плевки/мокрота — ... для ...».\n"
                                            "Не используй слова: слюна, плевки, мокрота, слизни, пиявки, блохи, черви, моль, помойка, бассейн, аквариум.\n"
                                            "Не начинай с порядковых слов: первая, вторая, третья, четвёртая, пятая, шестая, седьмая, восьмая.\n"
                                            "Сделай реакцию другой формы: сухое презрение, угроза пожаловаться смотрителю, саркастичный диагноз, "
                                            "унижение её попытки выглядеть дерзко, или холодный отшив."
                                            f"{extra_action_guard}"
                                        )
                                        await state.enqueue_llm_job(actor, action_prompt, current_mood)
                                continue

                            # ══ Админ-команды ══════════════════════════════════

                            if username == ADMIN_USER and text_lower.startswith("!настроение"):
                                match = re.search(r'(\d)', text)
                                if match:
                                    num = int(match.group(1))
                                    mood_list = list(MOODS.keys())
                                    if 1 <= num <= len(mood_list):
                                        current_mood = mood_list[num - 1]
                                        last_mood_change = time.time()
                                        print(f"[АДМИН] Настроение: {current_mood}")
                                        await send_text(page, f"{ADMIN_USER}, ладно... настроение: {current_mood} 😏")
                                    else:
                                        await send_text(page, f"{ADMIN_USER}, от 1 до {len(mood_list)} 😏")
                                continue

                            if username == ADMIN_USER and text_lower.startswith("!режим"):
                                match = re.search(r'(\d)', text)
                                if match:
                                    num = int(match.group(1))
                                    if num in SYSTEM_PROMPTS:
                                        state.set_mode(num)
                                        name = MODE_NAMES.get(num, str(num))
                                        print(f"[АДМИН] Режим: {name}")
                                        await send_text(page, f"{ADMIN_USER}, режим: {name} 😏")
                                    else:
                                        await send_text(page, f"{ADMIN_USER}, есть только {', '.join(f'{k} — {v}' for k, v in MODE_NAMES.items())} 😏")
                                continue

                            if username == ADMIN_USER and text_lower.startswith("!репутация"):
                                target = text[len("!репутация"):].strip()
                                if target:
                                    await send_response(page, format_reputation_info(target))
                                else:
                                    all_reps = get_all_reputations()
                                    if all_reps:
                                        lines = [
                                            f"{r['username']}{'💕' if r['is_lover'] else ''}: {r['level_name']} ({r['progress']*10}%)"
                                            for r in all_reps
                                        ]
                                        await send_response(page, "\n".join(lines))
                                    else:
                                        await send_text(page, f"{ADMIN_USER}, пока ни с кем не общалась 🤷‍♀️")
                                continue

                            if text_lower.startswith("!факты"):
                                facts = get_facts(username, limit=15)
                                if facts:
                                    await state._reply_queue.put({
                                        "type": "text",
                                        "text": f"{username}, вот что я о тебе знаю:\n" + "\n".join(f"• {f}" for f in facts),
                                    })
                                else:
                                    await state._reply_queue.put({
                                        "type": "text",
                                        "text": f"{username}, пока ничего интересного о тебе не накопила 🤷‍♀️",
                                    })
                                continue

                            if username == ADMIN_USER and text_lower.startswith("!ярлыки"):
                                target = text[len("!ярлыки"):].strip()
                                if target:
                                    traits = get_traits(target, limit=15)
                                    if traits:
                                        await send_response(page, f"{target}:\n" + "\n".join(f"• {t}" for t in traits))
                                    else:
                                        await send_text(page, f"{ADMIN_USER}, у {target} нет ярлыков 🤷‍♀️")
                                else:
                                    await send_text(page, f"{ADMIN_USER}, напиши !ярлыки Ник")
                                continue

                            if username == ADMIN_USER and text_lower.startswith("!дневник"):
                                today_key = datetime.now().strftime("%Y-%m-%d")
                                entries = get_diary_entries_for_date(today_key, limit=500)
                                if entries:
                                    digest = await get_daily_diary_summary(username, entries, current_mood)
                                    await send_response(page, digest)
                                else:
                                    await send_text(page, f"{ADMIN_USER}, за сегодня дневник пока пуст")
                                continue

                            if username == ADMIN_USER and text_lower.startswith("!игнорируй"):
                                target = text[len("!игнорируй"):].strip()
                                if target:
                                    if add_ignored(target):
                                        await send_text(page, f"{ADMIN_USER}, ок, {target} в игноре 🙈")
                                        print(f"[АДМИН] {target} добавлен в игнор")
                                    else:
                                        await send_text(page, f"{ADMIN_USER}, {target} уже в игноре")
                                else:
                                    ignored = get_ignored_list()
                                    await send_text(page, f"{ADMIN_USER}, в игноре: {', '.join(ignored)}" if ignored else f"{ADMIN_USER}, список игнора пуст")
                                continue

                            if username == ADMIN_USER and text_lower.startswith("!общайся"):
                                target = text[len("!общайся"):].strip()
                                if target:
                                    if remove_ignored(target):
                                        await send_text(page, f"{ADMIN_USER}, {target} убран из игнора 👌")
                                        print(f"[АДМИН] {target} убран из игнора")
                                    else:
                                        await send_text(page, f"{ADMIN_USER}, {target} и так не в игноре")
                                else:
                                    await send_text(page, f"{ADMIN_USER}, напиши !общайся Ник")
                                continue

                            # ══ Пользовательские команды ═══════════════════════

                            if text_lower.startswith("!утешить"):
                                target = text[len("!утешить"):].strip()
                                if target:
                                    print(f"[!утешить] от [{username}] для {target} → очередь")
                                    await state._reply_queue.put({"type": "comfort", "username": username, "target": target})
                                continue

                            if text_lower.startswith("!заагрись"):
                                if username != ADMIN_USER:
                                    await state._reply_queue.put({"type": "text", "text": f"{username}, эта команда только для админа"})
                                    continue
                                target = text[len("!заагрись"):].strip()
                                if target:
                                    print(f"[!заагрись] от [{username}] на {target} → очередь")
                                    await state._reply_queue.put({"type": "roast", "username": username, "target": target})
                                continue

                            if text_lower.startswith("!найдипару"):
                                cooldown_key = username.lower().strip()
                                now_ts = time.time()
                                last_used = find_pair_last_used.get(cooldown_key, 0.0)
                                cooldown_left = find_pair_cooldown_seconds - (now_ts - last_used)

                                if cooldown_left > 0:
                                    minutes_left = max(1, int((cooldown_left + 59) // 60))
                                    print(f"[!найдипару] от [{username}] на кулдауне: ещё {minutes_left} мин")
                                    await state._reply_queue.put({
                                        "type": "text",
                                        "text": (
                                            f"{username}, я тебе уже искала пару. "
                                            f"Подожди ещё примерно {minutes_left} мин., а то сваха перегреется 😏"
                                        ),
                                    })
                                    continue

                                find_pair_last_used[cooldown_key] = now_ts
                                print(f"[!найдипару] от [{username}] → очередь")
                                await state._reply_queue.put({
                                    "type": "find_pair",
                                    "username": username,
                                    "mood": current_mood,
                                })
                                continue

                            if text_lower.startswith("!найди"):
                                query = text[len("!найди"):].strip()
                                if query:
                                    print(f"[!найди] от [{username}]: {query} → очередь")
                                    await state._reply_queue.put({"type": "search", "username": username, "query": query, "mood": current_mood})
                                continue

                            if text_lower.startswith("!мнение"):
                                print(f"[!мнение] от [{username}] → очередь")
                                await state._reply_queue.put({"type": "opinion", "username": username, "mood": current_mood})
                                continue


                            if text_lower.startswith("!гороскоп"):
                                sign_raw = text[len("!гороскоп"):].strip()
                                if sign_raw:
                                    print(f"[!гороскоп] от [{username}]: {sign_raw} → очередь")
                                    await state._reply_queue.put({
                                        "type": "horoscope",
                                        "username": username,
                                        "sign": sign_raw,
                                        "mood": current_mood,
                                    })
                                else:
                                    await state._reply_queue.put({
                                        "type": "text",
                                        "text": f"{username}, укажи знак зодиака: !гороскоп овен",
                                    })
                                continue

                            if text_lower.startswith("!совместимость"):
                                rest = text[len("!совместимость"):].strip()
                                parts = [p for p in re.split(r"[\s,]+", rest) if p]
                                if len(parts) >= 2:
                                    nick1, nick2 = parts[0], parts[1]
                                    print(f"[!совместимость] от [{username}] для {nick1} и {nick2} → очередь")
                                    await state._reply_queue.put({
                                        "type": "compatibility",
                                        "username": username,
                                        "nick1": nick1,
                                        "nick2": nick2,
                                        "mood": current_mood,
                                    })
                                else:
                                    await state._reply_queue.put({
                                        "type": "text",
                                        "text": f"{username}, формат такой: !совместимость Ник1 Ник2",
                                    })
                                continue

                            if text_lower.startswith("!профиль") or text_lower.startswith("!profile"):
                                print(f"[!профиль] от [{username}] → очередь")
                                await state._reply_queue.put({"type": "profile_view", "username": username})
                                continue

                            if text_lower.startswith("!забудь") or text_lower.startswith("!forget"):
                                if username != ADMIN_USER:
                                    print(f"[!забудь] от [{username}] — не админ, игнорирую")
                                    continue
                                raw_cmd = text.strip()
                                parts_cmd = raw_cmd.split(None, 1)
                                forget_target = parts_cmd[1].strip() if len(parts_cmd) > 1 else ""
                                print(f"[!забудь] от [{username}]: {forget_target!r} → очередь")
                                await state._reply_queue.put({
                                    "type": "forget",
                                    "username": username,
                                    "target": forget_target,
                                })
                                continue

                            if text_lower.startswith("!история") or text_lower.startswith("!history"):
                                if username != ADMIN_USER:
                                    print(f"[!история] от [{username}] — не админ, игнорирую")
                                    continue
                                print(f"[!история] от [{username}] → очередь")
                                await state._reply_queue.put({"type": "history_view", "username": username})
                                continue

                            if text_lower.startswith("!достижения"):
                                print(f"[!достижения] от [{username}] → очередь")
                                await state._reply_queue.put({"type": "achievements_view", "username": username})
                                continue

                            # ══ Триггеры имени ══════════════════════════════════

                            if has_name_trigger(text) and any(t in text_lower for t in PRIVATE_TRIGGERS):
                                note_direct_interaction(username)
                                note_direct_message_context(username, text)
                                print(f"Триггер личка от [{username}] → очередь/склейка")
                                await state.enqueue_llm_job(username, text, current_mood)
                                continue

                            if has_name_trigger(text):
                                note_direct_interaction(username)
                                note_direct_message_context(username, text)
                                if detect_injection(text):
                                    reply = f"{username}, {random.choice(INJECTION_RESPONSES)}"
                                    print(f"[ИНЪЕКЦИЯ] от [{username}] → очередь")
                                    await state._reply_queue.put({"type": "text", "text": reply})
                                    continue

                                if looks_like_third_party_bot_mention(text):
                                    # Отправляем в LLM, но в специальном SELF_IDENTITY-режиме.
                                    # responder в этом режиме отключит сплетничий target_user,
                                    # mentioned_people_block и recent_chat_block, чтобы модель
                                    # не схватила случайного человека из соседних реплик.
                                    text_for_llm = build_third_party_bot_mention_prompt(username, text)
                                    print(f"Стороннее упоминание бота от [{username}] → очередь/склейка SELF_IDENTITY")
                                    await state.enqueue_llm_job(username, text_for_llm, current_mood)
                                    continue

                                print(f"Упоминание от [{username}] → очередь/склейка")
                                await state.enqueue_llm_job(username, text, current_mood)
                                continue

                        # ══ Редкий боковой комментарий в общий чат ═══════════════
                        if meaningful_new_msgs and state._reply_queue.empty():
                            now_ts = time.time()

                            recent_group_msgs = [
                                {"username": m.get("username", ""), "text": m.get("text", "")}
                                for m in list(state._global_buffer)[-12:]
                                if m.get("username")
                                and m.get("username") != "[ДЕЙСТВИЕ]"
                                and m.get("username", "").lower() != CHARACTER_NAME.lower()
                                and not is_ignored(m.get("username", ""))
                                and (m.get("text") or "").strip()
                            ]

                            recent_authors = {
                                m["username"].lower()
                                for m in recent_group_msgs[-8:]
                                if m.get("username")
                            }

                            # Встреваем только если это реально мини-диалог, а не одно случайное сообщение.
                            should_try_group_comment = (
                                len(recent_group_msgs) >= 4
                                and len(recent_authors) >= 2
                                and now_ts - last_group_comment_time >= GROUP_COMMENT_COOLDOWN
                            )

                            if should_try_group_comment and random.random() < random.uniform(0.08, 0.12):
                                print("[ОЧЕРЕДЬ] group_comment → очередь")
                                last_group_comment_time = now_ts
                                await state._reply_queue.put({
                                    "type": "group_comment",
                                    "mood": current_mood,
                                    "recent_messages": recent_group_msgs,
                                })

                    # ══ Смена настроения ════════════════════════════════════════
                    if time.time() - last_mood_change >= MOOD_CHANGE_INTERVAL:
                        other_moods = [m for m in MOODS if m != current_mood]
                        current_mood = random.choice(other_moods)
                        last_mood_change = time.time()
                        print(f"Настроение изменилось: {current_mood}")

                    # ══ Приветствие входящих ════════════════════════════════════
                    try:
                        fly_els = await asyncio.wait_for(
                            page.query_selector_all('div.channel-fly'),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        print("[ЦИКЛ] чтение channel-fly подвисло — пропускаю этот шаг")
                        fly_els = []
                    for fly in fly_els:
                        try:
                            # icon--join = вход в комнату, icon--leave = выход
                            icon_el = await fly.query_selector('img.icon--join')
                            if not icon_el:
                                continue  # это выход — игнорируем
                            text_el = await fly.query_selector('div.channel-fly__text')
                            if not text_el:
                                continue
                            joined = (await text_el.inner_text()).strip()
                            if joined and joined not in greeted_users and joined.lower() != CHARACTER_NAME.lower():
                                greeted_users.add(joined)
                                if random.random() < 0.25:
                                    print(f"Вошёл: {joined} → очередь на приветствие")
                                    await state._reply_queue.put({"type": "greeting", "username": joined, "mood": current_mood})
                                else:
                                    print(f"Вошёл: {joined} — пропускаю")
                        except Exception:
                            continue

                    # ══ Ежедневный вердикт в 22:00 ═══════════════════════════════
                    due, report_date = should_generate_today(datetime.now())
                    if due and state._daily_verdict_queued_for != report_date:
                        state._daily_verdict_queued_for = report_date
                        print(f"[ВЕРДИКТ ДНЯ] Планирую отправку за {report_date}")
                        await state._reply_queue.put({"type": "daily_verdict", "report_date": report_date})

                    # ══ Хук при тишине ══════════════════════════════════════════
                    now = time.time()
                    if (now - last_chat_activity >= HOOK_SILENCE
                            and now - last_hook_time >= HOOK_COOLDOWN
                            and state._reply_queue.empty()):
                        last_hook_time     = now
                        last_chat_activity = now
                        try:
                            state.update_activity()
                            mood_desc = MOODS.get(current_mood, "")
                            hook_resp = await ollama_chat(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": get_system_prompt()},
                                    {"role": "user", "content": (
                                        f"В чате давно тишина. Твоё настроение: {mood_desc}. "
                                        f"Ты сейчас: {state._current_activity}. "
                                        "Напиши одно короткое сообщение в общий чат, чтобы оживить разговор. "
                                        "Без ника в начале. Максимум 1 предложение."
                                    )},
                                ],
                                options={"temperature": 0.95, "num_ctx": 8192},
                            )
                            hook = clean_response(hook_resp['message']['content'])
                        except Exception as e:
                            print(f"Ошибка генерации хука: {e}")
                            hook = pick_hook(current_mood)
                        print(f"{LOG_NAME} хук ({current_mood}): {hook}\n")
                        await send_text(page, hook)

                    # ══ Фоновый анализ фактов ═══════════════════════════════════
                    unanalyzed = get_global_message_count()
                    if (unanalyzed >= state.FACTS_ANALYZE_EVERY
                            and time.time() - state._last_facts_analysis >= state.FACTS_ANALYZE_INTERVAL
                            and not state._facts_task_running):
                        print(f"[ФАКТЫ] Запускаю анализ ({unanalyzed} необработанных)...")
                        asyncio.create_task(_analyze_facts_background())

                    # ══ Дневник планеты ══════════════════════════════════════════
                    diary_candidates = (
                        get_global_messages_after(state._last_diary_created_at, limit=120)
                        if state._last_diary_created_at
                        else get_recent_global_messages(limit=80)
                    )
                    if (len(diary_candidates) >= state.DIARY_MIN_MESSAGES
                            and not state._diary_task_running):
                        print(f"[ДНЕВНИК] {len(diary_candidates)} сообщений → запускаю...")
                        asyncio.create_task(_generate_diary_background())

                    await page.wait_for_timeout(LOOP_INTERVAL)

                except Exception as e:
                    if "TargetClosed" in str(e):
                        print("Браузер закрыт.")
                        break
                    print(f"Ошибка в цикле: {e}")
                    await page.wait_for_timeout(3000)

        finally:
            try:
                await browser.close()
            except Exception:
                pass


asyncio.run(main())
