"""
worker.py — воркер очереди ответов.
Берёт задачи из _reply_queue и выполняет их по типу.
"""

import asyncio
import random
import re

from config import (
    CHARACTER_NAME,
    COMFORT_TEMPERATURE,
    JEALOUSY_TEMPERATURE,
    LOG_NAME,
    MODEL,
    MOODS,
    RESPONSE_COOLDOWN,
    ROAST_TEMPERATURE,
)
from database import save_message
from llm import ollama_chat
from browser import send_response, send_text
from prompt_builder import build_recent_chat_block, get_system_prompt, get_jealousy_system_prompt
from responder import (
    get_compatibility_response,
    get_find_pair_response,
    get_forced_search_response,
    get_greeting,
    get_horoscope_response,
    get_response,
    get_self_opinion_response,
    format_user_profile_info,
    format_forget_response,
    format_history_summary,
)
from analysis import evaluate_reputation_background
from utils import clean_response
from state import track_message_and_check
import state
from achievements import check_new_achievements, format_achievements
from daily_verdict import generate_daily_verdict
from jealousy import arm_jealous_afterglow, consume_post_return_strike, is_reaction_still_relevant, is_rival_reaction_still_relevant


_GROUP_COMMENT_TONE_INSTRUCTIONS = {
    "dark_joke": (
        "Тон чата: чёрный юмор / мрачная шутка. "
        "Реагируй сухо, язвительно, с лёгким цинизмом. "
        "Можно шутить про морг, драму, кладбищенский вайб, но без настоящей жестокости. "
        "СТРОГО нельзя писать нежности, сердечки, лепестки, романтичные образы, сочувственные открытки."
    ),
    "conflict": (
        "Тон чата: спор / подколы / напряжение. "
        "Реагируй как ехидный наблюдатель. Можно подлить сарказма, но не раздувай конфликт слишком сильно."
    ),
    "flirt": (
        "Тон чата: флирт / заигрывание. "
        "Реагируй колко и ревниво-насмешливо, без пошлости и без explicit."
    ),
    "absurd": (
        "Тон чата: абсурд / мемы / хаос. "
        "Реагируй как человек, который устал понимать этот цирк, но ему смешно."
    ),
    "neutral": (
        "Тон чата: обычный разговор. "
        "Реагируй коротко, наблюдательно и язвительно, без философии и без милоты."
    ),
}


def _format_group_comment_context(messages: list[dict], limit: int = 12) -> str:
    """Форматирует свежий кусок общего чата для бокового комментария."""
    rows = []
    for msg in messages[-limit:]:
        username = (msg.get("username") or "").strip()
        text = re.sub(r"\s+", " ", (msg.get("text") or "").strip())
        if not username or not text:
            continue
        rows.append(f"[{username}]: {text}")
    return "\n".join(rows)


def _detect_group_comment_tone(context: str) -> str:
    """Грубая классификация вайба, чтобы модель не отвечала сердечками на чёрный юмор."""
    t = (context or "").lower()

    serious_self_harm = (
        "хочу умереть",
        "убью себя",
        "покончу с собой",
        "суицид",
        "суицидн",
        "выпил таблетки",
        "выпила таблетки",
        "режу себя",
        "порезал себя",
        "порезала себя",
    )
    if any(x in t for x in serious_self_harm):
        return "skip"

    dark_markers = (
        "умер", "умерла", "умерли", "смерт", "труп", "морг", "похорон",
        "убил", "убила", "убили", "убью", "сдох", "сдохн", "кладбищ",
        "гроб", "кремаци", "внезапная смерть", "откинулся", "откинулась",
    )
    conflict_markers = (
        "дурак", "дура", "бесишь", "заткнись", "отстань", "иди нафиг",
        "иди нах", "ненавижу", "срач", "спор", "агр", "токсик",
    )
    flirt_markers = (
        "люблю", "целую", "поцел", "обним", "малыш", "малышка",
        "котик", "зая", "сердечко", "😘", "😍", "🥰", "❤️", "💋",
    )
    absurd_markers = (
        "ахах", "хаха", "ору", "кринж", "мем", "лол", "жиза",
        "что происходит", "цирк", "шиза", "абсурд",
    )

    if any(x in t for x in dark_markers):
        return "dark_joke"
    if any(x in t for x in conflict_markers):
        return "conflict"
    if any(x in t for x in flirt_markers):
        return "flirt"
    if any(x in t for x in absurd_markers):
        return "absurd"
    return "neutral"


def _clean_group_comment_line(raw: str, tone: str) -> str:
    """Чистит боковой комментарий и отсекает совсем неуместную милоту."""
    line = clean_response(raw or "").strip()
    line = re.sub(r"^[\"'«»]+|[\"'«»]+$", "", line).strip()
    line = re.sub(r"^(?:аня|нейроанька)\s*[:,-]\s*", "", line, flags=re.IGNORECASE).strip()

    line = line.splitlines()[0].strip()
    line = re.sub(r"\s+", " ", line)

    if len(line) > 170:
        cut = line.rfind(" ", 0, 170)
        line = line[: cut if cut > 80 else 170].rstrip(" ,;:-") + "…"

    if tone == "dark_joke":
        forbidden_soft = (
            "серд", "лепест", "нежн", "мил", "обнима", "солныш",
            "дожд", "цветоч", "любов", "🥰", "😍", "😘", "❤️", "💕", "💖",
        )
        if any(x in line.lower() for x in forbidden_soft):
            return ""

    if "?" in line:
        line = line.replace("?", ".")

    return line.strip()


def _fallback_group_comment(tone: str) -> str:
    fallbacks = {
        "dark_joke": [
            "ну всё, чат снова выбрал юмор с запахом морга.",
            "атмосфера такая, будто гроб уже заказали, а повод ещё ищут.",
            "мило, конечно. кладбищенский стендап подъехал.",
        ],
        "conflict": [
            "о, пошёл прогрев перед маленьким локальным срачем.",
            "так, градус растёт. осталось кому-нибудь хлопнуть дверью.",
            "люблю, когда чат делает вид, что это не срач, а дискуссия.",
        ],
        "flirt": [
            "вы тут флиртуете так тонко, что аж табличка «намёк» нужна.",
            "господи, у вас тут романтика или просто чат перегрелся.",
            "ну да, конечно, это всё просто дружеские искры. верим.",
        ],
        "absurd": [
            "я перестала понимать, но почему-то стало лучше.",
            "чат опять свернул туда, где навигатор плачет.",
            "это уже не диалог, это DLC к шизе.",
        ],
        "neutral": [
            "у вас диалог как сериал без бюджета, но я почему-то смотрю.",
            "продолжайте, я просто делаю вид, что не подслушиваю.",
            "такой спокойный хаос, почти уютно, если не вдумываться.",
        ],
    }
    return random.choice(fallbacks.get(tone, fallbacks["neutral"]))


async def _reply_worker(page):
    while True:
        job = await state._reply_queue.get()
        try:
            job_type = job["type"]
            state._llm_busy = True

            if job_type == "text":
                state._llm_busy = False
                await send_response(page, job["text"])
                print(f"  [ОЧЕРЕДЬ] текст: {job['text'][:80]}")

            elif job_type == "group_comment":
                mood = job.get("mood", "")
                mood_desc = MOODS.get(mood, mood or "нейтральное")
                print(f"  [ОЧЕРЕДЬ] group_comment (настроение: {mood})")
                try:
                    recent_messages = job.get("recent_messages") or list(state._global_buffer)[-12:]
                    context = _format_group_comment_context(recent_messages, limit=12)

                    if not context:
                        print("  [GROUP_COMMENT] нет контекста, пропускаю")
                        continue

                    tone = _detect_group_comment_tone(context)
                    if tone == "skip":
                        print("  [GROUP_COMMENT] серьёзный self-harm контекст, шутливый комментарий пропущен")
                        continue

                    tone_instruction = _GROUP_COMMENT_TONE_INSTRUCTIONS.get(
                        tone,
                        _GROUP_COMMENT_TONE_INSTRUCTIONS["neutral"],
                    )

                    system_prompt = (
                        f"Ты — Аня, 18-летняя язвительная альтушка из общего чата. "
                        f"Твой ник: {CHARACTER_NAME}. "
                        "Ты иногда встреваешь в чужой разговор короткой боковой репликой.\n\n"
                        "Главное правило: реагируй НА РЕАЛЬНЫЙ КОНТЕКСТ последних сообщений, а не на своё настроение. "
                        "Не выдумывай тему, не меняй эмоциональный тон сцены.\n\n"
                        f"{tone_instruction}\n\n"
                        "Стиль: коротко, разговорно, колко, как живая девчонка в чате. "
                        "Одна реплика, без ника в начале, без обращения к конкретному человеку. "
                        "Без кавычек, без префиксов вроде «Аня:». "
                        "Не используй сердечки и милые эмодзи, если контекст не флирт. "
                        "Не морализируй и не пиши философские открытки."
                    )

                    user_prompt = (
                        "Последние сообщения общего чата:\n"
                        f"{context}\n\n"
                        f"Текущее настроение Ани: {mood_desc}. Это вторично; контекст важнее настроения.\n\n"
                        "Напиши РОВНО одну короткую боковую реплику в общий чат. "
                        "Максимум 120 символов. "
                        "Она должна звучать так, будто Аня реально услышала последние сообщения и язвительно вставила своё."
                    )

                    resp = await ollama_chat(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        options={
                            "temperature": random.uniform(0.72, 0.82),
                            "repeat_penalty": 1.45,
                            "max_tokens": 80,
                            "num_ctx": 4096,
                        },
                    )

                    line = _clean_group_comment_line(resp["message"]["content"], tone)
                    if not line:
                        line = _fallback_group_comment(tone)

                    await send_text(page, line)
                    print(f"  [GROUP_COMMENT:{tone}] {line[:80]}")

                except Exception as e:
                    print(f"  [GROUP_COMMENT] ошибка: {e}")

            elif job_type == "llm":
                username = job["username"]
                text     = job["text"]
                mood     = job.get("mood", "")
                key = username.lower().strip()
                if state._pending_llm_jobs.get(key) is job:
                    state._pending_llm_jobs.pop(key, None)
                print(f"  [ОЧЕРЕДЬ] генерирую ответ для {username}...")

                # Извлекаем необычные слова юзера и кладём в карантин
                # чтобы бот не зеркалил их в своём ответе
                from utils import extract_unusual_words
                unusual = extract_unusual_words(text, username)
                if unusual:
                    state.add_vocab_quarantine(username, unusual)

                # Если у любимчика активны «последствия» — расходуем один страйк
                consume_post_return_strike()
                response = await get_response(username, text, mood)
                if response and response.strip():
                    print(f"  {LOG_NAME} отвечает: {response[:80]}")
                    await send_response(page, response)
                    # Тикаем карантин: слова живут N ответов, потом уходят
                    state.tick_vocab_quarantine(username)
                else:
                    print(f"  [ОЧЕРЕДЬ] {username}: ответ намеренно пропущен")
                check_new_achievements(username)
                if track_message_and_check(username):
                    asyncio.create_task(evaluate_reputation_background(username, page))

            elif job_type == "search":
                username = job["username"]
                query    = job["query"]
                mood     = job.get("mood", "")
                print(f"  [ОЧЕРЕДЬ] поиск для {username}: {query}")
                response = await get_forced_search_response(username, query, mood)
                await send_response(page, response)
                check_new_achievements(username)

            elif job_type == "opinion":
                username = job["username"]
                mood     = job.get("mood", "")
                print(f"  [ОЧЕРЕДЬ] мнение о {username}...")
                response = await get_self_opinion_response(username, mood)
                await send_response(page, response)
                check_new_achievements(username)

            elif job_type == "compatibility":
                username = job["username"]
                nick1    = job["nick1"]
                nick2    = job["nick2"]
                mood     = job.get("mood", "")
                print(f"  [ОЧЕРЕДЬ] совместимость для {username}: {nick1} + {nick2}")
                response = await get_compatibility_response(username, nick1, nick2, mood)
                await send_response(page, response)
                check_new_achievements(username)


            elif job_type == "find_pair":
                username = job["username"]
                mood     = job.get("mood", "")
                print(f"  [ОЧЕРЕДЬ] ищу пару для {username}")
                response = await get_find_pair_response(username, mood)
                await send_response(page, response)
                check_new_achievements(username)

            elif job_type == "horoscope":
                username = job["username"]
                sign     = job["sign"]
                mood     = job.get("mood", "")
                print(f"  [ОЧЕРЕДЬ] гороскоп для {username} ({sign})")
                response = await get_horoscope_response(username, sign, mood)
                await send_response(page, response)
                check_new_achievements(username)

            elif job_type == "achievements_view":
                username = job["username"]
                print(f"  [ОЧЕРЕДЬ] достижения для {username}")
                save_message(username, "user", f"Ник: {username}\nКоманда: !достижения")
                check_new_achievements(username)
                response = format_achievements(username)
                save_message(username, "assistant", response)
                await send_response(page, response)

            elif job_type == "profile_view":
                username = job["username"]
                print(f"  [ОЧЕРЕДЬ] профиль для {username}")
                save_message(username, "user", f"Ник: {username}\nКоманда: !профиль")
                response = format_user_profile_info(username)
                save_message(username, "assistant", response)
                await send_response(page, response)

            elif job_type == "forget":
                username = job["username"]
                target   = job.get("target", "")
                print(f"  [ОЧЕРЕДЬ] !забудь для {username}: {target!r}")
                save_message(username, "user", f"Ник: {username}\nКоманда: !забудь {target}")
                response = format_forget_response(username, target)
                save_message(username, "assistant", response)
                await send_response(page, response)

            elif job_type == "history_view":
                username = job["username"]
                print(f"  [ОЧЕРЕДЬ] история для {username}")
                response = format_history_summary(username)
                await send_response(page, response)

            elif job_type == "daily_verdict":
                report_date = job["report_date"]
                print(f"  [ОЧЕРЕДЬ] вердикт дня за {report_date}")
                verdict = await generate_daily_verdict(report_date)
                await send_response(page, verdict)

            elif job_type == "greeting":
                username = job["username"]
                mood     = job.get("mood", "")
                greeting = await get_greeting(username, mood)
                await send_text(page, greeting)
                save_message(username, "assistant", greeting)
                print(f"  Приветствие: {greeting}")

            elif job_type == "jealousy":
                username = job["username"]
                payload  = job["payload"]
                if not is_reaction_still_relevant(username, payload["triggered_at"]):
                    print(f"  [РЕВНОСТЬ] реакция устарела для {username}, пропускаю")
                else:
                    stage         = payload["stage"]
                    style         = payload.get("style", "passive")
                    recent        = "\n".join(payload.get("recent_messages", []))
                    rival_name    = payload.get("rival_name", "")
                    context_label = payload.get("context_label", "общение с другими")
                    penalty       = payload.get("penalty", 1)

                    rival_line = (
                        f" Конкретный раздражитель — {rival_name}. "
                        f"Можно прямо кинуть: «ну раз {rival_name} тебе интереснее — вот и иди» "
                        f"или «пусть {rival_name} тебя и развлекает»."
                    ) if rival_name else ""

                    # Стиль-зависимые инструкции
                    _STYLE_INSTRUCTIONS: dict[str, str] = {
                        "icy": (
                            "СТИЛЬ — ЛЕДЯНОЙ ИГНОР. Ответ предельно короткий: 1–2 слова или одно ёмкое предложение. "
                            "Ноль тепла, ноль объяснений. Примеры: «ок»; «угу»; «понятно, слушай»; «и?». "
                            "ЗАПРЕЩЕНО: длинные тирады, объяснения, эмоциональные высказывания."
                        ),
                        "passive": (
                            "СТИЛЬ — ПАССИВНАЯ АГРЕССИЯ. Каждая фраза с двойным дном. "
                            "Примеры: «ой, вдруг вспомнил(а) что я существую»; «нет, всё хорошо. просто хорошо»; "
                            "«интересно, а с другими ты такой же разговорчивый»; "
                            "«можно я тоже буду делать вид что тебя нет». "
                            "Тон — сухой, с иголками. Без открытого скандала."
                        ),
                        "possessive": (
                            "СТИЛЬ — СОБСТВЕННИЧЕСКАЯ ПРЕДЪЯВА. Заявляй права прямо и без стеснения. "
                            "Примеры: «ты мой/моя, на случай если забыл(а)»; «я не делю — это просто так устроено»; "
                            "«мне не нравится когда ты такой щедрый с другими»; "
                            "«у тебя есть я, зачем тебе весь этот цирк». "
                            "Давление без истерики."
                        ),
                        "meltdown": (
                            "СТИЛЬ — ПУБЛИЧНАЯ ИСТЕРИКА. Срываешься, злость без фильтра. "
                            "Примеры: «знаешь что, иди к своим раз я тебе не интересна»; "
                            "«это уже оскорбительно, РЕАЛЬНО»; «я не буду делать вид что меня всё устраивает»; "
                            "«хочешь флиртовать с кем попало — флиртуй, но не ко мне потом». "
                            "Можно капслок одного слова, обрыв фразы, ультиматум."
                        ),
                        "rival_agg": (
                            "СТИЛЬ — АГРЕССИЯ К СОПЕРНИКУ (говоришь это любимчику). "
                            f"Примеры: «передай своему новому другу что это мой/моя человек»; "
                            f"«если я ещё раз увижу такие нежности рядом с тобой — разговор будет другим»; "
                            f"«{rival_name or 'этот человек'} меня бесит, это заметно?». "
                            "Резко, территориально."
                        ),
                    }
                    style_instruction = _STYLE_INSTRUCTIONS.get(style, _STYLE_INSTRUCTIONS["passive"])

                    # Получаем любимчика для промпта
                    from jealousy import sync_current_lover
                    lover_name = sync_current_lover()

                    for attempt in range(2):
                        try:
                            resp = await ollama_chat(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": get_jealousy_system_prompt(username, lover_name)},
                                    {"role": "user", "content": (
                                        f"Ник: {username}\n"
                                        f"[РЕВНОСТЬ stage {stage}, стиль: {style}]\n"
                                        f"Твой любимчик {username} снова игнорит тебя и крутится с другими. "
                                        f"Триггер: {context_label}. Штраф по влюблённости: -{penalty}.{rival_line}\n"
                                        f"Его недавние реплики:\n{recent or '(нет данных)'}\n\n"
                                        f"{style_instruction}\n"
                                        f"ЗАПРЕЩЕНО: писать слово «ревную» прямым текстом. "
                                        f"ЗАПРЕЩЕНО: нейтральный или вежливый тон. "
                                        f"ЗАПРЕЩЕНО: смягчать до шутки.\n"
                                        f"Разрешено 1–3 предложения. "
                                        f"Начни с ника {username}."
                                    )},
                                ],
                                options={"temperature": JEALOUSY_TEMPERATURE},
                            )
                            reaction = clean_response(resp["message"]["content"], username)

                            # Пост-проверка: если модель всё равно написала «ревную» — второй круг
                            if attempt == 0 and reaction and any(
                                w in reaction.lower() for w in ("ревную", "ревнивая", "ревность")
                            ):
                                print(f"  [РЕВНОСТЬ] пост-проверка: обнаружено слово «ревную», перегенерирую...")
                                continue

                            if reaction:
                                await send_response(page, reaction)
                                arm_jealous_afterglow(username, stage)
                                print(f"  [РЕВНОСТЬ stage={stage} style={style}] {reaction[:80]}")
                            break
                        except Exception as e:
                            print(f"  [РЕВНОСТЬ] ошибка генерации (попытка {attempt+1}): {e}")
                            break

            elif job_type == "jealousy_rival":
                username = job["username"]
                payload  = job["payload"]
                lover_name = payload.get("lover_name", "")
                if not is_rival_reaction_still_relevant(username, lover_name, payload.get("triggered_at", 0.0)):
                    print(f"  [РЕВНОСТЬ] реакция на соперника устарела для {username}, пропускаю")
                else:
                    stage         = payload.get("stage", 2)
                    recent        = "\n".join(payload.get("recent_messages", []))
                    context_label = payload.get("context_label", "флирт с любимчиком")
                    penalty       = payload.get("penalty", 3)
                    claim_word    = payload.get("claim_word", "мой человек")
                    excerpt       = payload.get("excerpt", "")

                    stage_instruction = {
                        2: (
                            "Ты резко одёргиваешь и ставишь на место. "
                            f"Примеры: «{username}, {lover_name} — {claim_word}, и это не обсуждается»; "
                            f"«{username}, ты вообще понимаешь что делаешь?»; "
                            f"«{username}, руки убрал(а)». "
                            "Жёстко и конкретно. Не шути."
                        ),
                        3: (
                            "Ты в режиме открытого скандала — собственничество и злость без фильтра. "
                            f"Примеры: «{username}, это мой человек. Ты понял(а)?»; "
                            f"«{username}, ещё раз увижу такое — поговорим по-другому»; "
                            f"«{username}, ты сейчас вообще наглеешь». "
                            "Можно угрожать, давить, говорить жёстко. Никаких смягчений."
                        ),
                    }.get(stage, f"Ты агрессивно ставишь {username} на место.")

                    try:
                        resp = await ollama_chat(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": get_jealousy_system_prompt(username, lover_name)},
                                {"role": "user", "content": (
                                    f"Ник: {username}\n"
                                    f"[РЕВНОСТЬ К СОПЕРНИКУ stage {stage}]\n"
                                    f"{username} флиртует с твоим любимчиком {lover_name}. "
                                    f"{lover_name} — это {claim_word}.\n"
                                    f"Что написал {username}: «{excerpt or '(флирт в общем чате)'}»\n"
                                    f"Контекст: {context_label}. Штраф ревности: -{penalty}.\n\n"
                                    f"{stage_instruction}\n"
                                    f"ВАЖНО: ты отвечаешь именно {username} (сопернику), НЕ {lover_name}. "
                                    f"ЗАПРЕЩЕНО: нейтральный тон, шутить, смягчать. "
                                    f"ЗАПРЕЩЕНО: говорить что тебе «всё равно» или «неважно». "
                                    f"1–2 предложения. Начни строго с ника {username}."
                                )},
                            ],
                            options={"temperature": JEALOUSY_TEMPERATURE},
                        )
                        reaction = clean_response(resp["message"]["content"], username)
                        if reaction:
                            await send_response(page, reaction)
                            print(f"  [РЕВНОСТЬ сопернику stage={stage}] {reaction[:80]}")
                    except Exception as e:
                        print(f"  [РЕВНОСТЬ] ошибка реакции на соперника: {e}")

            elif job_type == "comfort":
                username = job["username"]
                target   = job["target"]
                print(f"  [ОЧЕРЕДЬ] утешаю {target}...")
                save_message(username, "user", f"Ник: {username}\nКоманда: !утешить {target}")
                try:
                    resp = await ollama_chat(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": get_system_prompt(target)},
                            {"role": "user", "content": (
                                f"Ник: {target}\n"
                                f"Придумай саркастичное, язвительное, но смешное утешение для {target}. "
                                f"Дружеское подшучивание — обидно, но без злобы. "
                                f"Начни строго с ника {target}. Одно предложение."
                            )},
                        ],
                        options={"temperature": COMFORT_TEMPERATURE, "num_ctx": 8192},
                    )
                    answer = clean_response(resp["message"]["content"], target)
                    if answer:
                        save_message(username, "assistant", answer)
                        await send_response(page, answer)
                        print(f"  [УТЕШИТЬ] {answer[:80]}")
                        check_new_achievements(username)
                except Exception as e:
                    print(f"  [УТЕШИТЬ] ошибка: {e}")

            elif job_type == "roast":
                username = job["username"]
                target   = job["target"]
                print(f"  [ОЧЕРЕДЬ] агрюсь на {target}...")
                save_message(username, "user", f"Ник: {username}\nКоманда: !заагрись {target}")
                roast_prompt = (
                    f"Ты Аня — резкая альтушка с острым языком, сейчас в бешенстве.\n"
                    f"Придумывай злые, язвительные подколки для {target}. "
                    f"Бей по самолюбию, используй неожиданные сравнения и абсурдные образы. "
                    f"Каждая реплика уникальна по структуре.\n"
                    f"Начни с ника {target}. Одно-два предложения."
                )
                try:
                    for i in range(3):
                        resp = await ollama_chat(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": roast_prompt},
                                {"role": "user", "content": (
                                    f"Реплика {i+1} из 3 для {target}. "
                                    f"Новый угол атаки, не повторяй предыдущие образы."
                                )},
                            ],
                            options={"temperature": ROAST_TEMPERATURE, "num_ctx": 8192},
                        )
                        insult = clean_response(resp["message"]["content"], target)
                        if insult:
                            save_message(username, "assistant", insult)
                            await send_response(page, insult)
                            print(f"  [ЗААГРИСЬ {i+1}/3] {insult[:80]}")
                        if i < 2:
                            await asyncio.sleep(random.uniform(22, 30))
                    check_new_achievements(username)
                except Exception as e:
                    print(f"  [ЗААГРИСЬ] ошибка: {e}")

        except Exception as e:
            print(f"  [ОЧЕРЕДЬ] ошибка обработки: {e}")
        finally:
            state._llm_busy = False
            state._reply_queue.task_done()
            cooldown = random.uniform(*RESPONSE_COOLDOWN)
            print(f"  [ОЧЕРЕДЬ] кулдаун {cooldown:.1f}с (осталось: {state._reply_queue.qsize()})")
            await asyncio.sleep(cooldown)
