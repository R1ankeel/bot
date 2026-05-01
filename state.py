"""
state.py — общее изменяемое состояние бота.

Все модули импортируют флаги, счётчики и буфер отсюда,
чтобы не было циклических зависимостей.
"""

import asyncio
import collections
import random
import re
import time
from datetime import datetime

from config import (
    CHARACTER_NAME,
    ACTIVITY_CHANGE_INTERVAL,
    DAY_ACTIVITY_SCHEDULE,
    DEFAULT_MODE,
    HIDDEN_REMINDER_MAX_EVERY,
    HIDDEN_REMINDER_MIN_EVERY,
)

# ── Настройки уровня бота (не конфигурация персонажа) ────────────────

REPUTATION_EVAL_EVERY  = 10    # анализ репутации каждые N сообщений от юзера
FACTS_ANALYZE_EVERY    = 30    # анализ фактов каждые N необработанных сообщений
FACTS_ANALYZE_INTERVAL = 120   # минимум секунд между анализами фактов
DIARY_INTERVAL         = 7200  # минимум секунд между записями дневника планеты
DIARY_MIN_MESSAGES     = 40    # минимум новых сообщений для записи дневника
FACTS_MENTION_EVERY    = 10    # вставлять факты в промпт каждые N сообщений от юзера
STORY_HINT_EVERY       = 6     # как часто подмешивать сюжетную подсказку

# ── Режим персонажа ───────────────────────────────────────────────────

current_mode: int = DEFAULT_MODE


def set_mode(new_mode: int):
    """Меняет режим персонажа и сбрасывает все style-сессии.

    При смене режима (милая↔стерва) старый стиль уже не подходит —
    сбрасываем чтобы все пользователи получили новый стиль сразу.
    """
    global current_mode
    if new_mode != current_mode:
        current_mode = new_mode
        _user_style_sessions.clear()
        print(f"[РЕЖИМ] переключён на {new_mode}, style-сессии сброшены")

# ── Флаги фоновых задач ───────────────────────────────────────────────

_facts_task_running = False
_diary_task_running = False
_llm_busy = False          # True пока воркер генерирует ответ

# ── Ollama lock ───────────────────────────────────────────────────────

_ollama_lock: asyncio.Lock | None = None


def get_ollama_lock() -> asyncio.Lock:
    global _ollama_lock
    if _ollama_lock is None:
        _ollama_lock = asyncio.Lock()
    return _ollama_lock


# ── Очередь ответов ───────────────────────────────────────────────────

_reply_queue: asyncio.Queue | None = None
MAX_MERGED_LLM_TEXT = 1200
_pending_llm_jobs: dict[str, dict] = {}


def _trim_merged_llm_text(text: str, max_len: int = MAX_MERGED_LLM_TEXT) -> str:
    """Ограничивает накопленный текст, сохраняя самую актуальную (последнюю) часть."""
    normalized = (text or "").strip()
    if len(normalized) <= max_len:
        return normalized
    trimmed = normalized[-max_len:]
    nl_idx = trimmed.find("\n")
    if 0 <= nl_idx < max_len - 1:
        trimmed = trimmed[nl_idx + 1:]
    return trimmed.strip()


async def enqueue_llm_job(username: str, text: str, mood: str):
    """Кладёт LLM-задачу в очередь или склеивает с уже ожидающей задачей юзера."""
    if _reply_queue is None:
        return
    key = (username or "").lower().strip()
    incoming = (text or "").strip()
    if not key or not incoming:
        return

    pending = _pending_llm_jobs.get(key)
    if pending is not None:
        pending["text"] = _trim_merged_llm_text(f"{pending.get('text', '').rstrip()}\n{incoming}")
        pending["mood"] = mood
        print(f"[ОЧЕРЕДЬ] склеено сообщение для {username}: теперь {len(pending['text'])} симв")
        return

    job = {
        "type": "llm",
        "username": username,
        "text": _trim_merged_llm_text(incoming),
        "mood": mood,
    }
    _pending_llm_jobs[key] = job
    await _reply_queue.put(job)
    print(f"[ОЧЕРЕДЬ] новый llm job для {username}")

# ── Счётчики сообщений на юзера ───────────────────────────────────────

_user_msg_counters:   dict[str, int] = {}
_user_facts_counters: dict[str, int] = {}
_user_story_counters: dict[str, int] = {}
_user_hidden_reminder_counters: dict[str, int] = {}
_user_hidden_reminder_thresholds: dict[str, int] = {}

# ── Таймеры фонового анализа ──────────────────────────────────────────

_last_facts_analysis    = 0.0
_last_diary_created_at  = ""
_daily_verdict_queued_for = ""

# ── Текущая активность (единая для всех) ──────────────────────────────

_current_activity: str = ""
_last_activity_change: float = 0.0
_current_activity_slot: tuple[int, int] | None = None


def _pick_activity_for_hour(hour: int) -> tuple[str, tuple[int, int]]:
    for (start, end), activities in DAY_ACTIVITY_SCHEDULE:
        if start <= hour <= end:
            return random.choice(activities), (start, end)
    return "сижу в чате и никого не трогаю", (hour, hour)


def update_activity(now_dt: datetime | None = None):
    """Обновляет текущую активность по таймеру и временному слоту суток."""
    global _current_activity, _last_activity_change, _current_activity_slot
    now_dt = now_dt or datetime.now()
    now_ts = time.time()
    hour = now_dt.hour
    _, target_slot = _pick_activity_for_hour(hour)

    should_refresh = (
        not _current_activity
        or now_ts - _last_activity_change >= ACTIVITY_CHANGE_INTERVAL
        or _current_activity_slot != target_slot
    )
    if should_refresh:
        _current_activity, _current_activity_slot = _pick_activity_for_hour(hour)
        _last_activity_change = now_ts
        print(f"[АКТИВНОСТЬ] {hour:02d}:00 → {_current_activity}")


# ── Состояние ревности ───────────────────────────────────────────────

_jealousy_current_lover:        str   = ""
_jealousy_last_direct_at:       float = 0.0
_jealousy_last_public_at:       float = 0.0
_jealousy_public_ignored_count: int   = 0
_jealousy_stage:                int   = 0
_jealousy_last_reaction_at:     float = 0.0
_jealousy_reply_hint_until:     float = 0.0
_jealousy_afterglow_until:      float = 0.0
_jealousy_afterglow_stage:      int   = 0
_jealousy_last_rival:           str   = ""
_jealousy_last_penalty:         int   = 0
_jealousy_last_context_label:   str   = ""
_jealousy_last_context_excerpt: str   = ""
_jealousy_recent_public: collections.deque = collections.deque(maxlen=5)

# ── Буфер последних сообщений общего чата (50 шт.) ───────────────────

_global_buffer: collections.deque = collections.deque(maxlen=50)


# ── Social Graph: кто с кем общается ──────────────────────────────────
# Храним "свежесть" взаимодействий и "аффинити" (сколько раз пересекались)
# за последние ~30–40 сообщений. Это чисто in-memory сигнал для логики.

_recent_pair_activity: dict[tuple[str, str], float] = {}
# (user1_lower, user2_lower) -> timestamp последнего взаимодействия

_pair_affinity: dict[tuple[str, str], int] = {}
# (user1_lower, user2_lower) -> сколько раз они общались за последние ~30–40 сообщений

# Внутренний скользящий журнал последних "парных событий", чтобы _pair_affinity
# примерно отражал активность в коротком окне (а не рос бесконечно).
_pair_event_window: collections.deque[tuple[str, str]] = collections.deque(maxlen=40)

# Пара считается "протухшей" если давно не было событий. Держим окно 20–25 минут.
_PAIR_STALE_TTL_SECONDS = 25 * 60


def _pair_key(a: str, b: str) -> tuple[str, str] | None:
    """Нормализует пару участников в канонический ключ (lower + сортировка).

    Возвращает None если данные невалидны или это self-pair.
    """
    a_l = (a or "").strip().lower()
    b_l = (b or "").strip().lower()
    if not a_l or not b_l:
        return None
    if a_l == b_l:
        return None
    return (a_l, b_l) if a_l < b_l else (b_l, a_l)


def _looks_like_username_mention(text_lower: str, username_lower: str) -> bool:
    """True если в тексте похоже упомянут username_lower.

    Для коротких ников используем word-boundary, чтобы не ловить мусор.
    Также поддерживаем '@ник' как частный случай.
    """
    if not text_lower or not username_lower:
        return False
    if len(username_lower) <= 4:
        return bool(re.search(r"(?<!\w)" + re.escape(username_lower) + r"(?!\w)", text_lower))
    if f"@{username_lower}" in text_lower:
        return True
    return username_lower in text_lower


def _purge_stale_pairs(now_ts: float) -> None:
    """Удаляет пары, у которых давно не было взаимодействий."""
    cutoff = now_ts - _PAIR_STALE_TTL_SECONDS
    stale = [k for k, ts in _recent_pair_activity.items() if ts < cutoff]
    for k in stale:
        _recent_pair_activity.pop(k, None)
        _pair_affinity.pop(k, None)

    # Очищаем окно событий от уже удалённых пар, чтобы не портить счётчики на pop().
    if stale and _pair_event_window:
        kept = [k for k in _pair_event_window if k in _recent_pair_activity]
        _pair_event_window.clear()
        _pair_event_window.extend(kept[-_pair_event_window.maxlen:])


def track_group_interaction(username: str, text: str) -> None:
    """Обновляет Social Graph по одному сообщению из общего чата.

    Правила:
    - игнорируем действия ([ДЕЙСТВИЕ]) и сообщения персонажа (CHARACTER_NAME)
    - ищем потенциальных участников: упомянутые в сообщении + авторы последних 10–12 сообщений
    - обновляем:
        - _recent_pair_activity (last seen timestamp)
        - _pair_affinity (сколько "событий пары" в окне ~40 событий)
    - чистим протухшие пары (старше ~25 минут)
    """
    try:
        author = (username or "").strip()
        if not author:
            return
        if author == "[ДЕЙСТВИЕ]":
            return
        if author.lower() == (CHARACTER_NAME or "").strip().lower():
            return

        msg = (text or "").strip()
        if not msg:
            return

        now_ts = time.time()

        # Собираем последних активных авторов из буфера.
        recent_authors: list[str] = []
        seen: set[str] = set()
        for m in reversed(_global_buffer):
            u = (m.get("username") or "").strip()
            if not u:
                continue
            if u == "[ДЕЙСТВИЕ]":
                continue
            if u.lower() == (CHARACTER_NAME or "").strip().lower():
                continue
            u_l = u.lower()
            if u_l in seen:
                continue
            seen.add(u_l)
            recent_authors.append(u)
            if len(recent_authors) >= 12:
                break

        # Из "известных" имён пытаемся вытащить упоминания из текста.
        # Берём немного шире чем 12, чтобы чаще попадали адресаты.
        known_users: list[str] = []
        known_seen: set[str] = set()
        for m in reversed(_global_buffer):
            u = (m.get("username") or "").strip()
            if not u or u == "[ДЕЙСТВИЕ]":
                continue
            if u.lower() == (CHARACTER_NAME or "").strip().lower():
                continue
            u_l = u.lower()
            if u_l in known_seen:
                continue
            known_seen.add(u_l)
            known_users.append(u)
            if len(known_users) >= 30:
                break

        text_lower = msg.lower()
        mentioned_users: list[str] = []
        for u in known_users:
            u_l = u.lower()
            if u_l == author.lower():
                continue
            if _looks_like_username_mention(text_lower, u_l):
                mentioned_users.append(u)

        # Итоговые "контакты" автора: упомянутые + недавние авторы.
        # Упор на уникальность по lower().
        contacts: list[str] = []
        contact_seen: set[str] = set()
        for u in (mentioned_users + recent_authors):
            u_l = (u or "").strip().lower()
            if not u_l or u_l == author.lower():
                continue
            if u_l in contact_seen:
                continue
            contact_seen.add(u_l)
            contacts.append(u)

        if not contacts:
            _purge_stale_pairs(now_ts)
            return

        # Записываем события по всем парам author<->contact.
        for other in contacts:
            key = _pair_key(author, other)
            if key is None:
                continue

            _recent_pair_activity[key] = now_ts

            # Скользящее окно: добавляем событие и поддерживаем счётчик.
            if len(_pair_event_window) >= _pair_event_window.maxlen:
                old = _pair_event_window.popleft()
                if old in _pair_affinity:
                    _pair_affinity[old] -= 1
                    if _pair_affinity[old] <= 0:
                        _pair_affinity.pop(old, None)
                        _recent_pair_activity.pop(old, None)

            _pair_event_window.append(key)
            _pair_affinity[key] = _pair_affinity.get(key, 0) + 1

        _purge_stale_pairs(now_ts)

    except Exception as e:
        # Ошибки Social Graph не должны ломать основной цикл.
        print(f"[SOCIAL_GRAPH] ошибка: {type(e).__name__}: {e}")


def buffer_add(username: str, text: str):
    _global_buffer.append({"username": username, "text": text})


def buffer_search(mention: str, limit: int = 10) -> list[dict]:
    mention_lower = mention.lower()
    results = [
        msg for msg in _global_buffer
        if mention_lower in msg["username"].lower() or mention_lower in msg["text"].lower()
    ]
    return results[-limit:]


def buffer_get_context(usernames: list[str], limit: int = 15) -> str:
    """Возвращает сообщения из буфера где упоминаются указанные пользователи.

    Матчит и по полю username (автор сообщения), и по тексту (упоминание ника).
    Для коротких ников (≤4 символа) использует word-boundary чтобы избежать
    ложных срабатываний.
    """
    user_lowers = [u.lower() for u in usernames if u]
    if not user_lowers:
        return ""

    matched = []
    for msg in _global_buffer:
        msg_author = msg["username"].lower()
        msg_text   = msg["text"].lower()

        hit = False
        for u in user_lowers:
            # Матч по автору
            if u in msg_author:
                hit = True
                break
            # Матч по тексту — word boundary для коротких ников
            if len(u) <= 4:
                if re.search(r'(?<!\w)' + re.escape(u) + r'(?!\w)', msg_text):
                    hit = True
                    break
            elif u in msg_text:
                hit = True
                break

        if hit:
            matched.append(msg)

    return "\n".join(f"[{m['username']}]: {m['text']}" for m in matched[-limit:]) if matched else ""


def buffer_get_recent(limit: int = 8) -> str:
    """Возвращает последние N сообщений из общего чата (все участники)."""
    msgs = list(_global_buffer)[-limit:]
    return "\n".join(f"[{msg['username']}]: {msg['text']}" for msg in msgs) if msgs else ""


# ── Сессионные стили ответа (per-user, меняются раз в 15 минут) ───────
# Это решает проблему хаотичности когда случайный hint меняется на каждый ответ.

STYLE_SESSION_TTL = 900  # 15 минут

_user_style_sessions: dict[str, tuple[str, float]] = {}


def get_user_style_hint(username: str, hints: list[str]) -> str:
    """Возвращает стиль для конкретного юзера.

    Стиль меняется не чаще раза в STYLE_SESSION_TTL секунд.
    Это делает поведение Аньки более цельным, а не рандомайзером.
    """
    if not hints:
        return ""
    now = time.time()
    entry = _user_style_sessions.get(username)
    if entry is None or now - entry[1] >= STYLE_SESSION_TTL:
        hint = random.choice(hints)
        _user_style_sessions[username] = (hint, now)
        print(f"  [СТИЛЬ] {username}: новый стиль-сессия → {hint[:60]}")
        return hint
    return entry[0]


def invalidate_user_style(username: str):
    """Принудительно сбрасывает стиль-сессию (при смене настроения и т.п.)."""
    _user_style_sessions.pop(username, None)


# ── Pending dialog state (открытый вопрос бота к юзеру) ──────────────
# Если бот задал вопрос, следующий короткий ответ юзера — это ответ на него.
# Позволяет строить натуральный follow-up вместо "перезагрузки" диалога.

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "музыка":     ["музык", "слуша", "группа", "трек", "альбом", "плейлист", "жанр"],
    "игры":       ["игр", "играешь", "геймер", "стрим", "персонаж", "квест"],
    "учёба":      ["учёб", "универ", "школ", "экзамен", "пара ", "препод", "сессия"],
    "работа":     ["работ", "офис", "коллег", "босс", "зарплат", "проект"],
    "отношения":  ["отношени", "парень", "девушк", "симпати", "нравится", "влюб"],
    "семья":      ["мама", "папа", "брат", "сестра", "родител", "семья"],
    "настроение": ["настроени", "грустн", "весел", "устал", "доволен", "скуч"],
    "планы":      ["планы", "собираешься", "хочешь", "мечта", "завтра", "выходные"],
    "спорт":      ["спорт", "трениров", "качалк", "бегаешь", "физкульт"],
    "еда":        ["ела", "ешь", "готовишь", "кушала", "голодн", "рецепт"],
}


def _extract_topic_from_question(question: str) -> str:
    """Определяет тему вопроса для pending_dialog."""
    q = question.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return topic
    return ""


PENDING_DIALOG_TTL = 300  # 5 минут

_pending_dialog: dict[str, dict] = {}
# {username: {question: str, topic: str, expires_at: float}}


def set_pending_dialog(username: str, question: str, topic: str = ""):
    """Запоминает открытый вопрос бота к пользователю.

    Если topic не передан — пробуем определить его из вопроса автоматически.
    """
    if not topic:
        topic = _extract_topic_from_question(question)
    _pending_dialog[username] = {
        "question": question,
        "topic":    topic,
        "expires_at": time.time() + PENDING_DIALOG_TTL,
    }
    topic_hint = f" (тема: {topic})" if topic else ""
    print(f"  [PENDING] {username}: открытый вопрос{topic_hint} → {question[:80]}")


def get_pending_dialog(username: str) -> dict | None:
    """Возвращает открытый вопрос если он ещё актуален, иначе None.

    Возвращаемый dict: {question, topic, expires_at}
    topic — автоматически определённая тема (музыка/игры/работа и т.п.)
    """
    entry = _pending_dialog.get(username)
    if entry is None:
        return None
    if time.time() >= entry["expires_at"]:
        del _pending_dialog[username]
        return None
    return entry


def clear_pending_dialog(username: str):
    """Закрывает открытый вопрос (пользователь ответил)."""
    removed = _pending_dialog.pop(username, None)
    if removed:
        print(f"  [PENDING] {username}: вопрос закрыт")


# ── Счётчики ──────────────────────────────────────────────────────────

def should_include_facts(username: str) -> bool:
    """True каждые FACTS_MENTION_EVERY сообщений от юзера."""
    _user_facts_counters[username] = _user_facts_counters.get(username, 0) + 1
    if _user_facts_counters[username] >= FACTS_MENTION_EVERY:
        _user_facts_counters[username] = 0
        return True
    return False


def should_include_story_hint(username: str) -> bool:
    """True каждые STORY_HINT_EVERY сообщений от юзера."""
    _user_story_counters[username] = _user_story_counters.get(username, 0) + 1
    if _user_story_counters[username] >= STORY_HINT_EVERY:
        _user_story_counters[username] = 0
        return True
    return False


def should_include_hidden_reminder(username: str) -> bool:
    """True раз в 8–10 сообщений — подмешать скрытое системное напоминание."""
    if not username:
        return False

    _user_hidden_reminder_counters[username] = _user_hidden_reminder_counters.get(username, 0) + 1
    threshold = _user_hidden_reminder_thresholds.get(username)
    if threshold is None:
        threshold = random.randint(HIDDEN_REMINDER_MIN_EVERY, HIDDEN_REMINDER_MAX_EVERY)
        _user_hidden_reminder_thresholds[username] = threshold

    if _user_hidden_reminder_counters[username] >= threshold:
        _user_hidden_reminder_counters[username] = 0
        _user_hidden_reminder_thresholds[username] = random.randint(HIDDEN_REMINDER_MIN_EVERY, HIDDEN_REMINDER_MAX_EVERY)
        return True
    return False


def track_message_and_check(username: str) -> bool:
    """True каждые REPUTATION_EVAL_EVERY сообщений — сигнал пересчитать репутацию."""
    _user_msg_counters[username] = _user_msg_counters.get(username, 0) + 1
    if _user_msg_counters[username] >= REPUTATION_EVAL_EVERY:
        _user_msg_counters[username] = 0
        return True
    return False


# ── Emotion tracking per user ─────────────────────────────────────────
# Отслеживаем эмоциональный тон последних сообщений пользователя.
# Это "мгновенное" состояние диалога — для current-thread block в промпте.

_user_last_emotion: dict[str, str] = {}
# Возможные значения: "flirty", "angry", "positive", "negative", "neutral"

_user_emotion_timestamps: dict[str, float] = {}
EMOTION_DECAY_TTL = 3600  # 1 час без новых сообщений → emotion сбрасывается в neutral

# Словари маркеров по тональностям (подстроки, регистр игнорируется)
_EMOTION_MARKERS: dict[str, list[str]] = {
    "flirty": [
        "люблю тебя", "обожаю", "нравишься", "целую", "обними",
        "красивая", "симпатичная", "флирт", "моя аня", "мечта",
        "сексуальная", "горячая", "хочу тебя",
    ],
    "angry": [
        "дура", "тупая", "бесишь", "ненавижу", "заткнись", "идиотка",
        "надоела", "отстань", "достала", "идиот", "тупой", "раздражаешь",
        "мусор", "ты хуже",
    ],
    "positive": [
        "спасибо", "благодарю", "здорово", "классно", "супер", "отлично",
        "кайф", "огонь", "топ", "круто", "молодец", "прекрасно", "люблю",
        "рад", "рада", "доволен", "счастлив", "лучшая",
    ],
    "negative": [
        "грустно", "плохо", "устал", "надоело", "скучно", "всё плохо",
        "грущу", "тяжело", "хреново", "не могу", "сложно", "устала",
        "депресс", "болею", "устал", "пофиг",
    ],
}


def detect_message_emotion(text: str) -> str:
    """Определяет эмоциональный тон сообщения по маркерам."""
    t = text.lower()
    # Проверяем в порядке приоритета: angry > flirty > positive > negative
    for emotion in ("angry", "flirty", "positive", "negative"):
        if any(marker in t for marker in _EMOTION_MARKERS[emotion]):
            return emotion
    return "neutral"


def update_user_emotion(username: str, text: str) -> str:
    """Обновляет эмоциональное состояние диалога с юзером и возвращает его.

    neutral не перезаписывает предыдущее — чтобы не сбрасывать контекст.
    Обновляет timestamp для механизма decay.
    """
    emotion = detect_message_emotion(text)
    if emotion != "neutral":
        _user_last_emotion[username] = emotion
        _user_emotion_timestamps[username] = time.time()
        print(f"  [ЭМОЦИЯ] {username}: {emotion}")
    else:
        # Обновляем timestamp даже для neutral — продлеваем жизнь текущей эмоции
        if username in _user_last_emotion:
            _user_emotion_timestamps[username] = time.time()
    return emotion


def get_user_emotion(username: str) -> str:
    """Возвращает текущее эмоциональное состояние с учётом decay.

    Если с последнего сообщения прошло > EMOTION_DECAY_TTL секунд — сбрасывает в neutral.
    """
    emotion = _user_last_emotion.get(username)
    if not emotion:
        return "neutral"

    ts = _user_emotion_timestamps.get(username, 0.0)
    if time.time() - ts > EMOTION_DECAY_TTL:
        # Эмоция устарела — сбрасываем
        _user_last_emotion.pop(username, None)
        _user_emotion_timestamps.pop(username, None)
        print(f"  [ЭМОЦИЯ] {username}: decay → neutral")
        return "neutral"

    return emotion


def clear_user_emotion(username: str):
    """Принудительно сбрасывает эмоциональное состояние."""
    _user_last_emotion.pop(username, None)
    _user_emotion_timestamps.pop(username, None)


def reset_user_runtime_state(username: str):
    """Полный сброс всего in-memory состояния для пользователя.

    Вызывается из !забудь профиль / !забудь всё чтобы бот не продолжал
    вести себя так, будто помнит прошлый вайб разговора.

    Чистит: pending dialog, emotion, style session, blocked patterns.
    """
    clear_pending_dialog(username)
    clear_user_emotion(username)
    invalidate_user_style(username)

    # Паттерн-блокировки
    _pattern_state.pop(username, None)

    # Карантин лексики
    _vocab_quarantine.pop(username, None)

    # Ревность — сбрасываем reply hint если это любимчик
    if _jealousy_current_lover.lower() == username.lower():
        global _jealousy_reply_hint_until, _jealousy_afterglow_until, _jealousy_afterglow_stage
        global _jealousy_last_rival, _jealousy_last_penalty, _jealousy_last_context_label, _jealousy_last_context_excerpt
        _jealousy_reply_hint_until = 0.0
        _jealousy_afterglow_until = 0.0
        _jealousy_afterglow_stage = 0
        _jealousy_last_rival = ""
        _jealousy_last_penalty = 0
        _jealousy_last_context_label = ""
        _jealousy_last_context_excerpt = ""

    print(f"  [RESET] {username}: всё in-memory состояние сброшено")


# ── Блокировка шаблонных паттернов per-user ───────────────────────────
# Если один и тот же паттерн встречается 2+ раза за короткое время —
# блокируем его на PATTERN_BLOCK_TTL секунд. Это делает антиповтор умнее:
# не просто сравниваем слова, а блокируем конкретные поведенческие шаблоны.

PATTERN_BLOCK_TTL    = 1500   # 25 минут блокировки после срабатывания
PATTERN_WINDOW_TTL   = 900    # 15 минут окно для подсчёта повторов

# Паттерны: (подстрока для поиска, человекочитаемое описание для промпта)
_TRACKABLE_PATTERNS: list[tuple[str, str]] = [
    ("обним",              "обнять/обниму/обнимаю"),
    ("поцел",              "поцелуй/поцелую"),
    ("прижм",              "прижмись/прижаться"),
    ("провокатор",         "ну ты и провокатор"),
    ("игривая, но",        "игривая, но приличная"),
    ("ты сегодня в атаке", "ты сегодня в атаке"),
    ("давай лучше про",    "давай лучше про твой день"),
    ("ой, ты",             "ой, ты ..."),
]

# {username: {pattern_key: {"count": int, "first_seen": float, "blocked_until": float}}}
_pattern_state: dict[str, dict[str, dict]] = {}


def check_and_block_patterns(username: str, bot_answer: str):
    """Проверяет паттерны в ответе бота и блокирует повторяющиеся.

    Вызывать после каждой успешной генерации ответа.
    """
    now = time.time()
    if username not in _pattern_state:
        _pattern_state[username] = {}

    answer_lower = bot_answer.lower()

    for key, _desc in _TRACKABLE_PATTERNS:
        if key not in answer_lower:
            continue

        entry = _pattern_state[username].get(key)
        if entry is None:
            _pattern_state[username][key] = {"count": 1, "first_seen": now, "blocked_until": 0.0}
            continue

        # Сбрасываем счётчик если окно устарело
        if now - entry["first_seen"] > PATTERN_WINDOW_TTL:
            _pattern_state[username][key] = {"count": 1, "first_seen": now, "blocked_until": 0.0}
            continue

        entry["count"] += 1
        # Порог 2 — заблокировать
        if entry["count"] >= 2 and now >= entry["blocked_until"]:
            entry["blocked_until"] = now + PATTERN_BLOCK_TTL
            entry["count"] = 0
            entry["first_seen"] = now
            print(f"  [ПАТТЕРН] {username}: заблокирован '{key}' на {PATTERN_BLOCK_TTL//60} мин")


def get_blocked_patterns(username: str) -> list[str]:
    """Возвращает человекочитаемые описания заблокированных сейчас паттернов."""
    now = time.time()
    result = []
    user_state = _pattern_state.get(username, {})
    for key, desc in _TRACKABLE_PATTERNS:
        entry = user_state.get(key)
        if entry and now < entry.get("blocked_until", 0.0):
            result.append(desc)
    return result


# ── Карантин лексики (анти-эхо) ──────────────────────────────────────
# Необычные слова из сообщений юзера кладём в карантин на N ответов или T минут.
# Промпт читает этот список и запрещает боту их использовать.
# Это решает проблему «морковка следующие N сообщений»:
# бот видит в промпте запрет — и не подхватывает лексикон юзера.

VOCAB_QUARANTINE_TTL = 600  # секунд (10 минут) до автоистечения
# Сколько ответов бота слово держится в карантине — в диапазоне 8–10 (случайно на каждое новое слово).
VOCAB_QUARANTINE_MAX_MSGS_MIN = 8
VOCAB_QUARANTINE_MAX_MSGS_MAX = 10

# {username: [{word: str, added_at: float, responses_left: int}]}
_vocab_quarantine: dict[str, list[dict]] = {}


def add_vocab_quarantine(username: str, words: list[str]):
    """Добавляет необычные слова юзера в карантин (если их там ещё нет)."""
    if not words:
        return
    now = time.time()
    bucket = _vocab_quarantine.setdefault(username, [])
    # Чистим протухшие сначала
    bucket[:] = [
        e for e in bucket
        if now - e["added_at"] < VOCAB_QUARANTINE_TTL and e["responses_left"] > 0
    ]
    existing = {e["word"] for e in bucket}
    added = []
    for w in words:
        if w not in existing:
            bucket.append({
                "word":           w,
                "added_at":       now,
                "responses_left": random.randint(
                    VOCAB_QUARANTINE_MAX_MSGS_MIN,
                    VOCAB_QUARANTINE_MAX_MSGS_MAX,
                ),
            })
            existing.add(w)
            added.append(w)
    if added:
        print(f"  [ВОКАБ-КАРАНТИН] {username}: добавлено {added}")


def tick_vocab_quarantine(username: str):
    """Вызывать после каждого успешного ответа бота — уменьшает счётчики.

    Когда responses_left доходит до 0 — слово покидает карантин.
    Это обеспечивает автоматическое затухание через N ответов.
    """
    bucket = _vocab_quarantine.get(username)
    if not bucket:
        return
    now = time.time()
    remaining = []
    for e in bucket:
        if now - e["added_at"] >= VOCAB_QUARANTINE_TTL:
            continue
        e["responses_left"] -= 1
        if e["responses_left"] > 0:
            remaining.append(e)
    _vocab_quarantine[username] = remaining


def get_vocab_quarantine(username: str) -> list[str]:
    """Возвращает активные слова в карантине для данного юзера."""
    now = time.time()
    return [
        e["word"] for e in _vocab_quarantine.get(username, [])
        if now - e["added_at"] < VOCAB_QUARANTINE_TTL and e["responses_left"] > 0
    ]


def clear_vocab_quarantine(username: str):
    """Принудительно сбрасывает карантин (напр. при !забудь)."""
    _vocab_quarantine.pop(username, None)


# Хранит текущий уровень усталости для каждого юзера.
# Обновляется из evaluate_reputation, читается из prompt_builder.

_spam_fatigue_level: dict[str, int] = {}    # username → 0..3
_spam_fatigue_updated: dict[str, float] = {}  # username → timestamp
SPAM_FATIGUE_DECAY_TTL = 600  # 10 минут без спама → усталость сбрасывается


def update_spam_fatigue(username: str, level: int):
    """Устанавливает текущий уровень спам-усталости для юзера."""
    _spam_fatigue_level[username] = level
    _spam_fatigue_updated[username] = time.time()


def get_spam_fatigue(username: str) -> int:
    """Возвращает текущий уровень спам-усталости (0-3).
    Автоматически сбрасывает если истёк TTL.
    """
    level = _spam_fatigue_level.get(username, 0)
    if level == 0:
        return 0
    updated = _spam_fatigue_updated.get(username, 0.0)
    if time.time() - updated > SPAM_FATIGUE_DECAY_TTL:
        _spam_fatigue_level[username] = 0
        return 0
    return level
