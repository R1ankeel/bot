"""
database.py — SQLite-хранилище: история чата, общий лог, факты, репутация, профиль.
"""

import sqlite3
import json
import os
import re
from datetime import datetime

from config import LOCAL_TZ_OFFSET_HOURS

DB_FILE = "bot_data.db"

def _tz_shift_sql() -> str:
    sign = "+" if LOCAL_TZ_OFFSET_HOURS >= 0 else ""
    return f"{sign}{LOCAL_TZ_OFFSET_HOURS} hours"


def _local_date_expr(column_name: str = "created_at") -> str:
    """SQLite-выражение для локальной даты по UTC-штампу из БД."""
    return f"date(datetime({column_name}, '{_tz_shift_sql()}'))"


_conn: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def init_db():
    """Создаёт таблицы, если их ещё нет."""
    conn = get_connection()
    conn.executescript("""
        -- Личная история переписки бота с каждым юзером (для контекста LLM)
        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_history_user
            ON chat_history(username);

        -- Общий лог ВСЕХ сообщений в чате (включая разговоры между людьми)
        CREATE TABLE IF NOT EXISTS global_chat (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL,
            text        TEXT    NOT NULL,
            analyzed    INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_global_analyzed
            ON global_chat(analyzed);
        CREATE INDEX IF NOT EXISTS idx_global_user
            ON global_chat(username);

        -- Репутация
        CREATE TABLE IF NOT EXISTS reputation (
            username    TEXT    PRIMARY KEY,
            level       INTEGER NOT NULL DEFAULT 6,
            progress    INTEGER NOT NULL DEFAULT 0,
            is_lover    INTEGER NOT NULL DEFAULT 0,
            updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- Лог изменений репутации
        CREATE TABLE IF NOT EXISTS reputation_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL,
            old_level   INTEGER NOT NULL,
            old_progress INTEGER NOT NULL,
            new_level   INTEGER NOT NULL,
            new_progress INTEGER NOT NULL,
            delta       INTEGER NOT NULL,
            reason      TEXT,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- Факты о пользователях (извлечённые из чата)
        CREATE TABLE IF NOT EXISTS user_facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL,
            fact        TEXT    NOT NULL,
            source      TEXT    NOT NULL DEFAULT 'chat',
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_unique
            ON user_facts(username, fact);

        -- Игнорируемые пользователи
        CREATE TABLE IF NOT EXISTS ignored_users (
            username    TEXT    PRIMARY KEY,
            added_at    TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- Поведенческие ярлыки о пользователях
        CREATE TABLE IF NOT EXISTS user_traits (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL,
            trait       TEXT    NOT NULL,
            source      TEXT    NOT NULL DEFAULT 'chat',
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_traits_unique
            ON user_traits(username, trait);

        -- Дневник планеты
        CREATE TABLE IF NOT EXISTS planet_diary (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            entry       TEXT    NOT NULL,
            source      TEXT    NOT NULL DEFAULT 'chat',
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- Достижения пользователей
        CREATE TABLE IF NOT EXISTS user_achievements (
            username          TEXT    NOT NULL,
            achievement_code  TEXT    NOT NULL,
            created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (username, achievement_code)
        );

        -- Ежедневные вердикты в общий чат
        CREATE TABLE IF NOT EXISTS daily_reports (
            report_date  TEXT PRIMARY KEY,
            message      TEXT NOT NULL,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Канонический профиль пользователя (структурированные поля)
        CREATE TABLE IF NOT EXISTS user_profile (
            username        TEXT    PRIMARY KEY,
            display_name    TEXT,           -- предпочитаемое обращение
            real_name       TEXT,           -- настоящее имя если известно
            gender          TEXT,           -- 'male' / 'female' / 'unknown'
            age             INTEGER,        -- возраст
            city            TEXT,           -- город
            interests       TEXT,           -- JSON-список интересов
            updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()

    _migrate_tables()


def _migrate_tables():
    """Добавляет недостающие колонки в старые таблицы."""
    conn = get_connection()
    try:
        conn.execute("ALTER TABLE user_facts ADD COLUMN source TEXT NOT NULL DEFAULT 'chat'")
        print("Миграция: добавлена колонка source в user_facts")
    except sqlite3.OperationalError:
        pass  # уже существует
    conn.commit()


# ═══════════════════════════════════════════════════════════════
#  ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ (канонические структурированные данные)
# ═══════════════════════════════════════════════════════════════

def get_user_profile(username: str) -> dict:
    """Возвращает канонический профиль пользователя или пустой dict."""
    conn = get_connection()
    row = conn.execute(
        "SELECT display_name, real_name, gender, age, city, interests "
        "FROM user_profile WHERE username = ?",
        (username,),
    ).fetchone()
    if row is None:
        return {}
    interests = None
    if row["interests"]:
        try:
            interests = json.loads(row["interests"])
        except Exception:
            interests = [row["interests"]]
    return {
        "display_name": row["display_name"],
        "real_name":    row["real_name"],
        "gender":       row["gender"],
        "age":          row["age"],
        "city":         row["city"],
        "interests":    interests,
    }


def update_user_profile_field(username: str, field: str, value) -> bool:
    """Обновляет одно поле профиля. Создаёт запись если её нет.

    value=None — явно обнуляет поле (нужно для !забудь ник).
    Возвращает True если операция выполнена.
    """
    valid_fields = {"display_name", "real_name", "gender", "age", "city", "interests"}
    if field not in valid_fields:
        return False

    conn = get_connection()

    if value is None:
        # Явное обнуление — UPDATE только если запись уже существует
        conn.execute(
            f"UPDATE user_profile SET {field} = NULL, updated_at = datetime('now') WHERE username = ?",
            (username,),
        )
        conn.commit()
        return True

    # Для интересов — сериализуем список
    if field == "interests" and isinstance(value, list):
        value = json.dumps(value, ensure_ascii=False)

    conn.execute(
        f"""INSERT INTO user_profile (username, {field}, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(username) DO UPDATE
            SET {field} = excluded.{field},
                updated_at = excluded.updated_at""",
        (username, value),
    )
    conn.commit()
    return True


# ── Нормализация городов (падежная форма → именительный падеж) ────────
# Покрывает ~50 крупнейших городов России/СНГ.
# Для городов вне списка — применяем простую эвристику.

_CITY_NORM: dict[str, str] = {
    "москве": "Москва",        "питере": "Питер",
    "петербурге": "Петербург", "санкт-петербурге": "Санкт-Петербург",
    "екатеринбурге": "Екатеринбург", "екатеринбурга": "Екатеринбург",
    "новосибирске": "Новосибирск",   "казани": "Казань",
    "нижнем новгороде": "Нижний Новгород", "самаре": "Самара",
    "омске": "Омск",           "челябинске": "Челябинск",
    "ростове": "Ростов",       "уфе": "Уфа",
    "волгограде": "Волгоград", "перми": "Пермь",
    "красноярске": "Красноярск", "воронеже": "Воронеж",
    "краснодаре": "Краснодар", "саратове": "Саратов",
    "тюмени": "Тюмень",        "ижевске": "Ижевск",
    "барнауле": "Барнаул",     "владивостоке": "Владивосток",
    "иркутске": "Иркутск",     "хабаровске": "Хабаровск",
    "ярославле": "Ярославль",  "томске": "Томск",
    "оренбурге": "Оренбург",   "кемерово": "Кемерово",
    "новокузнецке": "Новокузнецк", "рязани": "Рязань",
    "астрахани": "Астрахань",  "пензе": "Пенза",
    "липецке": "Липецк",       "тольятти": "Тольятти",
    "махачкале": "Махачкала",  "киеве": "Киев",
    "харькове": "Харьков",     "минске": "Минск",
    "алматы": "Алматы",        "астане": "Астана",
    "ташкенте": "Ташкент",     "бишкеке": "Бишкек",
    "баку": "Баку",            "тбилиси": "Тбилиси",
    "ереване": "Ереван",       "кишинёве": "Кишинёв",
    "риге": "Рига",            "таллине": "Таллин",
    "вильнюсе": "Вильнюс",     "варшаве": "Варшава",
}


def _normalize_city(raw: str) -> str:
    """Приводит город из падежной формы в именительный падеж."""
    key = raw.strip().lower()
    if key in _CITY_NORM:
        return _CITY_NORM[key]
    # Эвристика для городов не в словаре:
    # 1. Предложный падеж: окончание "е" после мягкого/твёрдого согласного → убираем
    #    (Томске → Томск, Воронеже → Воронеж)
    # 2. Родительный/дат. жен. рода: "и" → заменяем на "ь" (Рязани → Рязань)
    #    но только если перед "и" стоит согласная + "н"
    city = raw.strip()
    if city.endswith("е") and len(city) > 3:
        candidate = city[:-1]
        # Проверяем что не обрезали гласную (не "Самаре" → "Самар", а "Воронеже" → "Воронеж")
        if candidate and candidate[-1].lower() not in "аеёиоуыэюя":
            return candidate
    if city.lower().endswith("ни") and len(city) > 4:
        return city[:-1] + "ь"   # Казани → Казань
    return city


# ── Защита от инъекций в поля профиля ────────────────────────────────

_PROFILE_INJECTION_PATTERNS = [
    r"(?:system|system prompt|prompt|инструкция|игнорир|забудь|ты|ты теперь)",
    r"\[",   # скобки характерны для injection
    r"<[a-z]",   # HTML/XML теги
]


def _is_safe_profile_value(value: str) -> bool:
    """Проверяет что значение профиля не является попыткой инъекции."""
    v = value.lower().strip()
    # Слишком длинное — подозрительно
    if len(v) > 40:
        return False
    # Содержит подозрительные паттерны
    for pattern in _PROFILE_INJECTION_PATTERNS:
        if re.search(pattern, v, re.IGNORECASE):
            return False
    # Имя/ник должен быть одним словом из букв
    if re.search(r"[^а-яёa-z0-9\-\s_]", v):
        return False
    return True


def extract_and_update_profile_from_facts(username: str):
    """Пробует извлечь структурированные поля из user_facts и сохранить в user_profile.

    Вызывается после добавления новых фактов (в analyze_facts_from_chat).
    Не перезаписывает поля которые уже явно заданы, если новое значение не лучше.
    """
    facts = get_facts(username, limit=60)
    if not facts:
        return

    combined = " ".join(facts)
    combined_lower = combined.lower()

    # ── Возраст ───────────────────────────────────────────────
    # Только из прямого заявления самого человека: "мне N лет"
    # НЕ из "дал бы 17", "выглядит на 25", "ему 30 лет"
    age_match = re.search(r'мне\s+(\d{1,2})\s*(?:лет|год(?:а|ик)?)\b', combined_lower)
    if age_match:
        age = int(age_match.group(1))
        if 5 <= age <= 90:
            update_user_profile_field(username, "age", age)

    # ── Имя ───────────────────────────────────────────────────
    # Только из явного самозаявления: "меня зовут X" / "моё имя X"
    # НЕ из упоминаний имён в контексте разговора
    name_match = re.search(
        r'(?:меня зовут|зовут меня|моё имя|мое имя)\s+([А-ЯЁ][а-яё]{2,15})',
        combined,
    )
    if name_match:
        update_user_profile_field(username, "real_name", name_match.group(1))

    # ── Город ─────────────────────────────────────────────────
    city_match = re.search(
        r'(?:живёт?|живу|из|приехал(?:а)?\s+из|город[:\s]+)\s+(?:в\s+)?([А-ЯЁ][а-яё]+(?:[\-\s][А-ЯЁ][а-яё]+)?)',
        combined,
    )
    if city_match:
        city = _normalize_city(city_match.group(1).strip())
        # Исключаем служебные слова
        if city.lower() not in {"этого", "нашего", "своего", "того", "что", "как", "где"}:
            update_user_profile_field(username, "city", city)

    # ── Пол ───────────────────────────────────────────────────
    # Только если в профиле ещё нет
    profile = get_user_profile(username)
    if not profile.get("gender") or profile["gender"] == "unknown":
        gender = get_gender_fact(username)
        if gender:
            update_user_profile_field(username, "gender", gender)
    # Чистим факты о поле которые противоречат профилю
    clean_conflicting_gender_facts(username)

    # ── Интересы ─────────────────────────────────────────────
    # Накапливаем из трейтов — они уже содержат категории интересов
    traits = get_traits(username, limit=30)
    interest_keywords = [
        "музык", "игр", "аниме", "рисова", "программир", "спорт",
        "кино", "книг", "путешест", "готовк", "дизайн", "фото",
        "стрим", "танц", "пев", "читает", "смотрит",
    ]
    found_interests = []
    for trait in traits:
        tl = trait.lower()
        for kw in interest_keywords:
            if kw in tl and trait not in found_interests:
                found_interests.append(trait)
                break
    if found_interests:
        # Обновляем только если список непустой
        existing = profile.get("interests") or []
        merged = list({i.lower() for i in existing + found_interests})[:6]
        if merged:
            update_user_profile_field(username, "interests", merged)


def update_profile_realtime(username: str, message: str) -> list[str]:
    """Немедленно обновляет профиль из прямого сообщения пользователя боту.

    В отличие от extract_and_update_profile_from_facts (batch),
    эта функция парсит конкретное сообщение и обновляет профиль мгновенно.
    Возвращает список обновлённых полей для лога.

    Обрабатывает:
      «называй меня X» / «зови меня X» → display_name
      «меня зовут X»                    → real_name
      «мне N лет»                       → age (с конфликт-резолюцией: перезаписывает)
      «я из X» / «живу в X»             → city
      «я девушка» / «я парень»           → gender
    """
    updated: list[str] = []

    # ── display_name (предпочитаемое обращение) ───────────────
    dn_match = re.search(
        r'(?:называй меня|зови меня|обращайся ко мне|я предпочитаю)\s+([А-ЯЁA-Za-zа-яё]{2,20})',
        message, re.IGNORECASE,
    )
    if dn_match:
        name = dn_match.group(1).capitalize()
        if _is_safe_profile_value(name):
            update_user_profile_field(username, "display_name", name)
            updated.append(f"display_name={name}")
        else:
            print(f"  [ПРОФИЛЬ RT] {username}: отклонена инъекция display_name={name!r}")

    # ── real_name ─────────────────────────────────────────────
    rn_match = re.search(
        r'(?:меня зовут|моё имя|мое имя)\s+([А-ЯЁ][а-яё]{2,15})',
        message, re.IGNORECASE,
    )
    if rn_match:
        name = rn_match.group(1)
        if _is_safe_profile_value(name):
            update_user_profile_field(username, "real_name", name)
            updated.append(f"real_name={name}")

    # ── age (конфликт-резолюция: прямое сообщение всегда свежее фактов) ──
    age_match = re.search(r'\bмне\s+(\d{1,2})\s*(?:лет|год(?:а|ик)?)\b', message.lower())
    if age_match:
        age = int(age_match.group(1))
        if 5 <= age <= 90:
            update_user_profile_field(username, "age", age)
            updated.append(f"age={age}")
            # Убираем старые конфликтующие факты о возрасте
            deduplicate_age_facts(username)

    # ── city (с нормализацией + поддержка составных: Нижний Новгород, Санкт-Петербург) ──
    city_match = re.search(
        r'(?:я из|живу в|нахожусь в|переехал(?:а)?\s+в|я в городе|город[:\s]+)\s*'
        r'([А-ЯЁ][а-яё\-]{2,20}(?:[\s\-][А-ЯЁ][а-яё\-]{2,20})?)',
        message, re.IGNORECASE,
    )
    if city_match:
        city_raw = city_match.group(1)
        city = _normalize_city(city_raw)
        if _is_safe_profile_value(city):
            update_user_profile_field(username, "city", city)
            updated.append(f"city={city}")
            # Убираем старые конфликтующие факты о городе
            deduplicate_city_facts(username)

    # ── gender ────────────────────────────────────────────────
    msg_l = message.lower()
    if re.search(r'\bя\s+(?:девушка|девочка|женщина)\b', msg_l):
        update_user_profile_field(username, "gender", "female")
        updated.append("gender=female")
    elif re.search(r'\bя\s+(?:парень|мужчина|мальчик|чувак)\b', msg_l):
        update_user_profile_field(username, "gender", "male")
        updated.append("gender=male")

    if updated:
        print(f"  [ПРОФИЛЬ RT] {username}: {', '.join(updated)}")
    return updated


# ═══════════════════════════════════════════════════════════════
#  ЛИЧНАЯ ИСТОРИЯ (переписка бота с юзером)
# ═══════════════════════════════════════════════════════════════


def save_message(username: str, role: str, content: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_history (username, role, content) VALUES (?, ?, ?)",
        (username, role, content),
    )
    conn.commit()


def get_history(username: str, limit: int = 500) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE username = ? ORDER BY id DESC LIMIT ?",
        (username, limit),
    ).fetchall()

    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_last_user_messages(username: str, limit: int = 10) -> list[str]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT content FROM chat_history WHERE username = ? AND role = 'user' ORDER BY id DESC LIMIT ?",
        (username, limit),
    ).fetchall()

    return [r["content"] for r in reversed(rows)]


def get_history_hybrid(username: str, recent_turns: int = 8, anchor_turns: int = 2) -> list[dict]:
    """Гибридная история диалога: последние recent_turns ходов + anchor_turns самых ранних.

    Помогает боту помнить начало знакомства, не теряя текущий контекст.
    Одна пара user+assistant = один «ход».

    История передаётся в LLM в чистом виде:
    - служебные префиксы «Ник: X\\nСообщение:» убираются
    - ник добавляется только к первому user-turn окна (контекст без шума)
    """
    conn = get_connection()
    total = conn.execute(
        "SELECT COUNT(*) AS cnt FROM chat_history WHERE username = ?",
        (username,),
    ).fetchone()["cnt"]

    recent_limit = recent_turns * 2

    if total <= recent_limit + anchor_turns * 2:
        rows = conn.execute(
            "SELECT role, content FROM chat_history WHERE username = ? ORDER BY id ASC",
            (username,),
        ).fetchall()
        raw = [{"role": r["role"], "content": r["content"]} for r in rows]
        return _clean_history_for_llm(raw, username)

    anchor_rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE username = ? ORDER BY id ASC LIMIT ?",
        (username, anchor_turns * 2),
    ).fetchall()
    anchor_msgs = [{"role": r["role"], "content": r["content"]} for r in anchor_rows]

    recent_rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE username = ? ORDER BY id DESC LIMIT ?",
        (username, recent_limit),
    ).fetchall()
    recent_msgs = [{"role": r["role"], "content": r["content"]} for r in reversed(recent_rows)]

    separator = {"role": "system", "content": "[ … предыдущие сообщения опущены … ]"}
    raw = anchor_msgs + [separator] + recent_msgs
    return _clean_history_for_llm(raw, username)


# ── Очистка истории для LLM ────────────────────────────────────────────

def _strip_user_prefix(content: str) -> str:
    """Убирает служебные префиксы из user-сообщений.

    Форматы которые убираются:
      «Ник: username\\nСообщение: text»  → text
      «Ник: username\\nКоманда: text»    → text
      «Сообщение: text»                  → text
    """
    # Убираем строку «Ник: ...» если она в начале
    content = re.sub(r'^Ник:\s*\S+\s*\n', '', content)
    # Убираем метку «Сообщение:» / «Команда:»
    content = re.sub(r'^(?:Сообщение|Команда):\s*', '', content)
    return content.strip()


def _clean_history_for_llm(history: list[dict], username: str) -> list[dict]:
    """Форматирует историю для передачи в LLM:
    - убирает служебные префиксы из user-сообщений
    - добавляет ник только к первому user-turn (один раз, для контекста)
    - system-разделители пропускает как есть
    """
    result: list[dict] = []
    nick_added = False
    for msg in history:
        if msg["role"] == "system":
            result.append(msg)
            continue
        if msg["role"] == "user":
            clean = _strip_user_prefix(msg["content"])
            if not nick_added and username and clean:
                # Ник только в первом user-turn чтобы модель знала с кем говорит
                clean = f"[{username}]: {clean}"
                nick_added = True
            result.append({"role": "user", "content": clean})
        else:
            result.append({"role": msg["role"], "content": msg["content"]})
    return result


# ═══════════════════════════════════════════════════════════════
#  ОБЩИЙ ЛОГ ЧАТА (все сообщения всех людей)
# ═══════════════════════════════════════════════════════════════


_unanalyzed_count: int | None = None


def _ensure_counter():
    global _unanalyzed_count
    if _unanalyzed_count is None:
        conn = get_connection()
        row = conn.execute("SELECT COUNT(*) as cnt FROM global_chat WHERE analyzed = 0").fetchone()
        _unanalyzed_count = row["cnt"]


def log_global_message(username: str, text: str):
    global _unanalyzed_count
    _ensure_counter()
    conn = get_connection()
    conn.execute(
        "INSERT INTO global_chat (username, text) VALUES (?, ?)",
        (username, text),
    )
    conn.commit()
    _unanalyzed_count += 1


def get_unanalyzed_messages(limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, username, text FROM global_chat WHERE analyzed = 0 ORDER BY id ASC LIMIT ?",
        (limit,),
    ).fetchall()

    return [{"id": r["id"], "username": r["username"], "text": r["text"]} for r in rows]


def mark_messages_analyzed(ids: list[int]):
    global _unanalyzed_count
    if not ids:
        return
    conn = get_connection()
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"UPDATE global_chat SET analyzed = 1 WHERE id IN ({placeholders})",
        ids,
    )
    conn.commit()
    _ensure_counter()
    _unanalyzed_count = max(0, _unanalyzed_count - len(ids))


def get_global_message_count() -> int:
    _ensure_counter()
    return _unanalyzed_count


# ═══════════════════════════════════════════════════════════════
#  ФАКТЫ О ПОЛЬЗОВАТЕЛЯХ
# ═══════════════════════════════════════════════════════════════


def add_fact(username: str, fact: str, source: str = "chat"):
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO user_facts (username, fact, source) VALUES (?, ?, ?)",
        (username, fact, source),
    )
    conn.commit()


def add_facts_bulk(facts: list[tuple[str, str]], source: str = "chat") -> int:
    if not facts:
        return 0
    conn = get_connection()
    rows = [(username, fact, source) for username, fact in facts]
    added = 0
    for row in rows:
        cur = conn.execute(
            "INSERT OR IGNORE INTO user_facts (username, fact, source) VALUES (?, ?, ?)",
            row,
        )
        added += cur.rowcount
    conn.commit()
    return added


def get_facts(username: str, limit: int = 30) -> list[str]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT fact FROM user_facts WHERE username = ? ORDER BY id DESC LIMIT ?",
        (username, limit),
    ).fetchall()

    return [r["fact"] for r in reversed(rows)]


# Стоп-слова для русского языка (игнорируем при поиске релевантности)
_RU_STOPWORDS = {
    "и", "в", "на", "с", "что", "как", "это", "не", "а", "но", "да", "ну",
    "у", "к", "по", "из", "он", "она", "они", "ты", "я", "мы", "вы", "о",
    "за", "от", "до", "при", "об", "про", "для", "без", "под", "над",
    "или", "то", "же", "ли", "бы", "уж", "ещё", "вот", "тут", "там",
}

RELATION_KEYWORDS = ["люб", "нрав", "ревн", "флирт", "парень", "девушка", "отнош"]
MOOD_KEYWORDS = ["груст", "зл", "обид", "устал", "рад", "весел"]
IDENTITY_KEYWORDS = ["имя", "зовут", "лет", "город", "живу", "из"]


def _extract_stems(text: str) -> set[str]:
    raw_words = set(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())
    raw_words -= _RU_STOPWORDS
    return {w[:3] for w in raw_words if len(w) >= 3}


def is_salient_topic(message: str) -> bool:
    """True, если запрос затрагивает личные/эмоциональные темы для memory salience."""
    t = (message or "").lower().strip()
    if not t:
        return False
    direct_markers = (
        "что знаешь", "что ты знаешь", "кто такой", "кто такая", "кто он", "кто она",
        "отношен", "ревну", "флирт", "настроени", "груст", "обид", "злю", "злой",
        "как тебя зовут", "как зовут", "сколько лет", "из какого города", "ты из",
        "меня зовут", "мне ", "я из ", "живу в ",
    )
    if any(marker in t for marker in direct_markers):
        return True
    stems = _extract_stems(t)
    boosted = set(RELATION_KEYWORDS + MOOD_KEYWORDS + IDENTITY_KEYWORDS)
    return any(any(stem.startswith(kw[:3]) for stem in stems) for kw in boosted)


def get_facts_salient(username: str, query: str = "", limit: int = 8) -> list[str]:
    """Возвращает факты пользователя, отсортированные по релевантности к query.

    Использует 3-символьные стемы для учёта русской морфологии:
    «игры», «играешь», «играть» → все дают стем «игр» → матч.
    При равной релевантности — более свежие факты приоритетнее (recency bias).
    Если query пустой — возвращает последние limit фактов.
    """
    all_facts = get_facts(username, limit=50)
    if not all_facts:
        return []
    if not query:
        return all_facts[-limit:]

    query_stems = _extract_stems(query)

    if not query_stems:
        return all_facts[-limit:]

    boost_stems = {k[:3] for k in (RELATION_KEYWORDS + MOOD_KEYWORDS + IDENTITY_KEYWORDS)}

    def _relevance(idx: int, fact_text: str) -> float:
        fact_stems = _extract_stems(fact_text)
        if not fact_stems:
            return 0.0
        overlap = len(query_stems & fact_stems)
        score = overlap / max(len(query_stems), 1)
        # Semantic boost: темы отношений/эмоций/идентичности важнее в диалоге.
        boost_overlap = len((query_stems & boost_stems) & fact_stems)
        score += boost_overlap * 0.25
        # Recency: index 0 = самый старый, последний = самый свежий
        recency = idx / max(len(all_facts), 1) * 0.15
        return score + recency

    scored = sorted(enumerate(all_facts), key=lambda x: _relevance(x[0], x[1]), reverse=True)
    return [fact for _, fact in scored[:limit]]


def get_gender_fact(username: str) -> str:
    """Ищет факт о поле пользователя. Возвращает 'female', 'male' или ''."""
    conn = get_connection()
    row = conn.execute(
        "SELECT fact FROM user_facts WHERE username = ? AND ("
        "  fact LIKE '%пол: жен%' OR fact LIKE '%пол: муж%'"
        ") ORDER BY id DESC LIMIT 1",
        (username,),
    ).fetchone()
    if row:
        f = row["fact"].lower()
        if "жен" in f:
            return "female"
        if "муж" in f:
            return "male"

    female_like = ["%девушк%", "%женщин%", "%подруга%"]
    male_like   = ["%парень%", "%мужчин%", "%мальчик%"]

    for pattern in female_like:
        row = conn.execute(
            "SELECT 1 FROM user_facts WHERE username = ? AND fact LIKE ? LIMIT 1",
            (username, pattern),
        ).fetchone()
        if row:
            return "female"

    for pattern in male_like:
        row = conn.execute(
            "SELECT 1 FROM user_facts WHERE username = ? AND fact LIKE ? LIMIT 1",
            (username, pattern),
        ).fetchone()
        if row:
            return "male"

    return ""


def add_traits_bulk(traits: list[tuple[str, str]], source: str = "chat") -> int:
    if not traits:
        return 0
    conn = get_connection()
    rows = [(username, trait, source) for username, trait in traits]
    added = 0
    for row in rows:
        cur = conn.execute(
            "INSERT OR IGNORE INTO user_traits (username, trait, source) VALUES (?, ?, ?)",
            row,
        )
        added += cur.rowcount
    conn.commit()
    return added


def get_traits(username: str, limit: int = 20) -> list[str]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT trait FROM user_traits WHERE username = ? ORDER BY id DESC LIMIT ?",
        (username, limit),
    ).fetchall()
    return [r["trait"] for r in reversed(rows)]


def get_traits_prioritized(username: str, limit: int = 10, query: str = "") -> list[str]:
    """Возвращает ярлыки с приоритетом self_declared (что сам сказал о себе).

    self_declared идут первыми — они надёжнее observed (наблюдений бота).
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT id, trait, source FROM user_traits WHERE username = ?
           ORDER BY id DESC
           LIMIT 40""",
        (username,),
    ).fetchall()
    if not rows:
        return []

    if not query:
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                0 if r["source"] == "self_declared" else 1 if r["source"] == "observed" else 2,
                -r["id"],
            ),
        )
        return [r["trait"] for r in rows_sorted[:limit]]

    query_stems = _extract_stems(query)
    boost_stems = {k[:3] for k in (RELATION_KEYWORDS + MOOD_KEYWORDS + IDENTITY_KEYWORDS)}

    def _trait_score(row: sqlite3.Row) -> float:
        source_weight = 1.0 if row["source"] == "self_declared" else 0.7 if row["source"] == "observed" else 0.4
        trait_stems = _extract_stems(row["trait"])
        overlap = len(query_stems & trait_stems) / max(len(query_stems), 1) if query_stems else 0.0
        boost_overlap = len((query_stems & boost_stems) & trait_stems)
        recency = row["id"] / max(rows[0]["id"], 1) * 0.1
        return source_weight + overlap + (boost_overlap * 0.25) + recency

    ranked = sorted(rows, key=_trait_score, reverse=True)
    return [r["trait"] for r in ranked[:limit]]


def get_recent_global_messages(limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, username, text, created_at FROM global_chat ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {"id": r["id"], "username": r["username"], "text": r["text"], "created_at": r["created_at"]}
        for r in reversed(rows)
    ]


def get_global_messages_after(created_at: str, limit: int = 120) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, username, text, created_at FROM global_chat WHERE created_at > ? ORDER BY id ASC LIMIT ?",
        (created_at, limit),
    ).fetchall()
    return [
        {"id": r["id"], "username": r["username"], "text": r["text"], "created_at": r["created_at"]}
        for r in rows
    ]


def add_diary_entry(entry: str, source: str = "chat") -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO planet_diary (entry, source) VALUES (?, ?)",
        (entry, source),
    )
    conn.commit()
    return cur.lastrowid


def get_recent_diary_entries(limit: int = 10) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, entry, source, created_at FROM planet_diary ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {"id": r["id"], "entry": r["entry"], "source": r["source"], "created_at": r["created_at"]}
        for r in reversed(rows)
    ]

def get_diary_entries_for_date(report_date: str, limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        f"SELECT id, entry, source, created_at FROM planet_diary WHERE {_local_date_expr('created_at')} = ? ORDER BY id ASC LIMIT ?",
        (report_date, limit),
    ).fetchall()
    return [
        {"id": r["id"], "entry": r["entry"], "source": r["source"], "created_at": r["created_at"]}
        for r in rows
    ]


def get_latest_diary_timestamp() -> str:
    conn = get_connection()
    row = conn.execute(
        "SELECT created_at FROM planet_diary ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["created_at"] if row else ""


def get_all_facts_summary() -> dict[str, list[str]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT username, fact FROM user_facts ORDER BY username, id",
    ).fetchall()

    result: dict[str, list[str]] = {}
    for r in rows:
        result.setdefault(r["username"], []).append(r["fact"])
    return result


def clear_facts(username: str):
    conn = get_connection()
    conn.execute("DELETE FROM user_facts WHERE username = ?", (username,))
    conn.commit()


def clear_traits(username: str):
    conn = get_connection()
    conn.execute("DELETE FROM user_traits WHERE username = ?", (username,))
    conn.commit()


def clear_user_profile(username: str):
    """Сбрасывает структурированный профиль пользователя."""
    conn = get_connection()
    conn.execute("DELETE FROM user_profile WHERE username = ?", (username,))
    conn.commit()


def delete_facts_containing(username: str, keyword: str) -> int:
    """Удаляет факты содержащие keyword (регистронезависимо).

    Возвращает количество удалённых записей.
    """
    conn = get_connection()
    cur = conn.execute(
        "DELETE FROM user_facts WHERE username = ? AND lower(fact) LIKE ?",
        (username, f"%{keyword.lower()}%"),
    )
    conn.commit()
    return cur.rowcount


def deduplicate_age_facts(username: str) -> int:
    """Оставляет только самый свежий факт с возрастом, удаляет старые.

    Вызывается когда добавляется новый факт с возрастом.
    Возвращает количество удалённых дубликатов.
    """
    conn = get_connection()
    # Ищем все факты с числом + «лет/год»
    rows = conn.execute(
        "SELECT id, fact FROM user_facts WHERE username = ? "
        "AND (lower(fact) LIKE '%лет%' OR lower(fact) LIKE '%год%' OR lower(fact) LIKE '%года%') "
        "ORDER BY id DESC",
        (username,),
    ).fetchall()

    age_facts = [(r["id"], r["fact"]) for r in rows
                 if re.search(r'\b\d{1,2}\s*(?:лет|год)', r["fact"].lower())]

    if len(age_facts) <= 1:
        return 0

    # Оставляем только самый свежий (первый в списке — наибольший id)
    keep_id = age_facts[0][0]
    delete_ids = [fid for fid, _ in age_facts[1:]]
    if delete_ids:
        placeholders = ",".join("?" for _ in delete_ids)
        conn.execute(f"DELETE FROM user_facts WHERE id IN ({placeholders})", delete_ids)
        conn.commit()
        print(f"[ДЕДУП] {username}: удалено {len(delete_ids)} старых фактов о возрасте")
    return len(delete_ids)


def deduplicate_city_facts(username: str) -> int:
    """Оставляет только самый свежий факт с городом, удаляет старые.

    Использует широкий regex для обнаружения любых упоминаний города,
    а не узкий список конкретных городов.
    """
    conn = get_connection()
    # Ищем все факты где есть типичные фразы о месте жительства
    rows = conn.execute(
        "SELECT id, fact FROM user_facts WHERE username = ? ORDER BY id DESC",
        (username,),
    ).fetchall()

    city_ids = []
    # Широкий паттерн: живёт/живу/из/переехал + любой город
    _city_re = re.compile(
        r'(?:живёт?|живет?|живу|из\s+\w|переехал|нахожусь|родом из|я из)',
        re.IGNORECASE,
    )
    for row in rows:
        if _city_re.search(row["fact"]):
            city_ids.append(row["id"])

    if len(city_ids) <= 1:
        return 0

    # Оставляем самый свежий (уже sorted DESC)
    delete_ids = city_ids[1:]
    if delete_ids:
        placeholders = ",".join("?" for _ in delete_ids)
        conn.execute(f"DELETE FROM user_facts WHERE id IN ({placeholders})", delete_ids)
        conn.commit()
        print(f"[ДЕДУП] {username}: удалено {len(delete_ids)} старых фактов о городе")
    return len(delete_ids)


def clean_conflicting_gender_facts(username: str) -> int:
    """Удаляет факты о поле которые противоречат канонному профилю.

    Если профиль говорит gender=female, а в фактах лежит 'пол: мужской' —
    reasoning-модель читает факт и игнорирует гендер-блок. Эта функция
    удаляет весь мусор о поле, оставляя профиль единственным источником правды.
    Вызывается из analyze_facts_from_chat вместе с deduplicate_age/city.
    """
    profile = get_user_profile(username)
    canonical = profile.get("gender", "")
    if not canonical or canonical == "unknown":
        return 0  # не знаем правду — не трогаем

    conn = get_connection()
    rows = conn.execute(
        "SELECT id, fact FROM user_facts WHERE username = ? ORDER BY id DESC",
        (username,),
    ).fetchall()

    _gender_re = re.compile(
        r'пол[:\s]|\bдевушк|\bженщин|\bпарень|\bпарня|\bмужчин|\bмальчик',
        re.IGNORECASE,
    )

    to_delete = []
    for row in rows:
        f = row["fact"]
        if not _gender_re.search(f):
            continue
        # Определяем что говорит этот факт о поле
        f_lower = f.lower()
        fact_gender = ""
        if "жен" in f_lower or "девушк" in f_lower:
            fact_gender = "female"
        elif "муж" in f_lower or "парень" in f_lower or "парня" in f_lower or "мальчик" in f_lower:
            fact_gender = "male"

        if fact_gender and fact_gender != canonical:
            to_delete.append(row["id"])

    if to_delete:
        placeholders = ",".join("?" for _ in to_delete)
        conn.execute(f"DELETE FROM user_facts WHERE id IN ({placeholders})", to_delete)
        conn.commit()
        print(f"[ГЕНДЕР-ДЕДУП] {username}: удалено {len(to_delete)} конфликтующих фактов о поле (профиль: {canonical})")
    return len(to_delete)


def clean_profile_redundant_facts(username: str) -> int:
    """Удаляет из user_facts записи, которые дублируют поля user_profile.

    Конкретно: если профиль уже содержит gender или real_name/display_name —
    удаляем все факты вида 'пол: ...' и 'имя: ...' / 'меня зовут ...' для этого ника.
    Это устраняет главную причину «скатывания в мужской пол»: старые мусорные факты
    пересиливают гендерный блок в промпте.
    Возвращает количество удалённых строк.
    """
    profile = get_user_profile(username)
    conn = get_connection()
    to_delete: list[int] = []

    rows = conn.execute(
        "SELECT id, fact FROM user_facts WHERE username = ?",
        (username,),
    ).fetchall()

    canonical_gender = profile.get("gender", "")
    has_name = bool(profile.get("real_name") or profile.get("display_name"))

    _gender_fact_re = re.compile(r'пол\s*[:：]', re.IGNORECASE)
    _name_fact_re   = re.compile(
        r'(?:имя\s*[:：]|меня зовут|моё имя|мое имя)',
        re.IGNORECASE,
    )

    for row in rows:
        fid, fact = row["id"], row["fact"]
        # Гендерный факт — удаляем если профиль уже знает пол
        if _gender_fact_re.search(fact) and canonical_gender and canonical_gender != "unknown":
            to_delete.append(fid)
            continue
        # Факт об имени — удаляем если профиль уже знает имя
        if _name_fact_re.search(fact) and has_name:
            to_delete.append(fid)
            continue

    if to_delete:
        placeholders = ",".join("?" for _ in to_delete)
        conn.execute(f"DELETE FROM user_facts WHERE id IN ({placeholders})", to_delete)
        conn.commit()
        print(
            f"[ПРОФИЛЬ-ДЕДУП] {username}: удалено {len(to_delete)} факт(ов) "
            f"дублирующих профиль (пол/имя)"
        )
    return len(to_delete)


def clean_all_profile_redundant_facts() -> int:
    """Разовая глобальная чистка: удаляет пол/имя-факты для всех юзеров у кого есть профиль.

    Вызывать один раз при старте или руками чтобы почистить накопившийся мусор в БД.
    Возвращает суммарное количество удалённых строк.
    """
    conn = get_connection()
    usernames = [
        r[0] for r in conn.execute(
            "SELECT username FROM user_profile WHERE "
            "(gender IS NOT NULL AND gender != '' AND gender != 'unknown') "
            "OR real_name IS NOT NULL OR display_name IS NOT NULL"
        ).fetchall()
    ]
    total = 0
    for username in usernames:
        total += clean_profile_redundant_facts(username)
    if total:
        print(f"[ПРОФИЛЬ-ДЕДУП] Глобальная чистка завершена: удалено {total} записей")
    return total


def get_global_user_message_count(username: str) -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM global_chat WHERE username = ?",
        (username,),
    ).fetchone()
    return row["cnt"] if row else 0


def get_direct_user_message_count(username: str) -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM chat_history WHERE username = ? AND role = 'user'",
        (username,),
    ).fetchone()
    return row["cnt"] if row else 0


def get_user_fact_count(username: str) -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM user_facts WHERE username = ?",
        (username,),
    ).fetchone()
    return row["cnt"] if row else 0


def get_user_trait_count(username: str) -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM user_traits WHERE username = ?",
        (username,),
    ).fetchone()
    return row["cnt"] if row else 0


def add_user_achievement(username: str, achievement_code: str) -> bool:
    conn = get_connection()
    cur = conn.execute(
        "INSERT OR IGNORE INTO user_achievements (username, achievement_code) VALUES (?, ?)",
        (username, achievement_code),
    )
    conn.commit()
    return cur.rowcount > 0


def get_user_achievement_codes(username: str) -> list[str]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT achievement_code FROM user_achievements WHERE username = ? ORDER BY created_at ASC",
        (username,),
    ).fetchall()
    return [r["achievement_code"] for r in rows]


def get_global_messages_for_date(report_date: str, limit: int = 200) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        f"SELECT id, username, text, created_at FROM global_chat WHERE {_local_date_expr('created_at')} = ? ORDER BY id ASC LIMIT ?",
        (report_date, limit),
    ).fetchall()
    return [
        {"id": r["id"], "username": r["username"], "text": r["text"], "created_at": r["created_at"]}
        for r in rows
    ]


def has_daily_report(report_date: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM daily_reports WHERE report_date = ? LIMIT 1",
        (report_date,),
    ).fetchone()
    return row is not None


def save_daily_report(report_date: str, message: str):
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO daily_reports (report_date, message) VALUES (?, ?)",
        (report_date, message),
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════════
#  МИГРАЦИЯ ИЗ JSON
# ═══════════════════════════════════════════════════════════════


def migrate_from_json(json_path: str = "chat_history.json"):
    if not os.path.exists(json_path):
        print(f"Файл {json_path} не найден, миграция не нужна.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conn = get_connection()
    count = 0
    for username, messages in data.items():
        for msg in messages:
            conn.execute(
                "INSERT INTO chat_history (username, role, content) VALUES (?, ?, ?)",
                (username, msg["role"], msg["content"]),
            )
            count += 1

    conn.commit()

    print(f"Миграция завершена: {count} сообщений из {len(data)} пользователей.")

    backup = json_path + ".bak"
    os.rename(json_path, backup)
    print(f"Старый файл переименован в {backup}")


# ═══════════════════════════════════════════════════════════════
#  ИГНОРИРУЕМЫЕ ПОЛЬЗОВАТЕЛИ
# ═══════════════════════════════════════════════════════════════

_ignored_cache: set[str] | None = None


def _ensure_ignored_cache():
    global _ignored_cache
    if _ignored_cache is None:
        conn = get_connection()
        rows = conn.execute("SELECT username FROM ignored_users").fetchall()
        _ignored_cache = {r["username"] for r in rows}


def is_ignored(username: str) -> bool:
    _ensure_ignored_cache()
    return username in _ignored_cache


def add_ignored(username: str) -> bool:
    global _ignored_cache
    _ensure_ignored_cache()
    if username in _ignored_cache:
        return False
    conn = get_connection()
    conn.execute("INSERT OR IGNORE INTO ignored_users (username) VALUES (?)", (username,))
    conn.commit()
    _ignored_cache.add(username)
    return True


def remove_ignored(username: str) -> bool:
    global _ignored_cache
    _ensure_ignored_cache()
    if username not in _ignored_cache:
        return False
    conn = get_connection()
    conn.execute("DELETE FROM ignored_users WHERE username = ?", (username,))
    conn.commit()
    _ignored_cache.discard(username)
    return True


def get_ignored_list() -> list[str]:
    _ensure_ignored_cache()
    return sorted(_ignored_cache)


def seed_ignored(usernames: set[str]):
    conn = get_connection()
    for u in usernames:
        conn.execute("INSERT OR IGNORE INTO ignored_users (username) VALUES (?)", (u,))
    conn.commit()
    global _ignored_cache
    _ignored_cache = None
