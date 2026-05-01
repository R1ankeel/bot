"""
utils.py — чистые вспомогательные функции без побочных эффектов.
"""

import random
import re

from config import (
    CHARACTER_NAME,
    FALLBACK_RESPONSE,
    HOOKS,
    INJECTION_PATTERNS,
    MOOD_HOOKS,
    NAME_TRIGGERS,
    NAME_WORD_FORMS,
)


def clean_response(text: str, username: str = "") -> str:
    """Убирает HTML-теги и ник бота из начала ответа.
    Если передан username — восстанавливает оригинальный регистр ника в начале ответа.
    """
    text = re.sub(r'<channel\|([^>]*)>', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    for prefix in [CHARACTER_NAME + ",", CHARACTER_NAME + ":", CHARACTER_NAME]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip(" ,:")
            break

    # Восстанавливаем регистр ника в начале ответа если модель написала строчными
    if username:
        for sep in (", ", ": ", ",", ":"):
            prefix_lower = username.lower() + sep
            if text.lower().startswith(prefix_lower):
                # Заменяем только первое вхождение в начале строки
                text = username + sep + text[len(prefix_lower):]
                break

    return text.strip()


def ensure_nick(text: str, username: str) -> str:
    """Гарантирует, что ответ начинается с ника пользователя.

    ВАЖНО: эта функция НЕ должна «чинить» похожие ники (fuzzy) — этим занимается
    `fix_username_in_response`. Здесь только:
    - если ник уже корректно стоит в начале → нормализуем регистр/разделитель
    - иначе → добавляем `username, ` в начало

    Это защищает от регрессии вида: "Скуф, ск uf, ..." (когда ник уже был,
    но из-за неправильного порядка постобработки мы добавляли ещё один).
    """
    if not username:
        return (text or "").strip()
    t = (text or "").strip()
    if not t:
        return f"{username},"

    # Если уже начинается с корректного ника (в любом регистре/с лишними пробелами),
    # приводим строго к `username, ...` и выходим.
    normalized = normalize_strict_nick_prefix(t, username)
    if normalized != t:
        return normalized

    # Защита: если модель уже начала с ника (но без строгого разделителя),
    # не добавляем ник второй раз — просто нормализуем префикс.
    if t.lower().startswith(username.lower()):
        return normalize_strict_nick_prefix(t, username)

    # Ника нет — добавляем.
    return f"{username}, {t}"


def has_name_trigger(text: str) -> bool:
    t = text.lower()
    if any(trigger in t for trigger in NAME_TRIGGERS):
        return True
    words = re.split(r'\W+', t)
    return any(w in NAME_WORD_FORMS for w in words)


def detect_injection(text: str) -> bool:
    """Проверяет, пытается ли юзер внедрить команду для модели."""
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def pick_hook(mood: str) -> str:
    pool = MOOD_HOOKS.get(mood, HOOKS)
    return random.choice(pool)


def is_weather_query(message: str) -> bool:
    t = (message or "").lower().strip()
    if not t:
        return False
    weather_markers = (
        "погода", "прогноз", "дожд", "снег", "температур", "сколько градусов",
        "жара", "холодно", "ветер", "ливень", "мороз", "осадки",
    )
    weather_patterns = (
        r"что\s+там\s+в\s+[а-яё\-]+\s+по\s+погод",
        r"какая\s+погода\s+в\s+[а-яё\-]+",
        r"погода\s+в\s+[а-яё\-]+",
    )
    return any(marker in t for marker in weather_markers) or any(re.search(p, t) for p in weather_patterns)


def is_school_or_science_query(message: str) -> bool:
    t = (message or "").lower().strip()
    if not t:
        return False
    markers = (
        "квантов", "теория относительности", "физик", "химия", "математ",
        "программирован", "экономик", "политик", "медицин", "юридич",
        "теорем", "интеграл", "энтроп", "бозон", "кварк", "дифференциал",
        "алгоритм", "доказать", "дисперсия", "логарифм",
    )
    patterns = (
        r"объясни\s+теорем",
        r"реши\s+задач",
        r"докажи\s+что",
        r"что\s+такое\s+(бозон|кварк|интеграл|энтроп)",
    )
    return any(marker in t for marker in markers) or any(re.search(p, t) for p in patterns)


def is_culture_content_query(message: str) -> bool:
    t = (message or "").lower().strip()
    if not t:
        return False
    markers = (
        "текст песни", "строчки песни", "напой", "процитируй", "как там поется", "как там поётся",
        "любимая песня", "песня группы", "альбом", "исполнитель", "фильмограф", "цитата из фильма",
        "цитата из сериала", "цитата из аниме", "цитата из игры", "мем", "лор игры", "лор аниме",
        "что за трек", "кто поет", "кто поёт", "найди песню", "припев", "как поется в припеве",
        "как поётся в припеве", "песни у", "саундтрек", "дискография",
    )
    return any(marker in t for marker in markers)


def is_news_or_factual_query(message: str) -> bool:
    t = (message or "").lower().strip()
    if not t:
        return False
    markers = (
        "последние новости", "новости", "когда выш", "когда выйдет", "кто сейчас",
        "какой сегодня курс", "курс валют", "сколько стоит", "цена", "расписание",
        "результат матча", "счет матча", "счёт матча", "сегодня матч", "дата выхода",
        "кто победил", "турнирная таблица",
    )
    has_year = re.search(r"\b(19|20)\d{2}\b", t) is not None
    return any(marker in t for marker in markers) or has_year


def get_search_kind(message: str) -> str:
    if is_weather_query(message):
        return "blocked_weather"
    if is_school_or_science_query(message):
        return "blocked_science"
    if is_culture_content_query(message):
        return "culture"
    if is_news_or_factual_query(message):
        return "factual"
    return ""


def should_search(message: str) -> bool:
    """True только для запросов, где нужен внешний поиск."""
    return bool(get_search_kind(message))


# ── Парсинг строк "НИК: факт" от LLM ────────────────────────────────

def parse_fact_lines(raw: str) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    if not raw:
        return facts
    for line in raw.split("\n"):
        line = line.strip().lstrip("•-– ")
        if ":" not in line:
            continue
        nick, value = line.split(":", 1)
        nick = nick.strip()
        value = value.strip()
        if nick and value and len(value) > 2:
            facts.append((nick, value))
    return facts


def _normalize_compare_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_raw_chat_fragment(value: str, messages: list[dict], threshold: float = 0.85) -> bool:
    """Возвращает True, если значение слишком похоже на дословную цитату из чата."""
    value_norm = _normalize_compare_text(value)
    if not value_norm:
        return False
    value_words = set(value_norm.split())
    if len(value_words) < 3:
        return False
    for msg in messages:
        msg_norm = _normalize_compare_text(msg["text"])
        if not msg_norm:
            continue
        if value_norm in msg_norm or msg_norm in value_norm:
            return True
        msg_words = set(msg_norm.split())
        overlap = len(value_words & msg_words) / max(len(value_words), len(msg_words))
        if overlap >= threshold:
            return True
    return False


def is_valid_trait(trait: str) -> bool:
    trait = trait.strip().lower()
    if not trait or len(trait) > 40:
        return False
    if len(trait.split()) > 4:
        return False
    if re.search(r"[^а-яёa-z0-9\-\s]", trait):
        return False
    bad = {"сказал", "сказала", "написал", "написала", "ахаха", "хаха", "ну", "блин"}
    return trait not in bad


def fix_username_in_response(text: str, username: str) -> str:
    """Исправляет искажённый никнейм в ответе модели.

    Ключевые требования (анти-регрессия):
    - Никогда не коверкать уже правильный ник.
    - Исправлять ТОЛЬКО префикс (первый ник), не делать глобальную замену по тексту
      (иначе получались артефакты вроде "ск uf" в середине ответа).
    - Fuzzy-порог высокий (≈0.9) + дополнительные проверки (первые буквы, длина).

    Пример починки:
    - "скуф: привет" → "Скуф, привет"
    Но НЕ трогаем:
    - "Скуф, привет" (уже правильно)
    """
    import difflib

    if not username or not text:
        return text

    t = text.lstrip()

    # Парсим "первый токен" до типового разделителя.
    # Разделитель нужен, чтобы не пытаться чинить обычную фразу в начале.
    first_token: str | None = None
    first_sep: str | None = None
    for sep in (", ", ": ", ",", ":"):
        idx = t.find(sep)
        if idx <= 0:
            continue
        # Токен разумной длины (поддерживаем короткие ники).
        if idx > max(int(len(username) * 1.4) + 2, 24):
            continue
        first_token = t[:idx].strip()
        first_sep = sep
        break

    if not first_token or not first_sep:
        return text

    # Если уже совпадает (в любом регистре) — только нормализуем регистр.
    if first_token.lower() == username.lower():
        rest = t[len(first_token) + len(first_sep):]
        return (username + first_sep + rest).strip()

    def _norm(s: str) -> str:
        # Для сравнения убираем пробелы и пунктуацию, оставляем буквы/цифры.
        return re.sub(r"[^\w]", "", (s or "").lower())

    token_norm = _norm(first_token)
    user_norm = _norm(username)
    if not token_norm or not user_norm:
        return text

    # Никогда не трогаем уже корректный ник.
    if token_norm == user_norm:
        rest = t[len(first_token) + len(first_sep):]
        return (username + first_sep + rest).strip()

    # Дополнительные проверки: первые буквы и длина.
    # Это защищает от "случайных" похожих слов в начале ответа.
    if token_norm[:2] != user_norm[:2] and token_norm[:1] != user_norm[:1]:
        return text
    if abs(len(token_norm) - len(user_norm)) > max(2, int(len(user_norm) * 0.2)):
        return text

    ratio = difflib.SequenceMatcher(None, token_norm, user_norm).ratio()
    if ratio < 0.9:
        return text

    rest = t[len(first_token) + len(first_sep):]
    print(f"[НИК] Исправлен префикс '{first_token}' → '{username}' (ratio={ratio:.2f})")
    return (username + first_sep + rest).strip()


def deduplicate_username_in_response(text: str, username: str) -> str:
    """Оставляет ник максимум один раз — в начале ответа.

    Если модель написала ник 2+ раз (напр. «Вася, ой Вася, ну ты и...»),
    удаляем все повторные упоминания кроме первого.

    Требования (анти-регрессия):
    - Работает для коротких ников ("Скуф")
    - Работает для ников с пробелами
    - Удаляет повторные ники не только "Скуф", но и типичные дубли вида
      "Скуф, Скуф, ..." или "Скуф: Скуф — ..."

    Примеры (ожидаемое поведение):
    - "Скуф, Скуф, доела. вкусно вышло." -> "Скуф, доела. вкусно вышло."
    - "IWM: IWM, ok."                    -> "IWM, ok."
    - "Лена ,  Лена: привет"             -> "Лена, привет"
    - "Скуф, Скуф,"                      -> "Скуф,"
    """
    if not username or not text:
        return text

    t = text.strip()

    # --- ЖЁСТКИЙ ФИКС: убираем повтор ника в начале ---
    pattern = rf"^(?:\s*{re.escape(username)}[\s,:\-—–]*){{2,}}"
    if re.match(pattern, t, flags=re.IGNORECASE):
        t = re.sub(pattern, f"{username}, ", t, flags=re.IGNORECASE)

    # Готовим паттерн ника: учитываем ники с пробелами (между частями — любые пробелы).
    parts = [re.escape(p) for p in username.strip().split() if p]
    if not parts:
        return t
    nick_pat = r"\s+".join(parts)
    # Мягкие границы: не матчить подстроки внутри слов. Работает и для коротких ников.
    nick_re = re.compile(rf"(?i)(?<!\w){nick_pat}(?!\w)")
    # Разделители между дублями ника (в начале чаще всего)
    # Добавляем zero-width/BOM символы — иногда модель вставляет их, и дубли не ловятся.
    sep_re = r"[\s,\u200b\ufeff,:\-—–]*"

    # ── 1) Схлопываем дубли ника прямо в начале строки ─────────────────
    # Это главный фикс для "Скуф, Скуф, ..." — такие повторы должны исчезать
    # ДО любого дальнейшего нормалайза/ensure_nick, иначе появляются артефакты.
    #
    # Формат:
    #   ^ <nick> <sep> <nick> <sep> ... <rest>
    # Оставляем только первый <nick>, а хвост чистим от лишних разделителей.
    m0 = re.match(rf"(?is)^\s*({nick_pat})(?P<rest>.*)$", t)
    if m0:
        rest = t[m0.end(1):]
        # Удаляем последовательные дубли ника сразу после первого.
        # Лимит по итерациям — на случай мусора в начале.
        removed_prefix = 0
        for _ in range(8):
            mdup = re.match(rf"(?is)^{sep_re}({nick_pat})", rest)
            if not mdup:
                break
            rest = rest[mdup.end():]
            removed_prefix += 1
        if removed_prefix:
            # После удаления дублей — убираем лишние разделители и нормализуем запятую.
            rest = re.sub(r"^[\s,\u200b\ufeff,:\-—–]+", "", rest).lstrip()
            t = f"{username}, {rest}".strip() if rest else f"{username},"
            t = normalize_strict_nick_prefix(t, username)

    # ── 1b) Удаляем "искажённый дубль" ника сразу после префикса ───────
    # Это отдельный класс артефактов: модель пишет ник дважды, но второй раз
    # ломает пробелами/знаками: "Скуф, ск uf, ..." или "Лена, ле на: ...".
    #
    # ПРАВИЛО: если после первого корректного `username,` идёт кусок, который при
    # нормализации (убрать пробелы/пунктуацию) РАВЕН нику — вырезаем этот кусок.
    # Это НЕ fuzzy по смыслу; это строгое "тот же ник, только разорванный".
    try:
        # Стабилизируем старт, чтобы проще парсить хвост.
        t = normalize_strict_nick_prefix(t, username)
        prefix = f"{username},"
        if t.lower().startswith(prefix.lower()):
            tail = t[len(prefix):].lstrip()

            def _canon(s: str) -> str:
                return re.sub(r"[^\w]", "", (s or "").lower())

            user_can = _canon(username)
            if user_can:
                # Берём небольшой префикс хвоста: там должен быть дубль, если он есть.
                look = tail[:40]
                # Кандидат "ник-подобный": цепочка букв/цифр/пробелов/дефисов до первой "нормальной" точки текста.
                mnick = re.match(r"^[\w\s\-—–]+", look)
                if mnick:
                    cand_raw = mnick.group(0).strip()
                    # Срезаем, если кандидат слишком длинный (значит уже пошёл текст).
                    if 0 < len(cand_raw) <= max(len(username) + 6, 10):
                        cand_can = _canon(cand_raw)
                        if cand_can and cand_can == user_can:
                            # Удаляем кандидат и следующие разделители ", : —" и пробелы.
                            cut = len(cand_raw)
                            tail2 = tail[cut:]
                            tail2 = re.sub(r"^[\s,\u200b\ufeff,:\-—–]+", "", tail2).lstrip()
                            t = f"{username}, {tail2}".strip() if tail2 else f"{username},"
    except Exception:
        # Дедуп не должен ломать ответ.
        pass

    # ── 2) Удаляем все последующие упоминания ника (по всему тексту) ────
    # Требование: оставить ник ровно один раз — самый первый.
    matches = list(nick_re.finditer(t))
    if len(matches) <= 1:
        return t

    result = t
    removed = 0
    for m in reversed(matches[1:]):
        start, end = m.start(), m.end()

        # Захватываем типичные разделители вокруг ника:
        # ", Ник", "Ник,", "Ник: ", "— Ник" и т.п.
        left = result[:start]
        ltrim = re.search(r"[\s,\u200b\ufeff,:\-—–]*$", left)
        if ltrim:
            start = max(0, ltrim.start())

        right = result[end:]
        rtrim = re.match(r"^[\s,\u200b\ufeff,:\-—–]+", right)
        if rtrim:
            end = end + rtrim.end()

        result = (result[:start] + result[end:]).strip()
        removed += 1

    if removed:
        print(f"[НИК] Удалено {removed} лишних упоминаний '{username}'")

    # Финальная стабилизация, чтобы не осталось ", ," или "Ник, , ..."
    return normalize_strict_nick_prefix(result, username).strip()


def normalize_strict_nick_prefix(text: str, username: str) -> str:
    """Приводит начало ответа к строго `Ник, сообщение`.

    Иногда модель пишет варианты вроде:
    - `Ник... Сообщение`
    - `Ник, Ник, сообщение`

    В таких случаях нормализуем: один ник в начале, далее ровно `, `.
    """
    if not text or not username:
        return text

    t = (text or "").lstrip()

    # Ник с пробелами: допускаем любое количество пробелов между частями ника.
    parts = [re.escape(p) for p in username.strip().split() if p]
    if not parts:
        return t.strip()
    nick_pat = r"\s+".join(parts)

    m = re.match(rf"(?i)^{nick_pat}", t)
    if not m:
        return text

    rest = t[m.end():]

    # Убираем мусорные разделители сразу после ника:
    # запятая/двоеточие/точки/эллипсис/дефисы/тире + пробелы.
    rest = re.sub(r"^[\s,:;\.\u2026\-—–]+", "", rest).lstrip()

    # Если модель продублировала ник прямо после первого — вырезаем вторые/третьи повторы.
    for _ in range(4):
        m2 = re.match(rf"(?i)^{nick_pat}", rest)
        if not m2:
            break
        rest = rest[m2.end():]
        rest = re.sub(r"^[\s,:;\.\u2026\-—–]+", "", rest).lstrip()

    return f"{username}, {rest}".strip() if rest else f"{username},"


def strip_urls(text: str) -> str:
    """Удаляет URL, markdown-ссылки и цитатные артефакты поиска из текста ответа."""
    # markdown [текст](url)
    text = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', text)
    # голые URL
    text = re.sub(r'https?://\S+', '', text)
    # цитаты поиска: [[1]], [1], [1][2], [^1] и т.п.
    text = re.sub(r'(\[\[?\^?\d+\]?\])+', '', text)
    # убираем осиротевшие знаки препинания после удаления ссылки
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(\s)[,;:.]+(\s|$)', r'\2', text)
    return text.strip()


def build_daily_diary_digest(entries: list[dict], max_len: int = 800) -> str:
    """Собирает короткую сводку по записям дневника за день."""
    if not entries:
        return ""

    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        raw = str(entry.get('entry', '')).strip()
        if not raw:
            continue
        value = re.sub(r'\s+', ' ', raw).strip(' .')
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            cleaned.append(value)

    if not cleaned:
        return ""

    result = "За сегодня: "
    for part in cleaned:
        candidate = result + ("; " if result != "За сегодня: " else "") + part
        if len(candidate) <= max_len:
            result = candidate
            continue

        remaining = max_len - len(result)
        if remaining > 4:
            trimmed = part[: max(0, remaining - 1)].rstrip(' ,;:.')
            if trimmed:
                result += ("; " if result != "За сегодня: " else "") + trimmed + "…"
        break

    return result if len(result) <= max_len else result[: max_len - 1].rstrip(' ,;:.') + "…"



SELF_IDENTITY_MODE_MARKER = "[РЕЖИМ: SELF_IDENTITY]"

# ── Контекст упоминания бота ─────────────────────────────────────────

def looks_like_third_party_bot_mention(text: str) -> bool:
    """Улучшенная версия: чётко отличает прямое обращение от обсуждения в третьем лице."""
    raw = (text or "").strip()
    if not raw or not has_name_trigger(raw):
        return False

    t = raw.lower()
    bot_lower = CHARACTER_NAME.lower()

    # === ПРЯМОЕ ОБРАЩЕНИЕ (самое важное) ===
    # НейроАнька, привет!
    # Эй НейроАнька как дела
    # НейроАнька!
    direct_patterns = [
        rf"^{re.escape(bot_lower)}[,!?\s]",           # в начале + запятая/восклицание
        rf"^[^\w]*{re.escape(bot_lower)}[,!?\s]",     # с эмодзи или пробелами перед
        rf"[,!?]\s*{re.escape(bot_lower)}[ ,!?$]",    # , НейроАнька или ! НейроАнька
    ]
    if any(re.search(p, t) for p in direct_patterns):
        return False  # ← это прямое обращение, НЕ SELF_IDENTITY

    # === ТРЕТЬЕ ЛИЦО (обсуждение) ===
    # Маша, кто такая НейроАнька?
    # Ваня, что думаешь про НейроАньку?
    if re.search(r"^[^,!?]{2,35},\s+.*" + re.escape(bot_lower), t):
        return True

    # Дополнительный случай: имя бота не в начале сообщения
    if re.search(r"^[^\s,]+,\s+.*" + re.escape(bot_lower), t):
        return True

    # По умолчанию считаем прямым обращением
    return False

def _summarize_bot_mention_question(text: str) -> str:
    """Сжимает сторонний вопрос про бота без имён адресатов."""
    t = (text or "").lower()
    if re.search(r"кто\s+так(?:ая|ой|ое)|что\s+за", t):
        return "автор спрашивает, кто такая Анька"
    if re.search(r"знаешь", t):
        return "автор спрашивает, знают ли тебя"
    if re.search(r"что\s+ты\s+думаешь|что\s+думаешь|как\s+думаешь|думаешь|считаешь|норм|как\s+относишься|что\s+скажешь|что\s+можешь\s+сказать|к\s+[^?!,.]*(?:аньк|аня|аню|ане|нейроаньк)\s+относишься|относишься\s+к\s+[^?!,.]*(?:аньк|аня|аню|ане|нейроаньк)", t):
        return "автор спрашивает чужое мнение о тебе"
    return "автор обсуждает тебя в третьем лице"


def _extract_addressed_name(text: str) -> str:
    """Достаёт имя/ник перед первой запятой: «Маша, что думаешь про Аньку?» → «Маша»."""
    raw = (text or "").strip()
    m = re.match(r"^([^,!?]{2,30}),\s+", raw)
    return m.group(1).strip() if m else ""


def build_third_party_bot_mention_prompt(username: str, text: str) -> str:
    """Готовит безопасный LLM-ввод при стороннем упоминании бота.

    Важно: не отдаём обычный текст как есть. В нём есть адресат перед запятой
    («Оресель, ...»), и обычный responder может принять его за цель вопроса.
    Вместо этого передаём структурированную карточку и специальный режим.
    """
    summary = _summarize_bot_mention_question(text)
    addressed = _extract_addressed_name(text) or "не указан"
    return (
        f"{SELF_IDENTITY_MODE_MARKER}\n"
        "Тебя упомянули в стороннем диалоге. Это НЕ обычный вопрос про другого человека.\n"
        f"Автор сообщения: {username}\n"
        f"Адресат внутри реплики: {addressed}\n"
        "Объект обсуждения: НейроАнька / Анька — это ТЫ.\n"
        f"Смысл реплики: {summary}.\n\n"
        "Сгенерируй живой ответ автору сообщения. "
        "Отвечай от первого лица, про себя. "
        "Не рассказывай про адресата внутри реплики и не оценивай его. "
        "Не используй факты о людях из соседнего чата. "
        "Можно язвительно заметить, что ты сама тут и можешь ответить за себя. "
        "Начни строго с ника автора сообщения."
    )
def build_third_party_bot_mention_answer(username: str, text: str) -> str:
    """Детерминированный ответ на стороннее упоминание бота.

    Важно: не гоняем это через общий get_response, потому что туда
    подмешивается recent_chat_block и модель может схватить случайного
    человека из соседней реплики как цель вопроса.
    """
    summary = _summarize_bot_mention_question(text)
    if "чужое мнение" in summary:
        return (
            f"{username}, я вообще-то сама могу за себя сказать 😏 "
            "НейроАнька — это я: местная язвительная девочка из чата, с настроением и памятью."
        )
    if "кто такая" in summary or "кто такая Анька" in summary:
        return (
            f"{username}, я тут, если что 😏 "
            "НейроАнька — это я: местная язвительная девочка из чата, не справочное бюро."
        )
    if "знают ли" in summary:
        return (
            f"{username}, знают-знают, куда денутся 😏 "
            "я НейроАнька, местная болтливая заноза этого чата."
        )
    return (
        f"{username}, я вообще-то рядом стою 😏 "
        "НейроАнька — это я, так что можно спрашивать напрямую."
    )

# ── Анти-эхо: извлечение «заразных» слов из сообщения юзера ──────────

# Слова которые бот и так использует сам — не имеет смысла блокировать
_VOCAB_STOPWORDS: frozenset[str] = frozenset({
    # Наречия и частицы
    "просто", "очень", "немного", "конечно", "наверное", "вообще", "сейчас",
    "потом", "всегда", "никогда", "нормально", "понятно", "реально", "кстати",
    "короче", "кажется", "неужели", "буквально", "абсолютно", "серьёзно",
    "серьезно", "слишком", "зачем", "почему", "откуда", "давно", "скоро",
    "долго", "быстро", "тихо", "громко",
    # Глаголы-клише
    "хорошо", "плохо", "могу", "буду", "хочу", "люблю", "нравится",
    "говорить", "написать", "сказать", "посмотреть", "понять", "думаю",
    "знаю", "понял", "поняла", "увидеть", "слышать", "делать", "идти",
    "прийти", "пойти", "было", "будет", "была", "есть", "нету", "помню",
    "забыла", "забыл", "смотреть", "читать", "играть", "работать", "гулять",
    # Фатика / приветствия
    "привет", "пока", "ладно", "блин", "спасибо", "пожалуйста",
    "извини", "окей", "ясно", "давай", "угу", "ага", "неа",
    # Прилагательные-клише
    "хороший", "плохой", "красивый", "красивая", "умный", "умная",
    "классный", "классная", "странный", "смешной", "большой", "маленький",
    "старый", "новый", "первый", "последний",
    # Местоимения и связки
    "этого", "этой", "этому", "этим", "этих", "меня", "тебя", "себя",
    "него", "неё", "ними", "нами", "вами", "кого", "кому",
    # Существительные-клише
    "день", "ночь", "время", "люди", "слова", "вопрос", "ответ",
    "тема", "жизнь", "мысль", "вещи", "место", "часть", "точка",
    # Общие краткие слова (длина >= 4 но по смыслу неинтересны)
    "чего", "того", "этот", "тоже", "даже", "если", "чтобы", "когда",
    "пока", "хотя", "надо", "нужно", "можно", "нельзя",
})

# Очень частые / универсальные слова (не «заразная» лексика)
_VOCAB_GENERIC_EXTRA: frozenset[str] = frozenset({
    "сегодня", "вчера", "завтра", "сейчас", "опять", "снова", "здесь", "тут", "там",
    "куда", "где", "когда", "почему", "зачем", "откуда", "куда", "либо", "ибо",
    "будто", "вроде", "типа", "типо", "как", "так", "вот", "это", "эта", "эти",
    "этим", "этих", "этому", "том", "тем", "тех", "тому", "что", "чем", "чём",
    "кто", "кого", "кому", "чей", "чья", "чьё", "чьи", "весь", "всю", "все",
    "всё", "всем", "всеми", "всего", "всём", "свой", "своя", "своё", "свои",
    "мой", "моя", "моё", "мои", "твой", "твоя", "твоё", "твои", "наш", "наша",
    "наше", "наши", "ваш", "ваша", "ваше", "ваши", "какой", "какая", "какое",
    "какие", "такой", "такая", "такое", "такие", "один", "одна", "одно", "одни",
    "два", "две", "три", "раз", "раза", "разу", "много", "мало", "больше", "меньше",
    "очень", "мало", "нет", "да", "не", "ни", "уж", "ли", "же", "бы", "то",
    "ведь", "вот", "ну", "эх", "ох", "ах", "угу", "ага", "это", "там", "тут",
    "ещё", "еще", "уже", "ещё", "ибо", "лишь", "едва", "вряд", "хоть", "пусть",
    "пока", "пора", "разве", "всё", "все", "ничего", "никто", "ничто", "некого",
    "зато", "однако", "значит", "итак", "лишь", "про", "при", "над", "под",
    "без", "для", "про", "со", "об", "во", "до", "от", "из", "на", "по", "за",
    "нибудь", "либо", "когда", "тогда", "снова", "опять", "просто", "прямо",
})

# Обычное слово — не короче (иначе шум); короче допускаем только «странные» токены
_VOCAB_MIN_LEN = 4
_VOCAB_MAX_QUARANTINE_WORDS = 8
# Слишком длинный абзац в одном токене — скорее мусор/URL-обломок
_VOCAB_MAX_TOKEN_LEN = 32


def _vocab_is_weird_token(word: str) -> bool:
    """Эвристики придуманных/мемных/технических кусков — их ловим даже короче обычного порога."""
    if not word or len(word) > _VOCAB_MAX_TOKEN_LEN:
        return False
    if re.search(r'[0-9]', word):
        return True
    has_cy = bool(re.search(r'[а-яё]', word))
    has_la = bool(re.search(r'[a-z]', word))
    if has_cy and has_la:
        return True
    # Слишком длинный одиночный токен в чате чаще мусор/склейка, не обычное слово
    if len(word) >= 22:
        return True
    # троекратное повторение одного символа (мооорковка, ааа)
    if re.search(r'(.)\1{2,}', word):
        return True
    # длинная согласная цепь по-русски — нетипично для обычной речи
    if has_cy and re.search(r'[бвгджзйклмнпрстфхцчшщъь]{5,}', word):
        return True
    return False


def _vocab_is_trivial_repeat(word: str) -> bool:
    """Один и тот же символ подряд — не кладём в карантин (бессмысленно для анти-эхо)."""
    return len(word) >= 3 and len(set(word)) == 1


def _vocab_should_skip_length(word: str) -> bool:
    if len(word) < 3:
        return True
    if len(word) < _VOCAB_MIN_LEN and not _vocab_is_weird_token(word):
        return True
    return False


def extract_unusual_words(text: str, username: str = "") -> list[str]:
    """Извлекает необычные/специфичные слова из сообщения юзера.

    Цель — поймать лексику, которую модель тянет в ответы (мемы, придуманные слова,
    узкие термины), и не засорять карантин стоп-словами и слишком общими формами.

    Не включает: стоп-слова, очень общие слова, слишком короткие обычные токены,
    ник пользователя и ник бота.
    Сначала отдаются «странные» кандидаты, затем длинные редкие по виду.
    Не более ``_VOCAB_MAX_QUARANTINE_WORDS`` слов.
    """
    if not (text or "").strip():
        return []

    username_lower = (username or "").lower().strip()
    char_lower = CHARACTER_NAME.lower()

    # Слова и простые композиты через дефис (кусок-кусок)
    raw_words = re.findall(r'[а-яёa-z]+(?:-[а-яёa-z]+)*', text.lower())

    candidates: list[tuple[str, bool]] = []
    seen: set[str] = set()

    for word in raw_words:
        if word in seen:
            continue

        if _vocab_is_trivial_repeat(word):
            continue
        if _vocab_should_skip_length(word):
            continue
        if len(word) > _VOCAB_MAX_TOKEN_LEN:
            continue

        if word in _VOCAB_STOPWORDS or word in _VOCAB_GENERIC_EXTRA:
            continue

        if username_lower and (
            word == username_lower
            or (len(username_lower) >= 4 and username_lower.startswith(word))
            or (len(word) >= 4 and word.startswith(username_lower))
        ):
            continue
        if word == char_lower or (len(char_lower) >= 4 and word.startswith(char_lower)):
            continue

        weird = _vocab_is_weird_token(word)
        # Обычное слово ровно длины 3 без маркеров странности — почти всегда шум
        if len(word) == 3 and not weird:
            continue

        seen.add(word)
        candidates.append((word, weird))

    # Странные и длинные — вперёд; не раздуваем промпт
    candidates.sort(key=lambda t: (-int(t[1]), -len(t[0]), t[0]))

    return [w for w, _ in candidates[:_VOCAB_MAX_QUARANTINE_WORDS]]

