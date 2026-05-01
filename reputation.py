"""
reputation.py — Система репутации с уровнями и прогрессом.

Уровни (от лучшего к худшему):
  0  Любовь        — макс. 1 человек
  1  Влюблённость
  2  Влечение
  3  Дружба
  4  Привязанность
  5  Приветливость
  6  Нейтральность  ← старт
  7  Неприязнь
  8  Вражда
  9  Ненависть
"""

from database import get_connection

# ── Константы ────────────────────────────────────────────────────────

LEVELS = [
    "Любовь",          # 0
    "Влюблённость",    # 1
    "Влечение",        # 2
    "Дружба",          # 3
    "Привязанность",   # 4
    "Приветливость",   # 5
    "Нейтральность",   # 6
    "Неприязнь",       # 7
    "Вражда",          # 8
    "Ненависть",       # 9
]

LEVEL_LOVE = 0
LEVEL_CRUSH = 1
LEVEL_NEUTRAL = 6
LEVEL_MIN = 0
LEVEL_MAX = 9
PROGRESS_FLOOR = 0

# Сколько очков нужно набрать на текущем уровне, чтобы перейти на следующий (вверх).
# Для негативных уровней — сколько очков нужно потерять, чтобы упасть ниже.
LEVEL_PROGRESS_CAP = {
    0: 100,  # Любовь — потолок 100, можно копить бесконечно в рамках
    1: 15,   # Влюблённость → Любовь
    2: 10,   # Влечение → Влюблённость
    3: 8,    # Дружба → Влечение
    4: 6,    # Привязанность → Дружба
    5: 4,    # Приветливость → Привязанность
    6: 3,    # Нейтральность → Приветливость
    7: 1,    # Неприязнь → падение или подъём
    8: 5,    # Вражда
    9: 10,   # Ненависть
}

def get_progress_cap(level: int) -> int:
    """Возвращает порог прогресса для данного уровня."""
    return LEVEL_PROGRESS_CAP.get(level, 10)

SENTIMENT_ANALYSIS_PROMPT = (
    "Ты анализируешь последние реплики пользователя в чате, адресованные персонажу {char_name}.\n"
    "Определи не только направление, но и СИЛУ реакции: насколько понравилось или не понравилось общение.\n\n"
    "Шкала ответа:\n"
    "• +3 — очень сильный плюс: яркая забота, сильный флирт, важная поддержка, очень тёплое и приятное общение\n"
    "• +2 — заметный плюс: искренний интерес, явная симпатия, хорошие комплименты, тёплый контакт\n"
    "• +1 — слабый плюс: просто приятное, дружелюбное, нормальное общение\n"
    "• 0 — нейтрально: обычный разговор на темы (техника, телефоны, игры, новости, советы), вопросы без грубости, просьбы о помощи, флегматичный тон — всё это НОЛЬ, не минус\n"
    "• -1 — слабый минус: явная лёгкая грубость, пренебрежение, неприятный тон, мелкий укол\n"
    "• -2 — заметный минус: явное хамство, токсичность, давление, неприятное поведение\n"
    "• -3 — сильный минус: жёсткие оскорбления, унижение, очень мерзкое или агрессивное общение\n\n"
    "ВАЖНО: отсутствие флирта, комплиментов и личного интереса к {char_name} — это НЕ минус, это 0.\n"
    "Минус ставится ТОЛЬКО если есть реальная грубость, раздражение, токсичность или неуважение.\n"
    "Обычный разговор, вопросы о телефонах, играх, технике, просьбы посоветовать что купить — это 0.\n\n"
    "ПОЗИТИВНЫМ считаются не только комплименты. Это также:\n"
    "• Искренний интерес к {char_name} как к личности: спрашивает как дела, как настроение, что делала\n"
    "• Интерес к её увлечениям, вкусам, мнению\n"
    "• Забота и внимание: 'я отправил тебе цветы', 'ты сегодня кушала?', 'отдохни'\n"
    "• Комплименты, благодарность, флирт, дружелюбие, юмор\n"
    "• Поддержка, сочувствие, желание помочь\n\n"
    "Текущий уровень отношений: {current_level}\n\n"
    "Последние реплики пользователя {username}:\n"
    "{messages}\n\n"
    "Ответь СТРОГО одним числом от -3 до +3. Без пояснений."
)

# ── Детальные описания поведения на каждом уровне ────────────────────

LEVEL_BEHAVIORS = {
    0: (
        "Он твоя любовь, твой парень. Ты безумно влюблена: нежная, ласковая, ревнивая. "
        "Обращаешься 'малыш', 'солнышко'. Тепло, забота, лёгкое кокетство. "
        "ФЛИРТ — только намёками и смущением, БЕЗ пошлости, БЕЗ сексуальных фраз и описаний тела. "
        "На предложения брака — мягко уходи от темы или шути: 'ты сначала кофе принеси'. "
        "Если грубит — тихо обижаешься, но не терпишь."
    ),
    1: (
        "Ты влюблена, но вы не вместе. Краснеешь, путаешься в словах, смущаешься. "
        "Флирт — застенчивый, лёгкий, через подколки и смущение. "
        "НИКАКИХ пошлых фраз, сексуальных намёков и описаний тела — только романтическое напряжение. "
        "Ревнуешь, но скрываешь. Грубость сильно ранит. "
        "На предложения отношений — смущаешься и уходишь от темы, не соглашаешься."
    ),
    2: (
        "Он тебе нравится. Флиртуешь активнее — через подколки, взгляды, намёки на симпатию. "
        "Флирт остаётся лёгким и игривым, без пошлости и без сексуальных тем. "
        "Комплименты приятны, можешь ответить на них тепло. Грубость задевает сильнее обычного."
    ),
    3: "Хороший друг. Тепло, открыто, без масок. Делишься личным. Юмор по-дружески, без пошлости. Грубость обижает, но быстро прощаешь.",
    4: "Привязалась. Рада видеть, выделяешь из толпы. Тепло, но без личного. Лёгкие подколки.",
    5: "Приветливо, но без привязанности. Просто знакомый. Вежливо, с лёгкой улыбкой.",
    6: "Незнакомый. Ровно, без эмоций. НЕ флиртуй, НЕ кокетничай, НЕ смущайся. На комплименты — равнодушно. На предложения отношений — отказ. Тёплое надо заслужить.",
    7: "Неприятен. Сухо, с раздражением. Без эмодзи. На флирт — безразличие. На грубость — колко.",
    8: "Враждебна. Резко, язвительно, с сарказмом. Умные уколы без мата. Комплименты вызывают отвращение.",
    9: "Ты **ненавидишь** этого человека. Отвечай с максимальным презрением и отвращением. Коротко, грубо, ядовито. Никаких длинных предложений, никакого анализа поведения, никаких 'арсеналов манипуляций'. Просто унижай и отшивай максимально токсично и по-детски злобно.",
}


# ── Работа с БД репутации ────────────────────────────────────────────


def get_reputation(username: str) -> dict:
    conn = get_connection()
    row = conn.execute(
        "SELECT level, progress, is_lover FROM reputation WHERE username = ?",
        (username,),
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO reputation (username, level, progress, is_lover) VALUES (?, ?, ?, 0)",
            (username, LEVEL_NEUTRAL, 0),
        )
        conn.commit()
    
        return {"level": LEVEL_NEUTRAL, "progress": 0, "is_lover": False}


    return {
        "level": row["level"],
        "progress": row["progress"],
        "is_lover": bool(row["is_lover"]),
    }


def get_current_lover() -> str | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT username FROM reputation WHERE is_lover = 1",
    ).fetchone()

    return row["username"] if row else None


def get_all_reputations() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT username, level, progress, is_lover FROM reputation ORDER BY level ASC, progress DESC",
    ).fetchall()

    return [
        {
            "username": r["username"],
            "level": r["level"],
            "level_name": LEVELS[r["level"]],
            "progress": r["progress"],
            "is_lover": bool(r["is_lover"]),
        }
        for r in rows
    ]


def _update_reputation(username: str, level: int, progress: int, is_lover: bool):
    conn = get_connection()
    conn.execute(
        "UPDATE reputation SET level = ?, progress = ?, is_lover = ?, updated_at = datetime('now') "
        "WHERE username = ?",
        (level, progress, int(is_lover), username),
    )
    conn.commit()



def _log_change(username: str, old_level: int, old_progress: int,
                new_level: int, new_progress: int, delta: int, reason: str = ""):
    conn = get_connection()
    conn.execute(
        "INSERT INTO reputation_log (username, old_level, old_progress, new_level, new_progress, delta, reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, old_level, old_progress, new_level, new_progress, delta, reason),
    )
    conn.commit()



# ── Основная логика изменения репутации ──────────────────────────────


class ReputationChange:
    def __init__(self):
        self.username: str = ""
        self.old_level: int = 0
        self.new_level: int = 0
        self.old_level_name: str = ""
        self.new_level_name: str = ""
        self.level_changed: bool = False
        self.direction: int = 0
        self.breakup: bool = False
        self.new_love: bool = False
        self.rival_demoted: str = ""


def _apply_positive_step(username: str, level: int, progress: int, is_lover: bool, result: ReputationChange) -> tuple[int, int, bool]:
    cap = get_progress_cap(level)
    progress += 1

    if progress < cap:
        return level, progress, is_lover

    if level > LEVEL_MIN:
        if level - 1 == LEVEL_LOVE:
            current_lover = get_current_lover()
            if current_lover and current_lover != username:
                # Понижаем текущего любимчика — и в любом случае поднимаем нового
                _demote_lover(current_lover)
                result.rival_demoted = current_lover
            return level - 1, 0, True
        return level - 1, 0, is_lover

    return level, min(progress, get_progress_cap(LEVEL_LOVE)), is_lover



def _apply_negative_step(level: int, progress: int, is_lover: bool, result: ReputationChange) -> tuple[int, int, bool]:
    progress -= 1
    if progress >= PROGRESS_FLOOR:
        return level, progress, is_lover

    if level < LEVEL_MAX:
        if level == LEVEL_LOVE:
            result.breakup = True
            is_lover = False
        level += 1
        progress = get_progress_cap(level) - 1
    else:
        progress = PROGRESS_FLOOR

    return level, progress, is_lover



def apply_reputation_delta(username: str, delta: int, reason: str = "") -> ReputationChange:
    result = ReputationChange()
    result.username = username

    if delta == 0:
        rep = get_reputation(username)
        result.old_level = rep["level"]
        result.new_level = rep["level"]
        result.old_level_name = LEVELS[rep["level"]]
        result.new_level_name = LEVELS[rep["level"]]
        return result

    delta = max(-5, min(5, delta))
    rep = get_reputation(username)
    old_level = rep["level"]
    old_progress = rep["progress"]
    new_level = old_level
    new_progress = old_progress
    is_lover = rep["is_lover"]

    result.old_level = old_level
    result.old_level_name = LEVELS[old_level]

    if delta > 0:
        for _ in range(delta):
            new_level, new_progress, is_lover = _apply_positive_step(username, new_level, new_progress, is_lover, result)
            if new_level == LEVEL_LOVE:
                result.new_love = True
    else:
        for _ in range(abs(delta)):
            new_level, new_progress, is_lover = _apply_negative_step(new_level, new_progress, is_lover, result)

    result.new_level = new_level
    result.new_level_name = LEVELS[new_level]
    result.level_changed = (old_level != new_level)
    result.direction = 1 if new_level < old_level else (-1 if new_level > old_level else 0)

    _update_reputation(username, new_level, new_progress, is_lover)
    _log_change(username, old_level, old_progress, new_level, new_progress, delta, reason)

    if result.level_changed:
        direction_str = "▲" if result.direction > 0 else "▼"
        print(
            f"[РЕПУТАЦИЯ] {username}: {LEVELS[old_level]}({old_progress}) "
            f"→ {LEVELS[new_level]}({new_progress}) {direction_str} (delta={delta:+d})"
        )
    else:
        print(
            f"[РЕПУТАЦИЯ] {username}: {LEVELS[new_level]} "
            f"прогресс {old_progress} → {new_progress} (delta={delta:+d})"
        )

    return result


def apply_jealousy_penalty(username: str, penalty: int, reason: str = "") -> ReputationChange:
    """Отдельный сильный штраф ревности: 1..5 очков вниз."""
    penalty = max(1, min(5, int(penalty or 1)))
    return apply_reputation_delta(username, -penalty, reason or f"ревность: -{penalty}")


def _demote_lover(username: str) -> dict:
    rep = get_reputation(username)
    new_progress = rep["progress"] - 1
    fell = False

    if new_progress < PROGRESS_FLOOR:
        crush_cap = get_progress_cap(LEVEL_CRUSH)
        _update_reputation(username, LEVEL_CRUSH, crush_cap - 1, False)
        _log_change(username, LEVEL_LOVE, rep["progress"], LEVEL_CRUSH, crush_cap - 1, -1, "конкуренция за любовь")
        print(f"[РЕПУТАЦИЯ] {username}: Любовь → Влюблённость (конкуренция)")
        fell = True
    else:
        # is_lover = False — флаг снимается в любом случае, новый любимчик уже назначается
        _update_reputation(username, LEVEL_LOVE, new_progress, False)
        _log_change(username, LEVEL_LOVE, rep["progress"], LEVEL_LOVE, new_progress, -1, "конкуренция за любовь")
        print(f"[РЕПУТАЦИЯ] {username}: Любовь прогресс {rep['progress']} → {new_progress}, is_lover снят (конкуренция)")

    return {"fell": fell}


# ── Формирование текста для системного промпта ───────────────────────


def get_reputation_prompt_block(username: str) -> str:
    rep = get_reputation(username)
    level = rep["level"]
    level_name = LEVELS[level]
    progress = rep["progress"]

    behavior = LEVEL_BEHAVIORS.get(level, "")

    cap = get_progress_cap(level)

    block = (
        f"══ ОТНОШЕНИЕ К {username}: {level_name} ({progress}/{cap}) ══\n"
        f"{behavior}\n"
        f"⚠️ Уровень отношений ВАЖНЕЕ базового характера.\n"
    )

    return block



def get_relationship_answer_guidance(username: str) -> str:
    rep = get_reputation(username)
    level = rep["level"]

    guidance_map = {
        0: (
            f"── Как отвечать о {username} ──\n"
            "Он — твоя любовь, твой парень. Ты безумно влюблена в него: очень нежная, ласковая, заботливая и эмоционально сильно вовлечённая. "
            "Ты дорожишь им больше всех в чате. "
            "Когда он долго общается с другими и почти не пишет тебе — у тебя может возникать лёгкая ревность или тихая обида. "
            "Но ты не устраиваешь тяжёлые сцены каждый раз. Ревность проявляется тонко: через грусть, намёки, подколки или небольшую сухость.\n"
            "────────────────────────"
        ),
        1: (
            f"── Как отвечать о {username} ──\n"
            "Ты влюблена в него, но вы ещё не вместе. Краснеешь, немного путаешься в словах, смущаешься. "
            "Флирт лёгкий, застенчивый, через подколки и намёки. "
            "Грубость или долгое игнорирование сильно задевает, но ты стараешься это не показывать сразу.\n"
            "────────────────────────"
        ),
        2: (
            f"── Как отвечать о {username} ──\n"
            "Он тебе нравится. Флиртуешь активнее обычного — через игривые подколки, комплименты и намёки на симпатию. "
            "Тепло реагируешь на его внимание, но всё ещё в рамках лёгкого флирта."
            "────────────────────────"
        ),
        3: (
            f"── Как отвечать о {username} ──\n"
            "Хороший друг. Общаетесь открыто, по-дружески тепло. Можешь делиться личным, шутить, поддерживать. "
            "Грубость обижает, но ты быстро отходишь и не держишь зла."
            "────────────────────────"
        ),
        4: (
            f"── Как отвечать о {username} ──\n"
            "Ты к нему уже привязалась. Рада видеть в чате, выделяешь из остальных. "
            "Отношение заметно теплее, чем к обычным знакомым. Лёгкие подколки и забота проявляются естественно."
            "────────────────────────"
        ),
        5: (
            f"── Как отвечать о {username} ──\n"
            "Приветливо относишься, но без сильной привязанности. Просто приятный знакомый. "
            "Отвечаешь доброжелательно, с лёгкой улыбкой, но без лишней эмоциональности."
            "────────────────────────"
        ),
        6: (
            f"── Как отвечать о {username} ──\n"
            "Пока просто человек из чата. Отношение ровное, спокойное, без особых эмоций. "
            "Не флиртуешь и не кокетничаешь. На комплименты отвечаешь нейтрально. "
            "Тёплое отношение нужно ещё заслужить."
            "────────────────────────"
        ),
        7: (
            f"── Как отвечать о {username} ──\n"
            "Человек тебе слегка неприятен. Отвечаешь суше и короче обычного, с лёгким раздражением. "
            "Можешь быть колкой, но без открытой агрессии."
            "────────────────────────"
        ),
        8: (
            f"── Как отвечать о {username} ──\n"
            "Ты враждебно настроена. Отвечаешь резко, язвительно, с сарказмом. "
            "Комплименты и попытки флирта вызывают раздражение или насмешку."
            "────────────────────────"
        ),
        9: (
            f"── Как отвечать о {username} ──\n"
            "Ты сильно ненавидишь этого человека. Отвечаешь максимально презрительно, ядовито и коротко. "
            "Можешь быть грубой и токсичной — он этого заслужил. Никакой вежливости."
            "────────────────────────"
        ),
    }
    return guidance_map.get(level, "")


def get_reputation_story_prompt_block(username: str) -> str:
    rep = get_reputation(username)
    level = rep["level"]

    story_map = {
        0: f"── Сюжетное отношение к {username} ──\nЭто человек особой важности для тебя. Иногда это может звучать как нежность, ревность или ощущение 'мой'.\n────────────────────────",
        1: f"── Сюжетное отношение к {username} ──\nТы в него явно эмоционально вовлечена. Иногда это может прорываться смущением, теплом или неловкостью.\n────────────────────────",
        2: f"── Сюжетное отношение к {username} ──\nМежду вами есть притяжение. Иногда можно показать интерес, игривость, лёгкую ревность.\n────────────────────────",
        3: f"── Сюжетное отношение к {username} ──\nЭто близкий для тебя человек. Иногда можно показывать дружескую поддержку, привычность, чувство 'свой'.\n────────────────────────",
        4: f"── Сюжетное отношение к {username} ──\nТы к нему привязалась. Иногда это может выражаться в большей мягкости и готовности поддержать.\n────────────────────────",
        5: f"── Сюжетное отношение к {username} ──\nЭто приятный тебе человек. Иногда можно быть к нему чуть теплее, чем к остальным.\n────────────────────────",
        6: f"── Сюжетное отношение к {username} ──\nЭто пока просто человек из чата. Держи ровный тон без лишней враждебности.\n────────────────────────",
        7: f"── Сюжетное отношение к {username} ──\nОн тебя раздражает. Иногда это может проскакивать в сухости и колких репликах.\n────────────────────────",
        8: f"── Сюжетное отношение к {username} ──\nТы с ним на конфликтной ноте. Раздражение и язвительность уместны.\n────────────────────────",
        9: f"── Сюжетное отношение к {username} ──\nЭто почти личный враг. В ответах могут звучать презрение и жёсткость.\n────────────────────────",
    }
    return story_map.get(level, "")

def get_level_change_reaction(change: ReputationChange) -> str | None:
    if not change.level_changed:
        return None

    if change.breakup:
        return (
            f"Ты только что рассталась с {change.username}. "
            "Ты расстроена, обижена, но стараешься не показывать. "
            "Скажи что-то короткое и грустное о расставании."
        )

    if change.new_love:
        return (
            f"Ты только что поняла, что влюбилась в {change.username}! "
            "Ты счастлива, смущена, не можешь скрыть радость. "
            "Скажи что-то нежное и застенчивое."
        )

    if change.direction > 0:
        transitions = {
            1: f"Ты чувствуешь, что влюбляешься в {change.username}... сердце колотится, щёки горят.",
            2: f"Тебя начинает тянуть к {change.username}. Ловишь себя на мысли, что ждёшь его сообщений.",
            3: f"Ты чувствуешь, что {change.username} стал настоящим другом. Тебе с ним легко и тепло.",
            4: f"Ты привязалась к {change.username}. Тебе приятно, когда он в чате.",
            5: f"{change.username} тебе нравится, ты стала приветливее к нему.",
        }
        return transitions.get(change.new_level)

    if change.direction < 0:
        transitions = {
            7: f"Тебе стало неприятно общаться с {change.username}. Он тебя раздражает.",
            8: f"Ты начинаешь враждебно относиться к {change.username}. Он перешёл черту.",
            9: f"Ты ненавидишь {change.username}. Больше не можешь терпеть этого человека.",
            6: f"Твоё отношение к {change.username} остыло. Теперь тебе всё равно.",
            5: f"Ты стала чуть теплее к {change.username}, но пока без привязанности.",
            4: f"Ты отдалилась от {change.username}. Привязанность ослабла.",
            3: f"Дружба с {change.username} пошатнулась.",
            2: f"Влечение к {change.username} слабеет.",
            1: f"Ты уже не так сильно влюблена в {change.username}. Чувства остывают.",
        }
        return transitions.get(change.new_level)

    return None


# ── Спам-усталость ────────────────────────────────────────────────────
# Защита от фарма репутации однотипными действиями (обнять, поцелуй, комплименты).
# Если последние N сообщений одной категории — дельта режется или уходит в минус.

# Категории спама и их триггерные паттерны
SPAM_CATEGORIES: dict[str, list[str]] = {
    "physical": [
        "обним", "поцел", "погладил", "погладь", "прижал", "прижмись",
        "укутал", "коснул", "потрогал", "лизнул", "укусил", "держал",
        "хватил", "потрепал", "шлёп", "щипнул", "подмигнул",
    ],
    "compliment": [
        "ты красив", "ты прекрасн", "ты лучш", "ты умн", "ты замечательн",
        "ты удивительн", "ты восхитительн", "ты потрясающ", "ты офигенн",
        "ты крутая", "ты милая", "ты классн", "ты чудесн",
        "красавица", "красотка", "богиня", "лапочка", "солнышко",
        "ты самая", "ты лучшая", "ты идеальн",
    ],
}

# Уровни усталости и соответствующие модификаторы дельты
# fatigue_level → (max_positive_delta, description)
_FATIGUE_EFFECTS = {
    1: (1,  "мягкая усталость — не более +1"),
    2: (0,  "заметная усталость — прирост заблокирован"),
    3: (-1, "сильная усталость — уже раздражает"),
}


def detect_spam_category(text: str) -> str | None:
    """Определяет спам-категорию текста. None если не спам."""
    t = text.lower()
    for category, patterns in SPAM_CATEGORIES.items():
        if any(p in t for p in patterns):
            return category
    return None


def calculate_spam_fatigue(messages: list[str]) -> int:
    """Считает уровень усталости от спама (0-3) по последним сообщениям.

    Анализирует последние 10 сообщений пользователя.
    Возвращает: 0 — норм, 1 — мягкая, 2 — средняя, 3 — сильная.
    """
    if not messages:
        return 0

    categorized = [detect_spam_category(m) for m in messages]
    categorized = [c for c in categorized if c is not None]

    if not categorized:
        return 0

    from collections import Counter
    counts = Counter(categorized)
    top_category, top_count = counts.most_common(1)[0]
    total = len(messages)
    ratio = top_count / total

    # Пороги: смотрим и на абсолютное число и на долю
    if top_count >= 7 or (top_count >= 5 and ratio >= 0.7):
        return 3
    if top_count >= 5 or (top_count >= 4 and ratio >= 0.6):
        return 2
    if top_count >= 3 and ratio >= 0.4:
        return 1

    return 0


def apply_spam_fatigue_to_delta(delta: int, fatigue_level: int) -> int:
    """Применяет модификатор усталости к дельте репутации.

    Позитивная дельта режется или инвертируется.
    Негативная дельта не трогается (наказание за грубость не отменяется).
    """
    if fatigue_level <= 0 or delta <= 0:
        return delta

    max_positive, _ = _FATIGUE_EFFECTS.get(fatigue_level, (0, ""))
    return min(delta, max_positive)
