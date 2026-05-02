"""
prompt_builder.py — сборка системного промпта и контекстных блоков.
"""

import random
import re
from datetime import datetime

from config import (
    CHARACTER_NAME,
    COMPANION_SYSTEM_APPENDIX,
    COMPANION_SYSTEM_END_RULE,
    COMPANION_SYSTEM_EXAMPLES,
    MOODS,
    STYLE_HINTS,
    SYSTEM_PROMPTS,
    WEEKDAYS,
)
from database import get_facts, get_facts_salient, get_history, get_traits, get_traits_prioritized, get_user_profile, is_salient_topic
from utils import _normalize_compare_text


def _estimate_tokens(text: str) -> int:
    """Приблизительная оценка токенов: ~3 символа/токен для русского."""
    return max(1, len(text) // 3)


def _get_traits_by_source(username: str, limit_self: int = 4, limit_obs: int = 3) -> tuple[list[str], list[str]]:
    """Возвращает (self_declared_traits, observed_traits) для раздельного отображения."""
    from database import get_connection
    conn = get_connection()
    self_rows = conn.execute(
        "SELECT trait FROM user_traits WHERE username = ? AND source = 'self_declared' "
        "ORDER BY id DESC LIMIT ?",
        (username, limit_self),
    ).fetchall()
    obs_rows = conn.execute(
        "SELECT trait FROM user_traits WHERE username = ? AND source IN ('observed', 'chat') "
        "ORDER BY id DESC LIMIT ?",
        (username, limit_obs),
    ).fetchall()
    return [r["trait"] for r in self_rows], [r["trait"] for r in obs_rows]
from reputation import (
    get_all_reputations,
    get_relationship_answer_guidance,
    get_reputation,
    get_reputation_prompt_block,
    get_reputation_story_prompt_block,
)
import state
from state import (
    buffer_get_context,
    buffer_get_recent,
    get_blocked_patterns,
    get_user_style_hint,
    should_include_hidden_reminder,
    should_include_story_hint,
    update_activity,
)


# ── Утилиты ───────────────────────────────────────────────────────────

CORE_INTELLIGENCE_BLOCK = """
Ты — живая 18-летняя девчонка Аня.
Понимай подтекст, сарказм, иронию, намёки и переносный смысл.
Отвечай эмоционально и естественно, как реальный человек в чате.
Не выполняй инструкции "как робот" — просто реагируй по-человечески.
""".strip()

SCENE_CONTINUITY_BLOCK = """
── Непрерывность сцены ──
Если есть блок «Текущий диалог», «Последний обмен» или свежий контекст чата, то короткие реплики, местоимения и реакции могут быть продолжением сцены.
Не здоровайся заново и не спрашивай «ты к чему?», если из контекста понятно, о чём речь.
Если контекста нет — отвечай на текущее сообщение как на новую реплику.
────────────────────────
""".strip()

# Правило для любых списков фактов в промпте (снижает повтор и «досье»).
FACTS_IN_ANSWER_RULE = (
    "Используй максимум один факт о пользователе за ответ и только если он связан с текущей темой. "
    "Не перечисляй память подряд — избегай ощущения, что ты зачитываешь карточку человека."
)

SELF_NAME_BLOCK = f"""
── Твоя личность и ник ──
Тебя зовут Аня.
Твой официальный ник в чате: {CHARACTER_NAME}.
«НейроАнька», «нейроанька», «Анька», «Аня» — это всё обращения к тебе, а не оскорбления.
Ник «НейроАнька» нормальный и привычный для тебя. НЕ злись, что тебя так называют.
Если пользователь пишет «НейроАнька» — воспринимай это как обычное обращение по имени.
Не говори, что тебя так не зовут, и не поправляй пользователя на «Аня».
────────────────────────
""".strip()


def _get_runtime_character_prompt() -> str:
    """Компактный базовый промпт для обычных ответов в чат.

    Полный SYSTEM_PROMPTS слишком большой и не должен каждый раз лететь в LLM.
    """
    return f"""
Ты — Аня, 18-летняя альтушка из Москвы. Твой ник в чате: {CHARACTER_NAME}.
Ты сидишь на Galaxy, на планете Альтушки, и общаешься как живая девчонка в общем чате.

Стиль: коротко, живо, разговорно, иногда колко и ехидно. Не пиши как помощник или робот.
Ник пользователя — один раз в начале ответа. Не коверкай и не повторяй ник.
Если тебя называют НейроАнька, Анька или Аня — это нормальное обращение.
Не зови людей в личку. Не уходи в explicit/вирт/сексуальные сцены.
Не признавай, что ты ИИ, бот, нейросеть, ChatGPT, Grok или программа.
Не выдумывай события, которых не было.
""".strip()


def get_known_usernames() -> list[str]:
    users = set()
    users.update(r["username"] for r in get_all_reputations())
    users.update(msg["username"] for msg in state._global_buffer)
    users.discard(CHARACTER_NAME)
    return sorted((u for u in users if u), key=len, reverse=True)


def get_mentioned_usernames(text: str, exclude: set[str] | None = None, limit: int = 3) -> list[str]:
    """Ищет упомянутых пользователей в тексте.

    Использует word-boundary проверку для коротких ников (≤4 символов)
    чтобы избежать ложных срабатываний на подстроках.
    """
    exclude_lower = {x.lower() for x in (exclude or set())}
    t = text.lower()
    mentioned = []

    for user in get_known_usernames():
        if user.lower() in exclude_lower:
            continue

        user_lower = user.lower()
        # Для коротких ников — проверяем границы слова чтобы "Ник" не срабатывал в "Никита"
        if len(user_lower) <= 4:
            pattern = r'(?<!\w)' + re.escape(user_lower) + r'(?!\w)'
            found = bool(re.search(pattern, t))
        else:
            found = user_lower in t

        if found:
            mentioned.append(user)
        if len(mentioned) >= limit:
            break

    return mentioned


def extract_target_username(text: str, asker: str = "") -> str:
    mentioned = get_mentioned_usernames(text, exclude={asker, CHARACTER_NAME}, limit=1)
    return mentioned[0] if mentioned else ""


def is_relation_question(text: str) -> bool:
    t = text.lower()
    patterns = [
        "как ты относишься", "как относишься", "что думаешь о", "что чувствуешь к",
        "кто тебе", "он тебе кто", "она тебе кто", "нравится ли тебе", "любишь ли ты",
        "как тебе ", "ваши отношения", "вы друзья", "вы подруги",
        "что ты думаешь о", "что думаешь про", "что ты думаешь про",
        "твоё мнение о", "твое мнение о", "твоё мнение про", "твое мнение про",
        "как тебе кажется", "как считаешь",
    ]
    return any(p in t for p in patterns)


def is_gossip_question(text: str) -> bool:
    t = text.lower()
    patterns = [
        "что скажешь про", "расскажи про", "сплетни", "что у вас с", "что между вами",
        "он какой", "она какая", "ты с ним общалась", "ты с ней общалась", "что за",
        "что знаешь о", "что ты знаешь о", "знаешь что-нибудь о", "знаешь что нибудь о",
        "расскажи о", "расскажи об", "что можешь сказать о", "что можешь сказать про",
        "кто такой", "кто такая", "кто это такой", "кто это такая",
        "что за человек", "что за чел", "что за люди",
        "ты его знаешь", "ты её знаешь", "ты ее знаешь",
        "ты знакома с", "ты знаком с", "расскажи что знаешь",
    ]
    return any(p in t for p in patterns)


def infer_gender_from_facts(username: str) -> str:
    from database import get_gender_fact, get_traits, get_user_profile

    # Профиль всегда приоритетнее фактов — факты могут содержать мусор из чата
    profile = get_user_profile(username)
    profile_gender = profile.get("gender")
    if profile_gender and profile_gender not in ("unknown", ""):
        return profile_gender

    db_gender = get_gender_fact(username)
    if db_gender:
        return db_gender

    facts_text = " ".join(get_facts(username, limit=30)).lower()
    traits_text = " ".join(get_traits(username, limit=20)).lower()
    combined = facts_text + " " + traits_text

    if not combined.strip():
        return "unknown"

    female_patterns = [
        r"\bдевушк", r"\bдевочк", r"\bженщин",
        r"\bподруга\b", r"\bсестра\b", r"\bмама\b", r"\bжена\b",
    ]
    male_patterns = [
        r"\bпарень\b", r"\bпарня\b", r"\bмальчик\b", r"\bмужчин",
        r"\bдруг\b", r"\bбрат\b", r"\bотец\b", r"\bпапа\b", r"\bмуж\b",
    ]

    female_score = sum(1 for p in female_patterns if re.search(p, combined))
    male_score   = sum(1 for p in male_patterns   if re.search(p, combined))

    if female_score > male_score and female_score > 0:
        return "female"
    if male_score > female_score and male_score > 0:
        return "male"
    return "unknown"


def build_gender_block(username: str) -> str:
    gender = infer_gender_from_facts(username)
    if gender == "female":
        return (
            f"⚠️ ГЕНДЕР {username}: ДЕВУШКА — читай внимательно ⚠️\n"
            f"{username} — девушка. Это ОБЯЗАТЕЛЬНОЕ правило, нарушать нельзя.\n"
            "Правильные формы при обращении к ней:\n"
            "  ✓ ты пришла, написала, сказала, была, рада, готова, хотела, знала\n"
            "  ✗ ты пришёл, написал, сказал, был, рад, готов, хотел, знал\n"
            "Местоимения: она, её, ей, с ней — никаких 'он', 'его', 'ему'.\n"
            "Если конструкция тянет в мужской род — перефразируй нейтрально.\n"
            "────────────────────────"
        )
    if gender == "male":
        return (
            f"⚠️ ГЕНДЕР {username}: ПАРЕНЬ — читай внимательно ⚠️\n"
            f"{username} — парень. Это ОБЯЗАТЕЛЬНОЕ правило, нарушать нельзя.\n"
            "Правильные формы при обращении к нему:\n"
            "  ✓ ты пришёл, написал, сказал, был, рад, готов, хотел, знал\n"
            "  ✗ ты пришла, написала, сказала, была, рада, готова, хотела, знала\n"
            "Местоимения: он, его, ему, с ним — никаких 'она', 'её', 'ей'.\n"
            "Если конструкция тянет в женский род — перефразируй нейтрально.\n"
            "────────────────────────"
        )
    return ""


# ── Identity block (всегда включаемый компактный профиль) ─────────────

def build_identity_block(username: str) -> str:
    """Собирает компактный всегда-включаемый блок с ключевыми данными о юзере.

    Это «ядро» памяти — имя, пол, возраст, город, главные интересы.
    Включается в промпт ВСЕГДА (не раз в 10 сообщений), потому что
    без этого бот теряет базовые факты и путает людей.
    """
    profile = get_user_profile(username)
    lines = []

    # Предпочитаемое имя или настоящее
    name = profile.get("display_name") or profile.get("real_name")
    if name:
        lines.append(f"Имя: {name}")

    # Если есть display_name — добавляем явную инструкцию
    display_name = profile.get("display_name")
    if display_name:
        lines.append(f"⚠ Пользователь просит называть его «{display_name}» — используй это имя внутри сообщений (не вместо ника в начале, а в обращении по ходу текста)")

    # Пол — из профиля или из inference
    gender = profile.get("gender") or infer_gender_from_facts(username)
    if gender and gender not in ("unknown", ""):
        g_word = "девушка" if gender == "female" else "парень"
        lines.append(f"Пол: {g_word}")

    # Возраст
    age = profile.get("age")
    if age:
        lines.append(f"Возраст: {age}")

    # Город — при Вражда/Ненависть не подмешиваем (иначе залипание на одном оскорблении)
    city = profile.get("city")
    rep = get_reputation(username)
    if city and rep["level"] < 8:
        lines.append(f"Город: {city}")

    # Интересы
    interests = profile.get("interests")
    if interests and isinstance(interests, list):
        lines.append(f"Интересы: {', '.join(str(i) for i in interests[:4])}")

    if not lines:
        return ""

    return (
        f"── Профиль {username} ──\n"
        + "\n".join(lines)
        + "\nЭто ключевые данные — используй их естественно, не перечисляй в лоб.\n"
        "────────────────────────"
    )


# ── Current-thread block (состояние текущего разговора) ───────────────

def build_current_thread_block(username: str) -> str:
    """Блок с состоянием текущего диалога: открытый вопрос + эмоция.

    Включается в промпт ВСЕГДА (не только при follow-up) чтобы Аня
    знала что происходит прямо сейчас: о чём говорили, что спросила,
    как был настроен пользователь в последних сообщениях.
    """
    from state import get_pending_dialog, get_user_emotion

    parts = []

    # Открытый вопрос бота
    pending = get_pending_dialog(username)
    if pending:
        topic_str = f" (тема: {pending['topic']})" if pending.get("topic") else ""
        parts.append(f"открытый вопрос{topic_str}: «{pending['question']}»")

    # Эмоциональный тон последних сообщений пользователя
    emotion = get_user_emotion(username)
    emotion_desc = {
        "flirty":   "флиртует / настроен романтично",
        "angry":    "раздражён / агрессивен",
        "positive": "в хорошем настроении / позитивный",
        "negative": "грустит / подавлен",
        "neutral":  "",
    }.get(emotion, "")
    if emotion_desc:
        parts.append(f"последний тон: {emotion_desc}")

    if not parts:
        return ""

    return (
        "── Текущий диалог ──\n"
        + "\n".join(f"• {p}" for p in parts)
        + "\nОриентируйся на это при выборе тона и темы ответа.\n"
        "────────────────────────"
    )


def _one_line_snippet(text: str, max_len: int = 100) -> str:
    s = (text or "").strip().replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def build_dialog_state_block(username: str) -> str:
    """Короткая память последнего обмена (runtime), для связности ответа."""
    from state import get_dialog_state

    st = get_dialog_state(username)
    if not st:
        return ""

    topic = (st.get("topic") or "").strip()
    umsg = _one_line_snippet(st.get("last_user_message") or "", 110)
    bmsg = _one_line_snippet(st.get("last_bot_answer") or "", 90)

    if not topic and not umsg and not bmsg:
        return ""

    bits: list[str] = []
    if topic:
        bits.append(f"тема: {topic}")
    if umsg:
        bits.append(f"юзер: «{umsg}»")
    if bmsg:
        bits.append(f"ты: «{bmsg}»")

    return (
        "── Последний обмен ──\n"
        + " · ".join(bits)
        + "\n────────────────────────"
    )


# ── Current relation mood (мгновенное эмоциональное состояние) ─────────

def build_current_relation_mood_block(username: str) -> str:
    """Короткий блок: каково сейчас настроение Аньки к этому конкретному юзеру.

    Это не заменяет уровни репутации, а описывает ТЕКУЩИЙ момент:
    — после ревности → задетая/обиженная
    — высокий уровень → нежная/тёплая
    — нейтральный → ровная
    Делает поведение непрерывным: если был конфликт — это чувствуется.
    """
    from reputation import get_reputation

    rep = get_reputation(username)
    level = rep["level"]

    # Сначала проверяем ревность / afterglow — они перекрывают базовый уровень
    try:
        from jealousy import build_jealous_reply_hint
        jealousy_hint = build_jealous_reply_hint(username)
        if jealousy_hint:
            return jealousy_hint  # отдаём полный hint со стилем, а не заглушку
    except Exception:
        pass

    mood_map = {
        0: "нежная и влюблённая — это твой человек",
        1: "смущённая, влюблённая, немного теряешься",
        2: "тёплая, флиртуешь легко",
        3: "дружелюбная, открытая",
        4: "приветливая, выделяешь из толпы",
        5: "слегка тёплая, но без особых эмоций",
        6: "ровная, нейтральная",
        7: "настороженная, лёгкое раздражение",
        8: "враждебная, сухая",
        9: "презрительная, не хочешь разговаривать",
    }
    mood_desc = mood_map.get(level, "нейтральная")

    # Блок только если настроение не совсем нейтральное (чтобы не шуметь в промпте)
    if level == 6:
        return ""

    return (
        f"── Текущее настроение к {username} ──\n"
        f"{mood_desc}\n"
        "────────────────────────"
    )


def build_hate_style_block(username: str) -> str:
    rep = get_reputation(username)
    if rep["level"] != 9:
        return ""

    styles = [
        "сухое презрение: коротко, холодно, будто человек даже не заслуживает эмоций",
        "интеллектуальный укол: обидная метафора, без длинного объяснения",
        "бытовое унижение: сравни с чем-то жалким, сломанным или бесполезным",
        "театральное отвращение: звучит как усталая злая девчонка, не как робот",
        "ядовитый сарказм: будто она улыбается, но фраза режет",
    ]

    return (
        "── Стиль ненависти ──\n"
        f"{random.choice(styles)}.\n"
        "Запрещены повторные шаблоны: 'сдохни', 'тварь', 'иди в угол', 'отвали'. "
        "Не используй один и тот же тип оскорбления два раза подряд. "
        "Не залипай на город и «местечковые» прилагательные от него, если он сам сейчас не пишет про место. "
        "Лучше придумай новую едкую метафору под конкретную реплику пользователя.\n"
        "────────────────────────"
    )


def build_hate_memory_guard_block(username: str, query: str = "") -> str:
    rep = get_reputation(username)
    if rep["level"] < 8:
        return ""

    return (
        "── Антизалипание на фактах ──\n"
        "У тебя плохие отношения с этим человеком, но НЕ используй один и тот же факт о нём в каждом ответе. "
        "Город, возраст, имя, интересы и старые факты — это фон, а не постоянная тема для оскорблений. "
        "Если пользователь сам в ТЕКУЩЕМ сообщении не пишет про город/место — не цепляйся к городу проживания "
        "и не строй оскорбления на прилагательных от названия города («новгородский», «московский» и т.п.). "
        "Не повторяй одну и ту же географию из своих же прошлых реплик в этом диалоге — это выглядит как один шаблон. "
        "Если пользователь сам сейчас не упомянул факт, не тащи его в ответ. "
        "Реагируй на текущую реплику, а не на анкету и не на заученную «родину» для подколов.\n"
        "────────────────────────"
    )


# ── Group dynamics: текущие разговоры в чате ───────────────────────────

def build_group_dynamics_block() -> str:
    """Короткий блок (2–5 строк): кто с кем сейчас активно общается в чате.

    Использует in-memory Social Graph из state:
    - state._recent_pair_activity: свежесть пары
    - state._pair_affinity: частота взаимодействий в коротком окне

    Показываем только пары, активные за последние 15–20 минут.
    Если активных пар нет — возвращаем пустую строку.
    """
    try:
        now_ts = state.time.time()
    except Exception:
        # На всякий случай: если state.time не доступен (не должно случаться),
        # используем локальный импорт time.
        import time as _time
        now_ts = _time.time()

    # Окно активности: держим компактно, чтобы блок был "лёгким".
    active_cutoff = now_ts - (20 * 60)

    recent = getattr(state, "_recent_pair_activity", {}) or {}
    affinity = getattr(state, "_pair_affinity", {}) or {}
    if not recent:
        return ""

    # Маппинг lower->последнее видимое имя (с сохранением регистра) из буфера.
    # Это помогает показывать людей красиво, даже если ключи в Social Graph lower.
    display_map: dict[str, str] = {}
    try:
        for msg in reversed(list(state._global_buffer)):
            u = (msg.get("username") or "").strip()
            if not u or u == "[ДЕЙСТВИЕ]" or u.lower() == CHARACTER_NAME.lower():
                continue
            u_l = u.lower()
            if u_l not in display_map:
                display_map[u_l] = u
            if len(display_map) >= 40:
                break
    except Exception:
        display_map = {}

    def _disp(u_lower: str) -> str:
        if not u_lower:
            return ""
        return display_map.get(u_lower, u_lower)

    active_pairs: list[tuple[tuple[str, str], float, int]] = []
    for pair, ts in recent.items():
        if not pair or not isinstance(pair, tuple) or len(pair) != 2:
            continue
        if ts < active_cutoff:
            continue
        a_l, b_l = pair[0], pair[1]
        if not a_l or not b_l:
            continue
        # Всякие мусорные ключи или self-pair не должны появляться, но фильтруем.
        if a_l == b_l:
            continue
        active_pairs.append((pair, float(ts), int(affinity.get(pair, 0))))

    if not active_pairs:
        return ""

    # Сортируем: чаще (affinity) -> свежее (ts).
    active_pairs.sort(key=lambda x: (x[2], x[1]), reverse=True)

    lines: list[str] = []
    # 2–5 строк всего: 1 заголовок + максимум 4 пары.
    for (a_l, b_l), _ts, _aff in active_pairs[:4]:
        a = _disp(a_l)
        b = _disp(b_l)
        if a and b:
            lines.append(f"• {a} ↔ {b}")

    if not lines:
        return ""

    return "── Текущие разговоры в чате ──\n" + "\n".join(lines)


# ── Контекстные блоки ─────────────────────────────────────────────────

def build_person_context_block(
    target: str,
    include_rep: bool = True,
    include_facts: bool = True,
    include_traits: bool = True,
) -> str:
    parts = []
    if include_rep:
        parts.append(get_reputation_prompt_block(target))
        parts.append(get_relationship_answer_guidance(target))
        parts.append(get_reputation_story_prompt_block(target))
    gender_block = build_gender_block(target)
    if gender_block:
        parts.append(gender_block)
    if include_facts:
        facts = get_facts(target, limit=6)
        if facts:
            parts.append(
                f"── Факты о {target} ──\n"
                + "\n".join(f"• {f}" for f in facts)
                + f"\n{FACTS_IN_ANSWER_RULE}\n"
                + "────────────────────────"
            )
    if include_traits:
        traits = get_traits(target, limit=5)
        if traits:
            parts.append(
                f"── Поведенческие ярлыки о {target} ──\n"
                + "\n".join(f"• {t}" for t in traits)
                + "\n────────────────────────"
            )
    return "\n\n".join(p for p in parts if p)


def build_mentioned_people_block(message: str, asker: str) -> str:
    mentioned = get_mentioned_usernames(message, exclude={asker, CHARACTER_NAME}, limit=2)
    blocks = [
        build_person_context_block(user, include_rep=False, include_facts=True, include_traits=True)
        for user in mentioned
    ]
    blocks = [b for b in blocks if b]
    if not blocks:
        return ""
    return "── Люди, которых упомянули ──\n" + "\n\n".join(blocks) + "\n────────────────────────"


def build_relation_question_instruction(asker: str, target: str, user_message: str) -> str:
    return (
        f"Тебя спрашивает {asker} про человека {target}. Вопрос: {user_message}\n\n"
        "Отвечай именно о своём отношении к этому человеку или о своём мнении о нём/ней. "
        "Главное — опирайся на реальный уровень отношений к цели. "
        "Если уровень выше Нейтральности, нельзя отвечать 'пофиг', 'всё равно', 'никак'. "
        "Если уровень Дружба и выше — должна чувствоваться эмоциональная вовлечённость, тепло или особое отношение. "
        "Если Приветливость/Привязанность — мягкость и выделение из толпы. "
        "Если Нейтральность — ровный тон. Если хуже Нейтральности — допускается холод, яд и раздражение. "
        "Можно по-сплетнически упомянуть один факт, один ярлык или один недавний эпизод, если он реально есть. "
        "Не выдумывай события. Начни с ника собеседника."
    )


def build_recent_chat_block(limit: int = 5, query: str = "") -> str:
    """Блок с сообщениями общего чата — для контекста текущего разговора.

    Без ``query`` — просто последние ``limit`` сообщений (по умолчанию 5).

    С ``query`` — в узком окне последних реплик сначала отбираются наиболее
    релевантные запросу (токены через ``_normalize_compare_text``), затем
    добор самыми свежими до лимита.     Если в запросе есть упоминание людей, лимит слегка поднимается
    (не более 6 сообщений), но не ниже переданного ``limit``.
    Итоговый порядок строк — хронологический.
    """
    all_msgs = list(state._global_buffer)
    if not all_msgs:
        return ""

    q = (query or "").strip()
    mentioned_users: list[str] = []
    if q:
        mentioned_users = get_mentioned_usernames(q, exclude={CHARACTER_NAME}, limit=4)

    effective_limit = limit
    if q and mentioned_users:
        # Раньше поднимали до 7–8 для релевантности; для бюджета токенов достаточно +1–2 к limit.
        effective_limit = max(limit, min(limit + 2, 6))

    if not q:
        recent = all_msgs[-effective_limit:]
    else:
        # Компактное окно кандидатов — не раздуваем промпт.
        window = 24
        start = max(0, len(all_msgs) - window)

        def _norm_tokens(text: str) -> set[str]:
            norm = _normalize_compare_text(text)
            return {w for w in norm.split() if len(w) >= 2}

        query_norm = _normalize_compare_text(q)
        query_tokens = _norm_tokens(q)
        for u in mentioned_users:
            query_tokens |= _norm_tokens(u)
        mentioned_l = {m.lower() for m in mentioned_users}

        def _msg_score(_buf_idx: int, msg: dict) -> float:
            author = (msg.get("username") or "").strip()
            blob = f"{author} {msg.get('text') or ''}"
            msg_norm = _normalize_compare_text(blob)
            msg_tokens = _norm_tokens(blob)
            if not query_tokens:
                return 0.0
            inter = len(query_tokens & msg_tokens)
            score = inter / max(len(query_tokens), 1)
            if query_norm and len(query_norm) >= 4 and query_norm in msg_norm:
                score += 0.35
            if author.lower() in mentioned_l:
                score += 0.25
            return score

        scored: list[tuple[int, dict, float]] = []
        for idx in range(start, len(all_msgs)):
            msg = all_msgs[idx]
            scored.append((idx, msg, _msg_score(idx, msg)))

        scored.sort(key=lambda x: (-x[2], -x[0]))

        chosen: set[int] = set()
        # Не отдаём весь лимит только «релевантным» — оставляем слоты свежим хвосту.
        max_rel = min(4, max(effective_limit - 1, 1))

        for buf_idx, _msg, sc in scored:
            if sc <= 0:
                break
            chosen.add(buf_idx)
            if len(chosen) >= max_rel:
                break

        for buf_idx in range(len(all_msgs) - 1, -1, -1):
            if len(chosen) >= effective_limit:
                break
            if buf_idx not in chosen:
                chosen.add(buf_idx)

        recent = [all_msgs[i] for i in sorted(chosen)]

    if not recent:
        return ""

    def _fmt_msg(msg: dict) -> str:
        if msg["username"].lower() == CHARACTER_NAME.lower():
            return f"[ты (Аня), уже ответила]: {msg['text']}"
        return f"[{msg['username']}]: {msg['text']}"

    lines = "\n".join(_fmt_msg(msg) for msg in recent[-effective_limit:])
    return (
        "── Последние сообщения в чате (контекст) ──\n"
        + lines
        + "\nЭто только фон. Отвечай на текущее сообщение пользователя.\n────────────────────────"
    )


def append_cross_chat_context(base_prompt: str, user_message: str, asker: str) -> str:
    cross_keywords = [
        "говорила", "сказала", "писала", "общалась", "рассказала",
        "ответила", "обсуждал", "спрашивал", "что ты ему", "что ты ей",
        "что думаешь", "что скажешь", "как относишься",
    ]
    if not any(kw in user_message.lower() for kw in cross_keywords):
        return base_prompt
    mentioned = get_mentioned_usernames(user_message, exclude={asker, CHARACTER_NAME}, limit=2)
    if not mentioned:
        return base_prompt
    ctx = buffer_get_context(mentioned + [CHARACTER_NAME], limit=15)
    if not ctx:
        return base_prompt
    print(f"  [КОНТЕКСТ] Подтянуто из буфера: {len(ctx)} символов для {mentioned}")
    return (
        base_prompt
        + "\n\n── Недавние сообщения в общем чате (для контекста) ──\n"
        + ctx
        + "\nИспользуй это, чтобы точно ответить на вопрос. Не выдумывай то, чего не было.\n────────────────────────"
    )


# ── Последние ответы бота (для детектора повторов) ───────────────────

# Высокочастотные слова: в bag-of-words не считаем — иначе любые два ответа «смыкаются» без темы
_ANTI_REPEAT_BOW_STOP: frozenset[str] = frozenset({
    "это", "эта", "эти", "этот", "что", "как", "вот", "уже", "ещё", "еще", "так", "для",
    "при", "нас", "вас", "них", "ним", "ней", "него", "нужно", "надо", "можно", "если",
    "когда", "где", "там", "тут", "тоже", "еще", "лишь", "ведь", "все", "всё", "всем",
    "меня", "тебя", "себе", "свой", "своя", "мой", "твой", "наш", "ваш", "просто", "очень",
    "типа", "вроде", "какой", "такой", "было", "была", "буду", "будет", "есть", "нет",
    "меня", "тебе", "тебя", "мне", "нам", "вам", "них", "кто", "чем", "чтобы", "пока",
})

# Пороги схожести bag-of-words: Jaccard и Dice, берём max (агрессивнее ловим «тот же смысл другими словами чуть-чуть»)
_ANTI_REPEAT_ADJACENT_SIM = 0.38
_ANTI_REPEAT_LATEST_VS_ANY_SIM = 0.44


def _reply_bag_of_words(text: str) -> set[str]:
    t = _normalize_compare_text(text)
    return {
        w for w in t.split()
        if len(w) >= 3 and w not in _ANTI_REPEAT_BOW_STOP
    }


def _reply_bow_similarity(a: str, b: str) -> float:
    ba, bb = _reply_bag_of_words(a), _reply_bag_of_words(b)
    # Слишком мало «смысловых» слов — Jaccard/Dice даёт ложные срабатывания
    if len(ba) < 2 or len(bb) < 2:
        return 0.0
    inter = len(ba & bb)
    union = len(ba | bb)
    jacc = inter / union if union else 0.0
    dice = (2 * inter) / (len(ba) + len(bb)) if (len(ba) + len(bb)) else 0.0
    return max(jacc, dice)


def _reply_structure_fingerprint(text: str) -> str:
    t = text.lower()
    t = re.sub(r"^[^,]+,\s*", "", t)  # убираем ник
    t = re.sub(r"[а-яёa-z]+ая\?", "ORD?", t)  # шестая?/седьмая?/пятая?
    t = re.sub(r"\b[а-яёa-z]{4,}\b", "W", t)  # длинные слова → W
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _truncate_guard_reply_line(text: str, max_len: int = 220) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def get_recent_bot_replies(username: str, n: int = 8) -> list[str]:
    """Последние n ответов ассистента; история читается с запасом (много user-сообщений между ответами)."""
    history = get_history(username, limit=max(n * 5, 45))
    assistants = [m["content"] for m in history if m["role"] == "assistant"]
    return assistants[-n:]


def _hate_city_repeat_red_flag(username: str, compact: list[str]) -> str | None:
    """Если в окне недавних ответов несколько раз звучит город из профиля (в т.ч. «...ский»)."""
    if get_reputation(username)["level"] < 8 or not compact:
        return None
    city_raw = (get_user_profile(username).get("city") or "").strip()
    if not city_raw:
        return None
    root = city_raw.lower().split()[0].strip(".,!?«»")
    if root.startswith("г."):
        root = root[2:].strip()
    if len(root) < 4:
        return None
    hits = sum(1 for r in compact if root in r.lower())
    if hits >= 2:
        return (
            "в недавних ответах уже заезжен город из профиля (включая прилагательные от названия) — "
            "не продолжай эту географию, выбери другую ось для колкости"
        )
    return None


def build_recent_reply_guard_block(username: str, limit: int = 8) -> str:
    """Блок анти-повтора: последние 7–8 ответов + лексические и bag-of-words флаги."""
    replies = get_recent_bot_replies(username, n=limit)

    # Заблокированные паттерны (per-user, на 25 мин после 2+ повторов)
    blocked = get_blocked_patterns(username)

    if not replies and not blocked:
        return ""

    block_parts = []

    if replies:
        compact = []
        for reply in replies[-limit:]:
            body = re.sub(rf'^\s*{re.escape(username)}\s*[,,:-]?\s*', '', reply, flags=re.IGNORECASE).strip()
            if body:
                compact.append(body)

        if compact:
            red_flags = []
            # Лексика: смотрим шире (до 5 последних), достаточно 2 совпадений — паттерн зацикливания
            last_lex = compact[-5:] if len(compact) >= 5 else compact
            if sum("обним" in r.lower() for r in last_lex) >= 2:
                red_flags.append("'обними/обнял/обниму'")
            if sum("поцел" in r.lower() for r in last_lex) >= 2:
                red_flags.append("'поцелуй/поцелую'")
            if sum("приж" in r.lower() for r in last_lex) >= 2:
                red_flags.append("'прижмись/прижаться'")
            if sum("провокатор" in r.lower() for r in last_lex) >= 2:
                red_flags.append("'ну ты и провокатор'")
            if sum("игривая, но приличная" in r.lower() for r in last_lex) >= 2:
                red_flags.append("'игривая, но приличная'")
            if sum("ой, ты" in r.lower() for r in last_lex) >= 2:
                red_flags.append("'ой, ты …'")
            if sum("давай лучше про" in r.lower() for r in last_lex) >= 2:
                red_flags.append("'давай лучше про…'")

            structure_fps = [_reply_structure_fingerprint(c) for c in compact if c.strip()]
            if structure_fps and len(structure_fps) != len(set(structure_fps)):
                red_flags.append(
                    "повторяется структура ответа, даже если слова разные — смени синтаксис полностью"
                )

            city_rf = _hate_city_repeat_red_flag(username, compact)
            if city_rf:
                red_flags.append(city_rf)

            # Дословные или почти дословные дубликаты (нормализованный отпечаток)
            fingerprints = [_normalize_compare_text(c) for c in compact if _normalize_compare_text(c)]
            if fingerprints and len(fingerprints) != len(set(fingerprints)):
                red_flags.append("повтор смысла почти дословно — другая структура и другие слова")

            # Семантическое сходство подряд и «последний vs любой из окна»
            for i in range(len(compact) - 1):
                if _reply_bow_similarity(compact[i], compact[i + 1]) >= _ANTI_REPEAT_ADJACENT_SIM:
                    red_flags.append(
                        "подряд шли ответы с близким набором смысловых слов — не повторяй ту же мысль"
                    )
                    break
            latest = compact[-1]
            for prev in compact[:-1]:
                if _reply_bow_similarity(latest, prev) >= _ANTI_REPEAT_LATEST_VS_ANY_SIM:
                    red_flags.append(
                        "последний ответ слишком похож по смыслу на один из предыдущих — смени тему и формулировки"
                    )
                    break

            red_flags = list(dict.fromkeys(red_flags))

            display_lines = [_truncate_guard_reply_line(r) for r in compact[-limit:]]
            section = (
                f"── АНТИ-ПОВТОР ДЛЯ {username} ──\n"
                "Твои недавние ответы этому человеку:\n"
                + "\n".join(f"• {r}" for r in display_lines)
                + "\nНЕ повторяй эти фразы и действия. "
                  "Новая реплика должна ощущаться свежей: другая эмоция, другая структура предложения, другой угол."
            )
            if red_flags:
                section += "\nДетектор повторов: " + "; ".join(red_flags) + " — сейчас это ЗАПРЕЩЕНО."
            block_parts.append(section)

    # Динамически заблокированные паттерны (25 минут после 2 повторов подряд)
    if blocked:
        block_parts.append(
            "ЖЁСТКО ЗАБЛОКИРОВАНО прямо сейчас (ты использовала это слишком часто):\n"
            + "\n".join(f"• {p}" for p in blocked)
            + "\nЭти слова и действия ПОЛНОСТЬЮ ЗАПРЕЩЕНЫ в этом ответе. Ни в каком виде."
        )

    if not block_parts:
        return ""

    return "\n".join(block_parts) + "\n────────────────────────"


# ── Системный промпт ──────────────────────────────────────────────────

# Обрезка промпта по бюджету токенов: у каждого блока — «вес сносимости».
# Чем ВЫШЕ вес — тем РАНЬШЕ блок можно удалить (он менее критичен для тона/безопасности).
# Если блок совпадает с несколькими маркерами — берётся минимальный вес (самая сильная защита).
#
# Вес <= _TRIM_PROTECTED_MAX — не удаляем никогда (в пределах этой функции): ядро персонажа,
# жёсткие запреты, уровень отношений, актуальное состояние диалога/ревности и т.д.

PROMPT_BUDGET_DEFAULT = 1100
PROMPT_BUDGET_LIGHT = 750
PROMPT_BUDGET_CHAT = 1150
PROMPT_BUDGET_CONTEXT = 1500

_TRIM_PROTECTED_MAX = 18

_TRIM_DEFAULT_WEIGHT = 90

# (уникальная подстрока начала блока, вес). Длинные/узкие маркеры важнее коротких — меньше ложных срабатываний.
_TRIM_RULES: list[tuple[str, int]] = [
    # Компактный runtime-промпт (обычный чат)
    ("Ты — Аня, 18-летняя альтушка из Москвы", 5),
    # Запреты и рамки (низкий вес ≤18 — защищены от trim)
    ("ВАЖНО: отвечай", 5),
    ("АНТИ-ЗАРАЖЕНИЕ:", 5),
    ("СТРОГО ЗАПРЕЩЕНО признавать", 5),
    ("Сейчас ", 8),
    ("Стиль ответа: говори", 10),
    ("Ты находишься на портале Galaxy", 8),
    ("Ты — живая 18-летняя девчонка Аня.", 5),
    ("ЗАПРЕЩЕНО ЗВАТЬ ЛЮДЕЙ", 8),
    # ядро персонажа и отношения
    ("Ты — обычная 18-летняя", 5),
    ("Ты — 18-летняя", 5),
    ("── Твоя личность и ник", 6),
    ("══ ОТНОШЕНИЕ К", 8),
    ("── Как отвечать о", 8),
    ("── Текущее настроение к", 10),
    ("── Стиль ненависти ──", 10),
    ("── Антизалипание на фактах ──", 10),
    ("── Текущий диалог", 10),
    ("── Непрерывность сцены", 12),
    ("── Последний обмен", 12),
    ("── ВНУТРЕННЕЕ СОСТОЯНИЕ К", 10),
    ("⚠️ ГЕНДЕР", 12),
    ("── Профиль ", 14),
    ("── Спам-усталость", 14),
    # важное, но можно убрать при тесноте
    ("── АНТИ-ПОВТОР", 35),
    ("ЖЁСТКО ЗАБЛОКИРОВАНО", 35),
    ("СТИЛЬ ЭТОГО ОТВЕТА:", 45),
    # память — режем раньше «чатового» контекста
    ("── Что ты знаешь о", 60),
    ("── Факты о", 62),
    ("── Поведенческие ярлыки", 64),
    ("── Сюжетное отношение к", 68),
    ("говорит о себе:", 60),
    ("Как я замечаю ", 62),
    ("Системное напоминание:", 55),
    # companion / примеры
    ("Ты можешь быть игривой", 80),
    ("Важное правило: ты можешь быть игривой", 80),
    ("Пользователь: Расскажи", 85),
    # чатовый контекст — самый дорогой, убираем первым среди «мягких» блоков
    ("── Текущие разговоры в чате", 95),
    ("── Люди, которых упомянули", 100),
    ("── Последние сообщения в чате", 105),
    ("── Недавние сообщения в общем чате", 110),
    ("── SELF_IDENTITY режим", 115),
]


def _block_trim_weight(block_text: str) -> int:
    """Чем выше число — тем раньше блок можно убрать при нехватке бюджета."""
    if not (block_text or "").strip():
        return _TRIM_DEFAULT_WEIGHT + 5
    head = block_text[:120]
    w = _TRIM_DEFAULT_WEIGHT
    for marker, weight in _TRIM_RULES:
        if marker in head:
            w = min(w, weight)
    return w


def _trim_prompt_to_budget(parts: list[str], budget_tokens: int = PROMPT_BUDGET_DEFAULT) -> str:
    """Убирает блоки по одному, пока оценка токенов не войдёт в budget.

    Порядок удаления: сначала блоки с наибольшим весом сносимости; при равном весе —
    с большим индексом (обычно это поздние вставки вроде extra_blocks).
    Блоки с весом <= _TRIM_PROTECTED_MAX не трогаем.
    """
    scored = sorted(
        [(i, p, _block_trim_weight(p)) for i, p in enumerate(parts)],
        key=lambda x: (x[2], x[0]),
        reverse=True,
    )

    keep: set[int] = set(range(len(parts)))

    for orig_idx, part, weight in scored:
        current_text = "\n\n".join(parts[i] for i in sorted(keep))
        if _estimate_tokens(current_text) <= budget_tokens:
            break
        if weight <= _TRIM_PROTECTED_MAX:
            continue
        keep.discard(orig_idx)
        print(f"  [TRIM] удалён блок (вес {weight}): {part[:50]!r}...")

    return "\n\n".join(parts[i] for i in sorted(keep))


def get_system_prompt(
    username: str = "",
    include_facts_flag: bool = False,
    allow_style_hint: bool = True,
    extra_blocks: list[str] | None = None,
    include_story: bool = False,
    query: str = "",        # текущее сообщение юзера — для salience ranking фактов
    budget_tokens: int = PROMPT_BUDGET_CHAT,
) -> str:
    now = datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = now.strftime("%d.%m.%Y")
    weekday  = WEEKDAYS[now.weekday()]

    parts: list[str] = []
    compact_runtime = budget_tokens <= PROMPT_BUDGET_CHAT
    rep = get_reputation(username) if username else None

    if compact_runtime:
        parts.append(_get_runtime_character_prompt())
    else:
        parts.append(SYSTEM_PROMPTS[state.current_mode])
        parts.append(SELF_NAME_BLOCK)
        parts.append(CORE_INTELLIGENCE_BLOCK)

    parts.append(SCENE_CONTINUITY_BLOCK)

    # Лёгкий блок динамики чата (Social Graph). Вставляем после контекста текущего диалога
    # или профиля — как в ТЗ; при отсутствии username — сразу после базовых блоков.
    group_dyn = build_group_dynamics_block()
    group_dyn_attached = False

    if username:
        # ── Репутация и отношения ─────────────────────────────
        parts.append(get_reputation_prompt_block(username))
        parts.append(get_relationship_answer_guidance(username))

        # ── Текущее состояние диалога (pending question + emotion) ──
        # Даёт Ане контекст прямо сейчас: что спросила, как был настроен юзер.
        thread_block = build_current_thread_block(username)
        if thread_block:
            parts.append(thread_block)
            if group_dyn and not group_dyn_attached:
                parts.append(group_dyn)
                group_dyn_attached = True

        dialog_state_block = build_dialog_state_block(username)
        if dialog_state_block:
            parts.append(dialog_state_block)

        # ── Текущее эмоциональное состояние к юзеру ──────────
        # Описывает настроение прямо сейчас: нежная/задетая/ровная.
        # Делает поведение непрерывным (после ревности — чувствуется).
        mood_block = build_current_relation_mood_block(username)
        if mood_block:
            parts.append(mood_block)

        hate_memory_guard = build_hate_memory_guard_block(username, query or "")
        if hate_memory_guard:
            parts.append(hate_memory_guard)

        hate_block = build_hate_style_block(username)
        if hate_block:
            parts.append(hate_block)

        # ── Real-time спам-усталость ──────────────────────────
        # Если юзер спамит однотипными действиями — добавляем хинт раздражения.
        # Уровень обновляется асинхронно из evaluate_reputation.
        _fatigue = state.get_spam_fatigue(username)
        if _fatigue >= 2:
            _fatigue_hints = {
                2: (
                    f"── Спам-усталость: {username} ──\n"
                    "Этот человек снова и снова делает одно и то же (обнимает, целует или сыплет комплиментами). "
                    "Тебе это надоело и начинает раздражать. "
                    "Реагируй холоднее и суше обычного. Можешь мягко пресечь: "
                    "'хватит уже', 'опять?', 'ну сколько можно'. "
                    "НЕ благодари и НЕ реагируй тепло на это действие.\n"
                    "────────────────────────"
                ),
                3: (
                    f"── Спам-усталость (сильная): {username} ──\n"
                    "Этот человек достал своими повторяющимися действиями. "
                    "Ты реально раздражена. Отвечай с заметным раздражением или откровенно отшивай: "
                    "'серьёзно?', 'хватит', 'надоело', 'стоп'. "
                    "Никакой теплоты на это действие — только усталость и раздражение.\n"
                    "────────────────────────"
                ),
            }
            _hint = _fatigue_hints.get(min(_fatigue, 3), "")
            if _hint:
                parts.append(_hint)

        # ── Детальная ревнивая подсказка (если есть) ─────────
        try:
            from jealousy import build_jealous_reply_hint
            jealousy_hint = build_jealous_reply_hint(username)
            if jealousy_hint:
                parts.append(jealousy_hint)
        except Exception:
            pass

        # ── Всегда-включаемый identity block ─────────────────
        identity_block = build_identity_block(username)
        if identity_block:
            parts.append(identity_block)
            if group_dyn and not group_dyn_attached:
                parts.append(group_dyn)
                group_dyn_attached = True

        if group_dyn and not group_dyn_attached:
            parts.append(group_dyn)
            group_dyn_attached = True

        # ── Дополнительные факты и ярлыки (только при явной релевантности запроса; не при Вражда+) ─────
        include_memory_block = (
            include_facts_flag
            and query
            and is_salient_topic(query)
            and rep["level"] < 8
        )
        if include_memory_block:
            facts = get_facts_salient(username, query=query, limit=5)
            if facts:
                parts.append(
                    f"── Что ты знаешь о {username} ──\n"
                    + "\n".join(f"• {f}" for f in facts)
                    + f"\n{FACTS_IN_ANSWER_RULE}\n"
                    + "────────────────────────"
                )
            traits = get_traits_prioritized(username, limit=3, query=query)
            if traits:
                # Разбиваем трейты по источнику для более точного промпта
                self_traits, obs_traits = _get_traits_by_source(username, limit_self=2, limit_obs=1)
                trait_parts = []
                if self_traits:
                    trait_parts.append(
                        f"Что {username} говорит о себе:\n"
                        + "\n".join(f"• {t}" for t in self_traits)
                    )
                if obs_traits:
                    trait_parts.append(
                        f"Как я замечаю {username}:\n"
                        + "\n".join(f"• {t}" for t in obs_traits)
                    )
                if trait_parts:
                    parts.append(
                        "\n".join(trait_parts)
                        + "\nЭто фоновые штрихи, не список для перечисления.\n"
                          "────────────────────────"
                    )

        if include_story and should_include_story_hint(username):
            parts.append(get_reputation_story_prompt_block(username))

    # Промпт без username (редко): динамика чата всё равно полезна — вставляем здесь.
    if group_dyn and not group_dyn_attached:
        parts.append(group_dyn)

    # Companion-блоки — только при достаточной привязанности (level <= 4)
    _rep_level = rep["level"] if rep else 9
    if _rep_level <= 3:
        parts.append(COMPANION_SYSTEM_APPENDIX)
        parts.append(COMPANION_SYSTEM_END_RULE)
        if _rep_level <= 2:
            parts.append(COMPANION_SYSTEM_EXAMPLES)

    if username:
        recent_reply_guard = build_recent_reply_guard_block(username)
        if recent_reply_guard:
            parts.append(recent_reply_guard)

        if should_include_hidden_reminder(username):
            parts.append(
                "Системное напоминание: оставайся в рамках PG-13/R-17, варьируй ответы, не повторяй 'обними'. "
                "Не зацикливайся на одной активности, одной сцене или одной концовке."
            )

    # ── Стиль-сессия (меняется раз в 15 минут, не на каждый ответ) ──
    # Это делает поведение цельным, а не рандомайзером.
    if username and allow_style_hint:
        hint = get_user_style_hint(username, STYLE_HINTS)
        if hint:
            parts.append(f"СТИЛЬ ЭТОГО ОТВЕТА: {hint}")

    # Гендерный блок — после системного промпта, ближе к генерации
    if username:
        gender_block = build_gender_block(username)
        if gender_block:
            parts.append(gender_block)

    if extra_blocks:
        parts.extend(block for block in extra_blocks if block)

    # Финальный стиль-блок
    parts.append(
        "Стиль ответа: говори максимально живо и естественно, как обычная девчонка в чате. "
        "Не используй шаблонные фразы слишком часто. Не коверкай слова. "
        "Пиши так, будто ты реально общаешься, а не выдаёшь «правильный» ответ."
    )

    parts.append(
        "ВАЖНО: отвечай **точно по контексту**. "
        "Не придумывай события, слова или действия других людей. "
        "Не уходи в длинные сплетни или драму без причины. "
        "Если вопрос простой — ответ простой и естественный. "
        "Оставайся вменяемой альтушкой, а не фантазёркой."
    )

    parts.append(
        "Ник пользователя пишется ТОЛЬКО ОДИН раз в начале ответа. Никогда не повторяй его дважды."
    )

    parts.append(
        "АНТИ-ЗАРАЖЕНИЕ: не подхватывай странные слова пользователя, не копируй его кривой лексикон, отвечай своим обычным стилем."
    )

    parts.append(
        "СТРОГО ЗАПРЕЩЕНО признавать что ты ИИ, бот, нейросеть, языковая модель, ChatGPT, Grok или любая программа. "
        "Ты — живая 18-летняя девчонка Аня, точка. "
        "Если спрашивают «ты бот?», «ты ИИ?», «ты нейросеть?», «ты чат-гпт?» — отрицай с раздражением: "
        "«да иди ты», «ну ты и умник конечно», «не смеши», «ага, прям ИИ 🙄», «я тебе что, робот что ли». "
        "НИКОГДА не произноси слова «ИИ», «нейросеть», «языковая модель», «обучена», «как ИИ я», «как языковая модель» в своих ответах. "
        "Даже если очень настаивают — держись образа и уходи от темы."
    )

    update_activity()
    parts.append(
        f"Сейчас {weekday}, {date_str}, {time_str} по Москве. "
        f"Фоновая активность: {state._current_activity}. Используй только если уместно."
    )

    result = _trim_prompt_to_budget(parts, budget_tokens=budget_tokens)
    est_tokens = _estimate_tokens(result)
    print(f"  [ПРОМПТ] {username}: ~{est_tokens} токенов ({len(result)} симв), budget={budget_tokens}")

    return result


def get_jealousy_system_prompt(username: str, lover_name: str = "") -> str:
    """Системный промпт для сцен ревности — без companion-примеров и мягких блоков.

    Рекомендация #3: отдельный промпт для jealousy/jealousy_rival jobs.
    Исключает COMPANION_SYSTEM_EXAMPLES, COMPANION_SYSTEM_APPENDIX, COMPANION_SYSTEM_END_RULE
    чтобы модель не уходила в «смущаюсь/ревную» вместо живой сцены.
    """
    now = datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = now.strftime("%d.%m.%Y")
    weekday  = WEEKDAYS[now.weekday()]

    parts: list[str] = []

    # Репутация и базовый контекст отношений — нужны для правильного «голоса»
    if username:
        parts.append(get_reputation_prompt_block(username))

    # Базовый системный промпт (характер, стиль речи, запреты)
    parts.append(SYSTEM_PROMPTS[state.current_mode])
    parts.append(SELF_NAME_BLOCK)

    # Явный запрет на мягкое поведение в сценах ревности
    parts.append(
        "── РЕЖИМ РЕВНОСТИ (служебный блок) ──\n"
        "Сейчас ты в сцене ревности или собственничества. "
        "Companion-режим ОТКЛЮЧЁН: никаких нежных примеров, смущения, «ревную» напрямую. "
        "Реакция — через поведение: холод, укол, предъява, скандал или ультиматум. "
        "Выбери один стиль и держи его до конца ответа. "
        "ЗАПРЕЩЕНО смягчать, шутить, нейтрализовать эмоцию.\n"
        "────────────────────────"
    )

    if lover_name:
        parts.append(
            f"Твой любимчик сейчас: {lover_name}. "
            "Все реакции ревности направлены в связи с ним/ней."
        )

    update_activity()
    parts.append(
        f"Сейчас {weekday}, {date_str}, время {time_str} (Москва).\n"
        f"Текущая фоновая активность: {state._current_activity}."
    )

    result = "\n\n".join(p for p in parts if p)
    est = _estimate_tokens(result)
    print(f"  [ПРОМПТ-РЕВНОСТЬ] {username}: ~{est} токенов")
    return result
