"""
jealousy.py — отслеживание ревности к любимчику.

Логика:
- следим только за текущим любимчиком (уровень Любовь / is_lover)
- учитываем не только факт игнора, но и контекст общения с другими
- ОБЫЧНОЕ общение с кем-то другим снимает 1 очко, НО ТОЛЬКО в режиме ревности
- явное внимание / заигрывание снимает сильнее (всегда, независимо от режима)
- откровенный флирт может снять до 5 очков (всегда)
- прямые обращения к боту НЕ считаются публичным игнором,
  но если любимчик при этом обсуждает поцелуи/флирт с кем-то другим —
  это тоже триггерит ревность
- если рядом с любимчиком явно флиртует соперник, это усиливает злость —
  реакция срабатывает даже если соперник не назвал ник любимчика явно,
  при условии что любимчик недавно был активен в чате
- после вспышки ревности бот ещё некоторое время злая / колючая
- в этом состоянии с шансом 10% может просто проигнорировать сообщение любимчика
"""

from __future__ import annotations

import random
import re
import time
from collections import deque

from config import (
    CHARACTER_NAME,
    JEALOUSY_AFTERGLOW_MAX,
    JEALOUSY_AFTERGLOW_MIN,
    JEALOUSY_COOLDOWN,
    JEALOUSY_DIRECT_REPLY_HINT_TTL,
    JEALOUSY_IGNORE_CHANCE,
    JEALOUSY_IGNORE_CHANCE_AFTERGLOW,
    JEALOUSY_INSTANT_PENALTY_THRESHOLD,
    JEALOUSY_POST_RETURN_STRIKES,
    JEALOUSY_PUBLIC_MESSAGE_TRIGGER,
    JEALOUSY_PUBLIC_MESSAGE_TRIGGER_SOFT,
    JEALOUSY_RECENT_PUBLIC_LIMIT,
    JEALOUSY_SOFT_RESET_DECREMENT,
)
from reputation import apply_jealousy_penalty, get_all_reputations, get_current_lover
from database import get_user_profile
import state


_FLIRT_MARKERS = (
    "люблю", "скучаю", "скучал", "скучала", "малыш", "малышка", "солнышко", "зая",
    "зайка", "котик", "красотка", "красавица", "красивый", "красивая", "милая", "милый",
    "обним", "поцел", "поцелует", "поцелую", "поцелуй", "целую", "целуй", "поцеловал",
    "поцеловала", "страстно", "хочу тебя", "ты моя", "ты мой", "ты мне нрав", "без тебя",
    "сердце", "сердечко", "лапочка", "сладкий", "сладкая", "ревнуй", "женюсь", "выйдешь за",
    "любимая", "любимый", "в щечку", "в щёчку", "десерт",
)

_ATTENTION_MARKERS = (
    "как дела", "где ты", "иди сюда", "давай сюда", "подойди", "жду тебя", "только ты",
    "именно ты", "с тобой", "к тебе", "ответь", "напиши", "пойдём", "пойдем",
)

_FLIRT_EMOJIS = ("❤", "❤️", "💋", "😘", "😍", "🥰", "💕", "💞", "💖", "💜")


# ── Базовое состояние ────────────────────────────────────────────────

def _reset_state(new_lover: str = ""):
    state._jealousy_current_lover = new_lover
    now = time.time() if new_lover else 0.0
    state._jealousy_last_direct_at = now
    state._jealousy_last_public_at = 0.0
    state._jealousy_public_ignored_count = 0
    state._jealousy_stage = 0
    state._jealousy_last_reaction_at = 0.0
    state._jealousy_reply_hint_until = 0.0
    state._jealousy_afterglow_until = 0.0
    state._jealousy_afterglow_stage = 0
    state._jealousy_recent_public = deque(maxlen=JEALOUSY_RECENT_PUBLIC_LIMIT)
    state._jealousy_last_rival = ""
    state._jealousy_last_penalty = 0
    state._jealousy_last_context_label = ""
    state._jealousy_last_context_excerpt = ""
    state._jealousy_post_return_strikes = 0
    state._jealousy_style = ""


def sync_current_lover() -> str:
    """Синхронизирует любимчика из БД с in-memory состоянием ревности."""
    lover = get_current_lover() or ""
    if lover != state._jealousy_current_lover:
        print(f"[РЕВНОСТЬ] Любимчик изменился: '{state._jealousy_current_lover}' → '{lover}'")
        _reset_state(lover)
    return lover


# ── Поиск соперника / адресата ───────────────────────────────────────

def _known_other_users(lover: str) -> list[str]:
    users: set[str] = set()
    users.update(r["username"] for r in get_all_reputations() if r.get("username"))
    users.update(msg["username"] for msg in state._global_buffer if msg.get("username"))
    users.discard(lover)
    users.discard(CHARACTER_NAME)
    users.discard("[ДЕЙСТВИЕ]")
    return sorted((u for u in users if u), key=len, reverse=True)


def _text_mentions_user(text: str, username: str) -> bool:
    t = (text or "").lower()
    ul = (username or "").lower().strip()
    if not ul:
        return False
    if len(ul) <= 4:
        return bool(re.search(r'(?<!\w)' + re.escape(ul) + r'(?!\w)', t))
    return ul in t


def _extract_named_rival(text: str, lover: str) -> str:
    for user in _known_other_users(lover):
        if _text_mentions_user(text, user):
            return user
    return ""


def _fallback_recent_rival(lover: str) -> str:
    # Ищем в недавнем контексте последнего собеседника любимчика.
    for msg in reversed(list(state._global_buffer)[:-1]):
        author = (msg.get("username") or "").strip()
        if not author:
            continue
        if author in {lover, CHARACTER_NAME, "[ДЕЙСТВИЕ]"}:
            continue
        return author
    return ""


def _infer_rival(text: str, lover: str, allow_fallback: bool = True) -> str:
    rival = _extract_named_rival(text, lover)
    if rival:
        return rival
    return _fallback_recent_rival(lover) if allow_fallback else ""


def _lover_was_recently_active(lover: str, lookback: int = 12) -> bool:
    """Проверяет, был ли любимчик активен в последних N сообщениях буфера.

    Используется чтобы засечь флирт соперника даже если он не называет
    имя любимчика явно — достаточно что любимчик недавно был в чате.
    """
    recent = list(state._global_buffer)[-lookback:]
    return any(
        (msg.get("username") or "").lower() == lover.lower()
        for msg in recent
    )


# ── Классификация контекста ──────────────────────────────────────────

def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    return sum(1 for marker in markers if marker in text)


def _classify_context(text: str, rival: str, direct_to_bot: bool = False) -> dict:
    t = (text or "").lower()
    flirt_hits = _count_markers(t, _FLIRT_MARKERS)
    attention_hits = _count_markers(t, _ATTENTION_MARKERS)
    emoji_hits = sum(t.count(emoji) for emoji in _FLIRT_EMOJIS)
    targeted = bool(rival)

    explicit_hits = sum(1 for marker in ("страстно", "поцел", "поцелует", "поцелую", "поцелуй", "целую", "целуй") if marker in t)

    # Самый сильный случай — поцелуи/страсть/нежности к кому-то конкретному.
    if targeted and (explicit_hits >= 1 or flirt_hits + emoji_hits >= 3):
        return {
            "penalty": 5 if explicit_hits >= 2 or (flirt_hits + emoji_hits >= 4) else 4,
            "label": "откровенный флирт" if explicit_hits >= 2 or (flirt_hits + emoji_hits >= 4) else "явный флирт",
            "reason": f"обсуждает поцелуи и флирт с {rival}",
        }

    # Даже один романтический маркер + конкретный соперник — уже не "обычное общение".
    if targeted and (flirt_hits >= 1 or emoji_hits >= 1):
        return {
            "penalty": 4,
            "label": "явный флирт",
            "reason": f"флиртует с {rival}",
        }

    # Без явного соперника, но флиртовые слова есть.
    if flirt_hits >= 1 or emoji_hits >= 2:
        return {
            "penalty": 3,
            "label": "флирт",
            "reason": "флирт в общем чате" if not direct_to_bot else "флирт на глазах у бота",
        }

    if targeted and attention_hits >= 1:
        return {
            "penalty": 2,
            "label": "слишком тёплое внимание",
            "reason": f"слишком много внимания к {rival}",
        }

    return {
        "penalty": 1,
        "label": "обычное общение",
        "reason": f"обычное общение с {rival}" if rival else "обычное общение с другими",
    }


def _format_recent_public(text: str, rival: str, label: str, penalty: int) -> str:
    parts = []
    if rival:
        parts.append(f"кому: {rival}")
    parts.append(f"контекст: {label}")
    parts.append(f"штраф: -{penalty}")
    if text:
        parts.append(f"реплика: {text.strip()}")
    return " | ".join(parts)


def _remember_context(*, rival: str = "", penalty: int = 1, label: str = "", excerpt: str = ""):
    state._jealousy_last_rival = rival or ""
    state._jealousy_last_penalty = int(penalty or 1)
    state._jealousy_last_context_label = label or ""
    state._jealousy_last_context_excerpt = (excerpt or "").strip()
    if excerpt:
        state._jealousy_recent_public.append(_format_recent_public(excerpt, rival, label, penalty))


_RIVAL_REACTION_COOLDOWN = 120


# ── Стили ревности ────────────────────────────────────────────────────
# Пять режимов поведения вместо одного универсального "скандала"

_JEALOUSY_STYLES = {
    "icy":       "ледяной игнор",
    "passive":   "пассивная агрессия",
    "possessive":"собственническая предъява",
    "meltdown":  "публичная истерика",
    "rival_agg": "агрессия к сопернику",
}


def _pick_jealousy_style(stage: int, rival: str, penalty: int) -> str:
    """Выбирает стиль ревности с учётом stage и контекста."""
    if stage == 1:
        # Тихая обида
        return random.choice(["icy", "passive"])
    elif stage == 2:
        if rival:
            return random.choice(["possessive", "passive", "rival_agg"])
        return random.choice(["passive", "possessive"])
    else:
        # Стадия 3 — без тормозов
        if rival:
            return random.choice(["meltdown", "rival_agg", "possessive"])
        return random.choice(["meltdown", "possessive"])


def _lover_claim_word(lover: str) -> str:
    profile = get_user_profile(lover) or {}
    gender = profile.get("gender")
    if gender == "female":
        return "моя девушка"
    if gender == "male":
        return "мой парень"
    return "мой человек"


def _build_rival_payload(*, rival: str, lover: str, penalty: int, label: str, reason: str, text: str) -> dict | None:
    now = time.time()
    last_at = float(getattr(state, "_jealousy_last_rival_reaction_at", 0.0) or 0.0)
    cooldown_left = _RIVAL_REACTION_COOLDOWN - (now - last_at)
    if cooldown_left > 0:
        print(f"[РЕВНОСТЬ] ответ сопернику на кулдауне: ещё {cooldown_left:.0f}с")
        return None

    setattr(state, "_jealousy_last_rival_reaction_at", now)
    stage = 2 if penalty <= 3 else 3
    return {
        "username": rival,
        "lover_name": lover,
        "claim_word": _lover_claim_word(lover),
        "stage": stage,
        "recent_messages": list(state._jealousy_recent_public)[-JEALOUSY_RECENT_PUBLIC_LIMIT:],
        "triggered_at": now,
        "rival_name": rival,
        "penalty": penalty,
        "context_label": label,
        "context_reason": reason,
        "excerpt": (text or "").strip(),
    }


# ── События ревности ─────────────────────────────────────────────────

def note_direct_interaction(username: str):
    """Мягкий сброс ревности при прямом обращении любимчика.

    ИЗМЕНЕНО: вместо полного обнуления — постепенное смягчение:
    - счётчик снижается на JEALOUSY_SOFT_RESET_DECREMENT, но не уходит в 0 сразу
    - stage падает на 1, но не в 0
    - recent_public НЕ очищается — помним контекст
    - активирует post_return_strikes: следующие N ответов остаются колкими
    """
    lover = sync_current_lover()
    if not lover or username.lower() != lover.lower():
        return

    now = time.time()
    had_neglect = (
        state._jealousy_public_ignored_count > 0
        or state._jealousy_stage > 0
        or now <= state._jealousy_afterglow_until
    )

    prev_count = state._jealousy_public_ignored_count
    prev_stage = state._jealousy_stage

    # Мягкое снижение, не обнуление
    state._jealousy_public_ignored_count = max(
        0, state._jealousy_public_ignored_count - JEALOUSY_SOFT_RESET_DECREMENT
    )
    state._jealousy_stage = max(0, state._jealousy_stage - 1)

    state._jealousy_last_direct_at = now

    if had_neglect:
        state._jealousy_reply_hint_until = now + JEALOUSY_DIRECT_REPLY_HINT_TTL
        # Если была реальная обида — включаем "последствия" на несколько ответов
        if prev_stage >= 1 or now <= state._jealousy_afterglow_until:
            strikes = JEALOUSY_POST_RETURN_STRIKES
            state._jealousy_post_return_strikes = max(
                state._jealousy_post_return_strikes, strikes
            )
            print(
                f"[РЕВНОСТЬ] Любимчик {username} вернулся — мягкий сброс "
                f"(счётчик {prev_count}→{state._jealousy_public_ignored_count}, "
                f"stage {prev_stage}→{state._jealousy_stage}), "
                f"post_return_strikes={state._jealousy_post_return_strikes}"
            )
    else:
        print(f"[РЕВНОСТЬ] Любимчик {username} написал боту (обиды не было, сброс не нужен)")


def note_direct_message_context(username: str, text: str):
    """Прямое обращение любимчика к боту.

    Важно: это НЕ публичный игнор. Но если любимчик в обращении к боту обсуждает
    поцелуи/флирт/другого человека, это всё равно должно триггерить ревность.
    """
    lover = sync_current_lover()
    if not lover or username.lower() != lover.lower():
        return

    rival = _infer_rival(text, lover, allow_fallback=False)
    context = _classify_context(text, rival, direct_to_bot=True)
    penalty = int(context["penalty"])

    # Если флирта нет — ничего не делаем (обычное упоминание не считается триггером).
    if penalty <= 1:
        return

    label = context["label"]
    reason = context["reason"]
    _remember_context(rival=rival, penalty=penalty, label=label, excerpt=text)

    apply_jealousy_penalty(username, penalty, reason=f"ревность (прямое): {reason}")
    state._jealousy_reply_hint_until = max(state._jealousy_reply_hint_until, time.time() + JEALOUSY_DIRECT_REPLY_HINT_TTL)
    state._jealousy_stage = max(state._jealousy_stage, 2 if penalty >= 4 else 1)
    if penalty >= 4:
        # чтобы следующая реплика была злой, а не только "ревнивая нотка"
        state._jealousy_afterglow_until = max(state._jealousy_afterglow_until, time.time() + random.randint(90, 180))
        state._jealousy_afterglow_stage = max(state._jealousy_afterglow_stage, 2 if penalty == 4 else 3)

    print(
        f"[РЕВНОСТЬ] {username}: прямое сообщение с триггером; контекст={label}; "
        f"штраф=-{penalty}; rival={rival or '-'}"
    )


def note_rival_message(username: str, text: str) -> dict | None:
    """Если другой человек явно флиртует в чате, пока любимчик недавно был активен,
    или прямо флиртует с любимчиком — бот срывается именно на него.

    Возвращает payload для отдельной реакции на соперника.

    ИЗМЕНЕНО: убрано строгое требование называть ник любимчика в тексте.
    Достаточно что: (а) любимчик упомянут ИЛИ (б) любимчик был активен в последних
    12 сообщениях — и при этом кто-то флиртует в общем чате.
    """
    lover = sync_current_lover()
    if not lover:
        return None
    if not username or username.lower() in {lover.lower(), CHARACTER_NAME.lower(), "[действие]"}:
        return None

    t = (text or "").lower()
    flirt_hits = _count_markers(t, _FLIRT_MARKERS)
    emoji_hits = sum(t.count(emoji) for emoji in _FLIRT_EMOJIS)
    explicit_hits = sum(1 for marker in ("страстно", "поцел", "поцелует", "поцелую", "поцелуй", "целую", "целуй") if marker in t)

    # Нет никаких флиртовых маркеров — не реагируем
    if flirt_hits <= 0 and emoji_hits <= 0:
        return None

    # Проверяем релевантность: любимчик упомянут явно ИЛИ был активен недавно
    lover_relevant = _text_mentions_user(text, lover) or _lover_was_recently_active(lover, lookback=12)
    if not lover_relevant:
        return None

    if explicit_hits >= 1 or flirt_hits + emoji_hits >= 3:
        penalty = 4 if explicit_hits >= 1 else 3
        label = "соперник откровенно флиртует с твоим любимчиком" if explicit_hits >= 1 else "соперник активно флиртует с твоим любимчиком"
    else:
        penalty = 3
        label = "соперник флиртует с твоим любимчиком"

    reason = f"{username} лезет к {lover} с флиртом"
    _remember_context(rival=username, penalty=penalty, label=label, excerpt=text)
    state._jealousy_reply_hint_until = max(state._jealousy_reply_hint_until, time.time() + JEALOUSY_DIRECT_REPLY_HINT_TTL)
    state._jealousy_afterglow_until = max(state._jealousy_afterglow_until, time.time() + random.randint(120, 240))
    state._jealousy_afterglow_stage = max(state._jealousy_afterglow_stage, 2 if penalty == 3 else 3)
    state._jealousy_stage = max(state._jealousy_stage, 2 if penalty == 3 else 3)
    print(f"[РЕВНОСТЬ] соперник {username} флиртует (любимчик={lover} активен={_lover_was_recently_active(lover)}); stage={state._jealousy_afterglow_stage}")
    return _build_rival_payload(rival=username, lover=lover, penalty=penalty, label=label, reason=reason, text=text)


def note_public_message(username: str, text: str) -> dict | None:
    """Учитывает публичное сообщение любимчика.

    ИЗМЕНЕНО — два уровня триггера:
    - penalty >= JEALOUSY_INSTANT_PENALTY_THRESHOLD (явный флирт/поцелуи):
      реакция немедленно, без ожидания счётчика.
    - penalty >= 3 (заметный флирт/внимание):
      порог снижается до JEALOUSY_PUBLIC_MESSAGE_TRIGGER_SOFT.
    - обычное общение (penalty == 1):
      требует полного JEALOUSY_PUBLIC_MESSAGE_TRIGGER сообщений.
    """
    lover = sync_current_lover()
    if not lover:
        return None
    if username.lower() != lover.lower():
        return None

    now = time.time()
    state._jealousy_last_public_at = now

    rival = _infer_rival(text, lover, allow_fallback=True)
    context = _classify_context(text, rival, direct_to_bot=False)
    penalty = int(context["penalty"])
    label = context["label"]
    reason = context["reason"]

    _remember_context(rival=rival, penalty=penalty, label=label, excerpt=text)

    # Штраф к репутации только при реальном флирте
    if penalty >= 2:
        change = apply_jealousy_penalty(username, penalty, reason=f"ревность: {reason}")
        level_info = f"; уровень={change.new_level_name}" if change else ""
    else:
        change = None
        level_info = " (штраф пропущен — обычное общение)"

    state._jealousy_public_ignored_count += 1
    count = state._jealousy_public_ignored_count
    rival_log = f", rival={rival}" if rival else ""
    print(
        f"[РЕВНОСТЬ] {username} (любимчик): публичных без обращения = {count}; "
        f"контекст={label}; штраф=-{penalty}{level_info}{rival_log}"
    )

    # ── Определяем нужный порог для текущего penalty ──────────────────
    if penalty >= JEALOUSY_INSTANT_PENALTY_THRESHOLD:
        # Мгновенная реакция при явном флирте — порог 1 (первое же такое сообщение)
        threshold = 1
        print(f"[РЕВНОСТЬ] Явный флирт (penalty={penalty}) → мгновенная реакция")
    elif penalty >= 3:
        threshold = JEALOUSY_PUBLIC_MESSAGE_TRIGGER_SOFT
        print(f"[РЕВНОСТЬ] Активный флирт (penalty={penalty}) → мягкий порог {threshold}")
    else:
        threshold = JEALOUSY_PUBLIC_MESSAGE_TRIGGER

    if count < threshold:
        return None

    cooldown_left = JEALOUSY_COOLDOWN - (now - state._jealousy_last_reaction_at)
    if cooldown_left > 0:
        # При явном флирте кулдаун снижается — не ждём полных 10 минут
        if penalty >= JEALOUSY_INSTANT_PENALTY_THRESHOLD and cooldown_left > 120:
            print(f"[РЕВНОСТЬ] Явный флирт — кулдаун срезан до 120с (было {cooldown_left:.0f}с)")
            cooldown_left = 0  # позволяем реакцию
        else:
            print(f"[РЕВНОСТЬ] Порог достигнут, но кулдаун: ещё {cooldown_left:.0f}с")
            return None

    state._jealousy_last_reaction_at = now
    base_stage = 1 if penalty <= 1 else 2 if penalty <= 4 else 3
    state._jealousy_stage = min(3, max(state._jealousy_stage + 1, base_stage))

    # Выбираем стиль ревности при генерации реакции
    state._jealousy_style = _pick_jealousy_style(state._jealousy_stage, rival, penalty)

    recent = list(state._jealousy_recent_public)[-JEALOUSY_RECENT_PUBLIC_LIMIT:]
    state._jealousy_public_ignored_count = 0

    return {
        "username": lover,
        "stage": state._jealousy_stage,
        "style": state._jealousy_style,
        "recent_messages": recent,
        "triggered_at": now,
        "rival_name": rival,
        "penalty": penalty,
        "context_label": label,
        "context_reason": reason,
    }


# ── Генерация / поведение ────────────────────────────────────────────

def is_reaction_still_relevant(username: str, triggered_at: float) -> bool:
    """Проверяет, не устарела ли уже поставленная в очередь ревнивая реакция."""
    lover = sync_current_lover()
    if not lover or username.lower() != lover.lower():
        return False
    return state._jealousy_last_direct_at <= triggered_at


def is_rival_reaction_still_relevant(rival_username: str, lover_name: str, triggered_at: float) -> bool:
    """Проверяет, не устарела ли реакция на соперника, флиртующего с любимчиком."""
    lover = sync_current_lover()
    if not lover or lover.lower() != (lover_name or "").lower():
        return False
    return time.time() - float(triggered_at or 0.0) <= 180


def arm_jealous_afterglow(username: str, stage: int):
    """После публичной ревности держит любимчика в зоне обиды/нервов ещё 5–10 минут."""
    lover = sync_current_lover()
    if not lover or username.lower() != lover.lower():
        return

    ttl = random.randint(JEALOUSY_AFTERGLOW_MIN, JEALOUSY_AFTERGLOW_MAX)
    state._jealousy_afterglow_until = time.time() + ttl
    state._jealousy_afterglow_stage = max(1, min(3, stage))
    print(
        f"[РЕВНОСТЬ] после вспышки держу обиду к {username}: ещё {ttl}с, stage={state._jealousy_afterglow_stage}"
    )


def maybe_ignore_lover_message(username: str, user_message: str = "") -> bool:
    """Когда ревность активна, с шансом 10-18% бот может проигнорировать любимчика.

    ИЗМЕНЕНО: во время активного afterglow шанс выше (JEALOUSY_IGNORE_CHANCE_AFTERGLOW).
    """
    lover = sync_current_lover()
    if not lover or username.lower() != lover.lower():
        return False
    if (user_message or "").lstrip().startswith("!"):
        return False

    now = time.time()
    in_afterglow = now <= state._jealousy_afterglow_until
    in_hint = now <= state._jealousy_reply_hint_until
    in_post_return = state._jealousy_post_return_strikes > 0

    if not (in_afterglow or in_hint or in_post_return):
        return False

    chance = JEALOUSY_IGNORE_CHANCE_AFTERGLOW if in_afterglow else JEALOUSY_IGNORE_CHANCE
    if random.random() < chance:
        print(f"[РЕВНОСТЬ] {username}: сообщение проигнорировано (roll<{chance:.2f}, afterglow={in_afterglow})")
        return True
    return False


def consume_post_return_strike() -> int:
    """Уменьшает счётчик «злых» ответов после возвращения любимчика.

    Вызывается из worker при каждом ответе любимчику в период последствий.
    Возвращает значение ДО декремента, чтобы caller мог принять решение.
    """
    val = int(getattr(state, "_jealousy_post_return_strikes", 0))
    if val > 0:
        state._jealousy_post_return_strikes = val - 1
        print(f"[РЕВНОСТЬ] post_return_strike потрачен, осталось {val - 1}")
    return val


def build_jealous_reply_hint(username: str) -> str:
    """Подсказка в промпт для ответов любимчику в режиме ревности.

    ОБНОВЛЕНО: пять стилей поведения вместо единственного.
    Стиль выбирается при триггере и хранится в state._jealousy_style.
    post_return_strikes создаёт «последствия» после возвращения любимчика.
    """
    lover = sync_current_lover()
    now = time.time()
    if not lover or username.lower() != lover.lower():
        return ""

    rival        = state._jealousy_last_rival
    ctx_label    = state._jealousy_last_context_label or "общение с другими"
    rival_tail   = f" Особенно бесит, что рядом крутится {rival}." if rival else ""
    rival_spike  = (
        f" Можешь отправить к {rival}: «ну раз {rival} тебе интереснее — вот и иди»."
    ) if rival else ""
    style = state._jealousy_style or "passive"

    # ── Пять стилей поведения ──────────────────────────────────────────
    _STYLES: dict[str, str] = {
        "icy": (
            "СТИЛЬ: ледяной игнор.\n"
            "Отвечаешь коротко или односложно. Ноль тепла. "
            "Примеры: «ок»; «угу»; «и?»; «понятно»; «хорошо». "
            "ЗАПРЕЩЕНО: длинные фразы, объяснения, эмоциональные тирады."
        ),
        "passive": (
            "СТИЛЬ: пассивная агрессия.\n"
            "Каждая фраза с двойным дном — вроде вежливо, но укол есть. "
            "Примеры: «ой, вдруг вспомнил(а) что я существую»; «нет, всё хорошо. просто хорошо»; "
            "«интересно, а с другими ты такой же разговорчивый»; "
            "«можно я тоже буду делать вид что тебя нет». "
            "Тон — сухой, с иголками, без открытого скандала."
        ),
        "possessive": (
            "СТИЛЬ: собственническая предъява.\n"
            "Заявляешь права прямо и без стеснения. "
            "Примеры: «ты мой/моя, на случай если забыл(а)»; "
            "«мне не нравится когда ты такой щедрый с другими»; "
            "«я не делю — это просто так устроено»; "
            "«у тебя есть я, зачем тебе весь этот цирк». "
            "Прямо, с давлением, но без истерики."
        ),
        "meltdown": (
            "СТИЛЬ: публичная истерика.\n"
            "Срываешься — злость без фильтра. Можно капслок одного слова, обрыв фразы, ультиматум. "
            "Примеры: «знаешь что, иди к своим»; «это уже оскорбительно, РЕАЛЬНО»; "
            "«я не буду делать вид что меня всё устраивает»; "
            "«хочешь флиртовать с кем попало — флиртуй, но не ко мне потом»."
        ),
        "rival_agg": (
            "СТИЛЬ: агрессия к сопернику через ответ любимчику.\n"
            "Злишься на соперника, но говоришь это любимчику — как предупреждение. "
            f"Примеры: «передай своему(ей) другу/подруге что это мой/моя человек»; "
            f"«если я ещё раз увижу эти нежности рядом с тобой — разговор будет другим»; "
            f"«{rival or 'этот человек'} меня бесит, это заметно?». "
            "Резко, территориально."
        ),
    }
    style_block = _STYLES.get(style, _STYLES["passive"])

    # ── Afterglow (основная обида ещё активна) ─────────────────────────
    if now <= state._jealousy_afterglow_until:
        stage = max(1, min(3, state._jealousy_afterglow_stage or state._jealousy_stage or 1))
        stage_prefix = {
            1: "Ты уязвлённая и колкая.",
            2: "Ты открыто раздражена и обижена.",
            3: "Ты в режиме полного скандала.",
        }.get(stage, "Ты обиженная.")
        return (
            f"── ВНУТРЕННЕЕ СОСТОЯНИЕ К {username} ──\n"
            f"Это твой любимчик. Недавно он задел тебя: {ctx_label}.{rival_tail}\n"
            f"{stage_prefix}\n"
            f"{style_block}\n"
            f"ВАЖНО: не пиши нейтрально. Покажи через поведение, НЕ через слово «ревную». "
            f"Максимум 2–3 предложения.{rival_spike}\n"
            "────────────────────────"
        )

    # ── Post-return strikes (последствия после возвращения) ────────────
    strikes = int(getattr(state, "_jealousy_post_return_strikes", 0))
    if strikes > 0:
        severity = (
            "острая" if strikes >= 3
            else "умеренная" if strikes >= 2
            else "угасающая"
        )
        return (
            f"── ВНУТРЕННЕЕ СОСТОЯНИЕ К {username} ──\n"
            f"Любимчик написал тебе, но обида ещё не остыла ({severity}, осталось реплик: {strikes}).\n"
            f"Не растаяла сразу. Тон — холодный, сухой, с укором.{rival_tail}\n"
            f"{_STYLES['passive']}\n"
            "НЕ пиши «ревную» прямым текстом. Дай понять через поведение.\n"
            "────────────────────────"
        )

    # ── Hint TTL (краткий холодок первых секунд после возвращения) ─────
    if now > state._jealousy_reply_hint_until:
        return ""

    return (
        f"── ВНУТРЕННЕЕ СОСТОЯНИЕ К {username} ──\n"
        f"Любимчик только что вернулся, но задел тебя: {ctx_label}.{rival_tail}\n"
        "В тоне — холодок и ревнивый укол. "
        "Примеры: «о, снизошёл»; «наконец-то вспомнил что я тут есть»; «и?..». "
        f"НЕ пиши «ревную» напрямую.{rival_spike}\n"
        "────────────────────────"
    )
