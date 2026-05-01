"""
achievements.py — каталог достижений, проверка и форматирование.
"""

from dataclasses import dataclass

from database import (
    add_user_achievement,
    get_direct_user_message_count,
    get_global_user_message_count,
    get_user_achievement_codes,
    get_user_fact_count,
    get_user_trait_count,
)
from reputation import LEVELS, get_reputation


@dataclass(frozen=True)
class Achievement:
    code: str
    title: str
    description: str
    target: int | None = None


ACHIEVEMENTS: list[Achievement] = [
    Achievement("first_step", "первый шум", "оставил первое сообщение на планете", 1),
    Achievement("regular", "свой в чате", "написал 25 сообщений в общий чат", 25),
    Achievement("legend", "ветеран шума", "набил 100 сообщений в общий чат", 100),
    Achievement("noticed_bot", "замечен Анькой", "5 раз обратился к боту напрямую", 5),
    Achievement("memory_trace", "след в памяти", "о тебе накопилось хотя бы 3 факта и ярлыка", 3),
    Achievement("friendly", "свой человек", f"дошёл до уровня «{LEVELS[5]}» или выше"),
    Achievement("attached", "почти родной", f"дошёл до уровня «{LEVELS[4]}» или выше"),
    Achievement("lover", "любимчик Аньки", "стал единственным любимчиком"),
]


def _build_stats(username: str) -> dict:
    rep = get_reputation(username)
    return {
        "global_messages": get_global_user_message_count(username),
        "direct_messages": get_direct_user_message_count(username),
        "memory_items": get_user_fact_count(username) + get_user_trait_count(username),
        "level": rep["level"],
        "is_lover": rep["is_lover"],
    }


def _is_unlocked(code: str, stats: dict) -> bool:
    if code == "first_step":
        return stats["global_messages"] >= 1
    if code == "regular":
        return stats["global_messages"] >= 25
    if code == "legend":
        return stats["global_messages"] >= 100
    if code == "noticed_bot":
        return stats["direct_messages"] >= 5
    if code == "memory_trace":
        return stats["memory_items"] >= 3
    if code == "friendly":
        return stats["level"] <= 5
    if code == "attached":
        return stats["level"] <= 4
    if code == "lover":
        return bool(stats["is_lover"])
    return False


def _progress_line(code: str, stats: dict) -> str:
    if code == "first_step":
        return f"{min(stats['global_messages'], 1)}/1 сообщений"
    if code == "regular":
        return f"{min(stats['global_messages'], 25)}/25 сообщений"
    if code == "legend":
        return f"{min(stats['global_messages'], 100)}/100 сообщений"
    if code == "noticed_bot":
        return f"{min(stats['direct_messages'], 5)}/5 обращений"
    if code == "memory_trace":
        return f"{min(stats['memory_items'], 3)}/3 следов в памяти"
    if code == "friendly":
        return f"сейчас: {LEVELS[stats['level']]}"
    if code == "attached":
        return f"сейчас: {LEVELS[stats['level']]}"
    if code == "lover":
        return "ещё не любимчик" if not stats["is_lover"] else "получено"
    return ""


def check_new_achievements(username: str) -> list[Achievement]:
    """Проверяет новые достижения.

    При самом первом чеке уже выполненные условия тихо синхронизируются в БД
    без объявления в общий чат. Дальше возвращаются только реально новые ачивки.
    """
    earned_codes = set(get_user_achievement_codes(username))
    stats = _build_stats(username)

    if not earned_codes:
        for achievement in ACHIEVEMENTS:
            if _is_unlocked(achievement.code, stats):
                add_user_achievement(username, achievement.code)
        return []

    unlocked: list[Achievement] = []
    for achievement in ACHIEVEMENTS:
        if achievement.code in earned_codes:
            continue
        if _is_unlocked(achievement.code, stats) and add_user_achievement(username, achievement.code):
            unlocked.append(achievement)
    return unlocked


def build_achievement_notification(username: str, unlocked: list[Achievement]) -> str:
    if not unlocked:
        return ""
    if len(unlocked) == 1:
        achievement = unlocked[0]
        return f"{username}, ачивка открыта: «{achievement.title}» — {achievement.description}."
    names = ", ".join(f"«{achievement.title}»" for achievement in unlocked)
    return f"{username}, у тебя сразу несколько ачивок: {names}. Нормально ты разогнался."


def format_achievements(username: str) -> str:
    earned_codes = set(get_user_achievement_codes(username))
    stats = _build_stats(username)

    earned_lines: list[str] = []
    locked_lines: list[str] = []
    for achievement in ACHIEVEMENTS:
        line = f"• {achievement.title} — {achievement.description}"
        if achievement.code in earned_codes:
            earned_lines.append(line)
        else:
            locked_lines.append(f"• {achievement.title} — {_progress_line(achievement.code, stats)}")

    parts = [f"{username}, твои достижения:"]
    parts.append("Открыто:\n" + ("\n".join(earned_lines) if earned_lines else "• пока пусто"))

    if locked_lines:
        parts.append("Ещё можно выбить:\n" + "\n".join(locked_lines[:5]))

    return "\n\n".join(parts)
