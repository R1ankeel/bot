"""
browser.py — всё что связано с браузером:
отправка сообщений, парсинг DOM, смайлы, определение новых сообщений.
"""

import asyncio
import random
import re

from config import CHARACTER_NAME, CHUNK_SIZE, CHUNK_DELAY

# ── Таблица смайлов платформы ─────────────────────────────────────────

SMILE_LABELS: dict[str, str] = {
    "1": "🙂", "2": "😌", "4": "😉", "5": "😁", "6": "😛", "8": "😳",
    "9": "🙁", "10": "😮", "11": "😐", "12": "недовольный",
    "13": "хмыкает / скепсис", "14": "😱", "15": "😢", "16": ">_<",
    "17": "кричит", "18": "❤️", "19": "⛔", "20": "🍺", "21": "😈",
    "22": "🥰", "23": "👄", "24": "💋", "32": "💤", "41": "🤢",
    "42": "😎", "43": "сомнение / неловко", "46": "дурацкая улыбка",
    "50": "🌹", "51": "👍", "52": "👎", "53": "😇", "54": "🥳",
    "55": "озадачен", "56": "с соской", "67": "💔", "68": "🤦",
    "69": "🤣", "70": "😨", "71": "троллфейс", "72": "жалобный / умоляет",
    "73": "серьёзный с сигаретой", "74": "😺", "75": "кровь из глаз",
    "76": "ест", "77": "😊", "78": "😡", "79": "😂", "80": "😝",
    "81": "недовольство / неприятно", "82": "💜", "83": "глупо и смешно",
    "84": "😭", "85": "😟", "86": "😏", "87": "😍", "88": "😘",
    "89": "скепсис / недовольный", "90": "💩", "91": "😑",
    "92": "гламур / модница", "93": "🙏", "94": "🌞", "95": "🚀",
    "96": "💎", "97": "🔥", "98": "закрыл глаза руками", "99": "за решёткой",
    "100": "💪", "101": "жарко", "102": "🤯", "103": "😱", "104": "👌",
    "105": "➕", "106": "уныло / безнадёжно",
    "107": "бьётся головой об стену", "108": "сбит с толку", "109": "🤔",
    "110": "😬", "111": "огненный шар", "112": "цветок", "113": "🤡",
    "114": "💯", "115": "⚡", "116": "🤩", "117": "😔", "118": "🌽",
    "119": "🎉", "120": "✨", "121": "👻", "122": "🎄", "123": "🎃",
    "124": "🥲", "125": "☠️", "126": "🎁", "127": "💐", "128": "⭐",
    "129": "❄️", "130": "😊", "131": "🤣", "132": "🤞",
}

TEXT_SMILE_ALIASES: dict[str, str] = {
    "]:->": "😈",
    "( ]:->": "😈",
    "]:-&gt;": "😈",
}


def _format_smile_token(label: str) -> str:
    label = (label or "").strip()
    if not label:
        return ""
    if re.search(r"[а-яёa-z]", label.lower()):
        return f"[смайл: {label}]"
    return label


def _replace_text_smile_aliases(text: str) -> str:
    for raw, label in TEXT_SMILE_ALIASES.items():
        text = text.replace(raw, label)
    return text


def _normalize_key_part(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


async def _extract_message_time(msg) -> str:
    time_el = await msg.query_selector("div.channel-message__time")
    if not time_el:
        return ""
    try:
        raw = (await time_el.inner_text()).strip()
    except Exception:
        return ""
    match = re.search(r"\b\d{1,2}:\d{2}\b", raw)
    return match.group(0) if match else ""


async def _extract_message_text(text_el) -> str:
    smile_map = {sid: _format_smile_token(label) for sid, label in SMILE_LABELS.items()}
    text = ""

    try:
        html = await text_el.inner_html()
    except Exception:
        html = ""

    if html:
        def _smile_repl(match: re.Match) -> str:
            sid = match.group(1) or ""
            return f" {smile_map.get(sid, f'[смайл:{sid or '?'}]')} "

        text = html
        text = re.sub(
            r'<div[^>]*class="[^"]*smile-big[^"]*"[^>]*data-smile-id="(\d+)"[^>]*>\s*</div>',
            _smile_repl,
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>\s*<div[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>\s*<p[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<channel\|([^>]*)>', r'\1', text)
        text = re.sub(r'<[^>]+>', '', text)

    if not text:
        try:
            text = await text_el.inner_text()
        except Exception:
            text = ""

    text = _replace_text_smile_aliases(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_message_base_key(username: str, text: str, visible_time: str) -> str:
    return "|".join([
        _normalize_key_part(username),
        _normalize_key_part(visible_time),
        _normalize_key_part(text),
    ])


# ── Отправка ──────────────────────────────────────────────────────────

def split_message(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= chunk_size:
            parts.append(text)
            break
        cut = text.rfind(" ", 0, chunk_size)
        if cut == -1:
            cut = chunk_size
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    return parts


async def send_text(page, text: str):
    input_field = await page.query_selector('input.channel-new-message__text-field__input')
    if input_field:
        await input_field.fill(text)
        await page.keyboard.press("Enter")


async def send_response(page, text: str):
    parts = split_message(text)
    for i, part in enumerate(parts):
        await send_text(page, part)
        label = f"[часть {i+1}/{len(parts)}] " if len(parts) > 1 else ""
        print(f"  {label}{part[:80]}")
        if i < len(parts) - 1:
            delay = random.uniform(*CHUNK_DELAY)
            await asyncio.sleep(delay)


# ── Парсинг DOM ───────────────────────────────────────────────────────

async def parse_messages(page) -> list[dict[str, str]]:
    messages = await page.query_selector_all('div.channel-message__content')
    result: list[dict[str, str]] = []
    duplicate_counters: dict[str, int] = {}

    for msg in messages:
        try:
            title_el = await msg.query_selector("span.channel-message__content__title")
            text_el = await msg.query_selector("div.channel-message__content__text")
            if not text_el:
                continue

            visible_time = await _extract_message_time(msg)

            if title_el:
                username = await title_el.evaluate("""el => {
                    return Array.from(el.childNodes)
                        .filter(n => n.nodeType === Node.TEXT_NODE)
                        .map(n => n.textContent.trim())
                        .filter(s => s.length > 0)
                        .join('');
                }""")
                username = re.sub(r'<[^>]+>', '', username).strip()
                text = await _extract_message_text(text_el)
                if username and text:
                    base_key = _build_message_base_key(username, text, visible_time)
                    duplicate_counters[base_key] = duplicate_counters.get(base_key, 0) + 1
                    result.append({
                        "username": username,
                        "text": text,
                        "time": visible_time,
                        "key": f"{base_key}#{duplicate_counters[base_key]}",
                    })
            else:
                # Действие без ника: "^Ник пожал руку НейроАнька"
                text = await _extract_message_text(text_el)
                lines = [
                    l.strip() for l in text.split("\n")
                    if l.strip() and not re.match(r'^\d{1,2}:\d{2}$', l.strip())
                ]
                if lines:
                    action_text = lines[0]
                    username = "[ДЕЙСТВИЕ]"
                    base_key = _build_message_base_key(username, action_text, visible_time)
                    duplicate_counters[base_key] = duplicate_counters.get(base_key, 0) + 1
                    result.append({
                        "username": username,
                        "text": action_text,
                        "time": visible_time,
                        "key": f"{base_key}#{duplicate_counters[base_key]}",
                    })
        except Exception as e:
            print(f"[PARSE_ERROR] {type(e).__name__}: {e}")
            continue

    return result


def find_new_messages(
    all_msgs: list[dict[str, str]],
    last_seen_index: int | None,
    last_seen_key: str | None,
) -> tuple[list[dict[str, str]], int]:
    if not all_msgs:
        return [], last_seen_index if last_seen_index is not None else -1
    if last_seen_index is None or not last_seen_key:
        return [], len(all_msgs) - 1
    if last_seen_index < len(all_msgs) and all_msgs[last_seen_index]["key"] == last_seen_key:
        return all_msgs[last_seen_index + 1:], len(all_msgs) - 1
    for i in range(len(all_msgs) - 1, -1, -1):
        if all_msgs[i]["key"] == last_seen_key:
            return all_msgs[i + 1:], len(all_msgs) - 1

    print("[DOM] Потерян якорь последних сообщений, включаю дедуп через cache")
    return all_msgs, len(all_msgs) - 1
