"""Extract explicit English date expressions without searching arbitrary numbers."""

import calendar
import re
from datetime import date, timedelta

MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})
WEEKDAYS = {name.lower(): index for index, name in enumerate(calendar.day_name)}
MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_RE = re.compile(
    rf"\b(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})\b"
    rf"|\b(?P<slash>\d{{1,2}}/\d{{1,2}}(?:/\d{{4}})?)\b"
    rf"|\b(?P<month>{MONTH_PATTERN})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:,?\s+(?P<year>\d{{4}}))?\b"
    rf"|\b(?P<relative>today|tomorrow|tonight)\b"
    rf"|\b(?:in\s+)(?P<count>\d+)\s+(?P<unit>days?|weeks?)\b"
    rf"|\b(?P<qualifier>this\s+|next\s+)?(?P<weekday>{'|'.join(WEEKDAYS)})\b"
    rf"|\b(?:on|by|due|before|until)\s+(?:the\s+)?(?P<ordinal>\d{{1,2}})(?:st|nd|rd|th)\b",
    re.I,
)
DEADLINE_RE = re.compile(r"\b(?:deadline|due(?!\s+to\b)|asap|eod|cob|no later than)\b", re.I)
TIME_RE = re.compile(r"\b(?:[0-2]?\d:[0-5]\d|(?:1[0-2]|0?[1-9])\s*(?:am|pm)|noon|midnight)\b", re.I)
VAGUE_TIME_RE = re.compile(
    r"\b(?:(?:this|next)\s+(?:week|month|year|morning|afternoon|evening)"
    r"|(?:by|before|until|at|after)\s+(?:breakfast|lunch|dinner)"
    r"|in\s+\d+\s+(?:hours?|minutes?|months?|years?))\b", re.I,
)


def next_month_day(month, day, anchor):
    for year in range(anchor.year, anchor.year + 9):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= anchor:
            return candidate
    raise ValueError("Invalid month/day")


def resolve_match(match, anchor):
    if match["iso"]:
        return date.fromisoformat(match["iso"])
    if match["slash"]:
        parts = list(map(int, match["slash"].split("/")))
        day, month = parts[:2]
        if day <= 12 and month <= 12 and day != month:
            raise ValueError("Ambiguous numeric date; use YYYY-MM-DD")
        if len(parts) == 3:
            return date(parts[2], month, day)
        return next_month_day(month, day, anchor)
    if match["month"]:
        month, day = MONTHS[match["month"].lower()], int(match["day"])
        if match["year"]:
            return date(int(match["year"]), month, day)
        return next_month_day(month, day, anchor)
    if match["relative"]:
        return anchor + timedelta(days=int(match["relative"].lower() == "tomorrow"))
    if match["count"]:
        days = int(match["count"]) * (7 if match["unit"].lower().startswith("week") else 1)
        return anchor + timedelta(days=days)
    if match["weekday"]:
        weekday = WEEKDAYS[match["weekday"].lower()]
        qualifier = (match["qualifier"] or "").strip().lower()
        if qualifier == "this":
            return anchor + timedelta(days=weekday - anchor.weekday())
        if qualifier == "next":
            return anchor + timedelta(days=7 - anchor.weekday() + weekday)
        return anchor + timedelta(days=(weekday - anchor.weekday()) % 7 or 7)
    day = int(match["ordinal"])
    if not 1 <= day <= 31:
        raise ValueError("Invalid day of month")
    for offset in range(13):
        month = (anchor.month - 1 + offset) % 12 + 1
        year = anchor.year + (anchor.month - 1 + offset) // 12
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= anchor:
            return candidate
    raise ValueError("Invalid day of month")


def extract_deadline(text, anchor=None):
    anchor = anchor or date.today()
    matches = list(DATE_RE.finditer(text))
    result = {"has_deadline_signal": bool(matches or DEADLINE_RE.search(text)
                                          or TIME_RE.search(text) or VAGUE_TIME_RE.search(text)),
              "parsed_date": None, "deadline_reason": None}
    if len(matches) > 1:
        result["deadline_reason"] = "Multiple dates; choose the intended deadline"
    elif matches:
        try:
            result["parsed_date"] = resolve_match(matches[0], anchor).isoformat()
            result["deadline_reason"] = f"Date expression: {matches[0].group().strip()}"
        except (ValueError, OverflowError) as error:
            result["deadline_reason"] = str(error)
    elif result["has_deadline_signal"]:
        result["deadline_reason"] = "Deadline or time mentioned without a supported calendar date"
    return result
