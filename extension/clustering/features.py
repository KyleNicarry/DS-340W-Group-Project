"""Conservative, auditable contract attributes; unknowns remain explicit."""

from __future__ import annotations

import re

NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def extract_attributes(row: dict) -> dict:
    """Use observable wording and series codes, never prices or outcomes.

    Series codes are heuristic identifiers, not authoritative categories. Exact
    event IDs track dependence; they are not used as semantic clustering inputs.
    """
    title = row.get("title", "") or ""
    subtitle = row.get("yes_sub_title", "") or ""
    text = f"{title} {subtitle}".strip()
    lower = text.lower()
    series = (row.get("event_ticker", "") or "").split("-")[0].removeprefix("KX")
    domain, underlying, provenance = "unknown", series or "unknown", "series_fallback"
    for keyword, name in [("bitcoin", "BTC"), ("ethereum", "ETH"), ("dogecoin", "DOGE")]:
        if keyword in lower:
            domain, underlying, provenance = "crypto", name, "title_rule"
            break
    if "temp" in lower and ("high" in lower or "low" in lower):
        domain, provenance = "weather", "title_rule"
        # Preserve station/measurement identity rather than pooling different cities.
        underlying = series
    elif "s&p 500" in lower or "nasdaq" in lower:
        domain, underlying, provenance = "equities", "SP500" if "s&p 500" in lower else "NASDAQ100", "title_rule"
    elif re.match(r"(?:NFL|NBA|NCAA|MLB|NHL|WNBA|ATP|WTA|PGA)", series):
        domain, provenance = "sports", "series_rule"
        underlying = re.match(r"NCAAMB|NCAAF|WNBA|NFL|NBA|MLB|NHL|ATP|WTA|PGA", series)
        underlying = underlying.group() if underlying else series
    elif "MENTION" in series:
        domain, provenance = "mentions", "series_rule"

    # Prefer the affirmative subtitle: it often contains the actual proposition.
    proposition = (subtitle or title).lower()
    if "wins by" in lower:
        structure = "spread"
    elif "total points" in lower or "points scored" in lower:
        structure = "total"
    elif re.search(rf"{NUMBER}\s*°?\s*(?:to|–|—)\s*\$?{NUMBER}", proposition):
        structure = "interval"
    elif re.search(r"\b(?:above|over|exceed|greater|at least)\b|>|\d\+", proposition):
        structure = "above"
    elif re.search(r"\b(?:below|under|less|at most)\b|<", proposition):
        structure = "below"
    elif "winner" in lower or re.search(r"\bwin\b", lower):
        structure = "winner"
    else:
        structure = "unknown"
    # Do not pool distinct player-prop mechanisms just because they share a sport.
    if structure == "unknown" or (domain == "sports" and structure in {"above", "below"}):
        underlying = series or "unknown"
    numbers = re.findall(NUMBER, subtitle)
    template = re.sub(NUMBER, "[number]", text.lower())
    return {
        "series": series,
        "domain": domain,
        "underlying": underlying,
        "structure": structure,
        "attribute_source": provenance,
        "threshold_tokens": "|".join(numbers),
        "units": "degrees" if "°" in text or "temp" in lower else "USD" if "$" in text else "unknown",
        "family": f"{domain}/{underlying}/{structure}",
        "text": text,
        "template": template,
        "settlement_source": "unknown",
    }
