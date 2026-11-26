"""Downloads the official daily dam bulletins (PDF) of the Tunisian Ministry
of Agriculture from onagri.nat.tn into a folder, for src/official_fill.py to
read. One bulletin every `step_days`; if a date is missing, its neighbours
(+1, -1 day) are tried, because a few days have no file.

File names are built from dates only (never from server responses), each
download is checked to be a real PDF of sane size, and requests are
sequential with a short pause. The PDFs are untrusted web content: keep them
outside the project and only parse them as text.

    python -m src.fetch_bulletins <quarantine folder> [start end step_days]
"""
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

BASE = "https://www.onagri.nat.tn//uploads/barrages/BARRAGES-{d}-{m}-{y}.pdf"
MAX_BYTES = 5_000_000
PAUSE_S = 0.4


def bulletin_url(day):
    return BASE.format(d=day.day, m=day.month, y=day.year)


def is_pdf(content):
    return content[:5] == b"%PDF-" and len(content) < MAX_BYTES


def fetch(day, timeout=40):
    request = urllib.request.Request(bulletin_url(day), headers={"User-Agent": "Mozilla/5.0 (research; dam fill validation)"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    return content if is_pdf(content) else None


def download(folder, start, end, step_days=7):
    os.makedirs(folder, exist_ok=True)
    got, missing = [], []
    day = start
    while day <= end:
        for offset in (0, 1, -1):
            candidate = day + timedelta(days=offset)
            name = f"BARRAGES-{candidate.isoformat()}.pdf"
            path = os.path.join(folder, name)
            if os.path.exists(path):
                got.append(candidate)
                break
            content = fetch(candidate)
            time.sleep(PAUSE_S)
            if content is not None:
                with open(path, "wb") as f:
                    f.write(content)
                got.append(candidate)
                break
        else:
            missing.append(day)
        day += timedelta(days=step_days)
    return got, missing


if __name__ == "__main__":
    folder = sys.argv[1]
    start = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2024, 1, 3)
    end = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2025, 10, 8)
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    got, missing = download(folder, start, end, step)
    print(f"{len(got)} bulletins downloaded, {len(missing)} dates unavailable")
    if got:
        print("first:", min(got), "last:", max(got))
    if missing:
        print("unavailable:", [d.isoformat() for d in missing][:12])
