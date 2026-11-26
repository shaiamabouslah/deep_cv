"""Reads the official daily dam bulletins of the Tunisian Ministry of
Agriculture (Direction générale des barrages et des grands travaux
hydrauliques, published by ONAGRI as `BARRAGES-D-M-YYYY.pdf`) into
per-dam fill-rate points, the ground truth the satellite estimate is
calibrated and validated against.

A bulletin is an Excel export: one row per dam, with the day's stock, the
fill rate (% of the dam's current capacity) and the stock a year earlier as
the last three numbers of the row. Only those numbers are used. The PDF is
untrusted web content: it is only ever parsed as text (pypdf, no code from
the file is executed) and only numbers matching the expected row shape are
kept.

    python -m src.official_fill <folder with bulletins>
"""
import glob
import json
import os
import re
import sys
from datetime import datetime

# Name as printed in the bulletin table -> name used in src/sites.py.
BULLETIN_NAMES = {
    "MELLEGUE": "Mellègue", "SIDI SALEM": "Sidi Salem", "S. EL BARRAK": "Sidi El Barrak",
    "BIR M'CHERGA": "Bir M'cherga", "NEBHANA": "Nebhana", "LAKHMESS": "Lakhmes",
}
NUMBER = re.compile(r"-?\d+(?:,\d+)?")
OBSERVED = re.compile(r"observ\S*\s+le\s+(\d\d)-(\d\d)-(\d\d)")
POINTS_PATH = "data/official/fill_points.json"


def parse_bulletin(text):
    """Returns {"date": iso, "dams": {name: {fill_pct, stock_mm3, stock_year_ago_mm3}}}."""
    header = OBSERVED.search(text)
    if header is None:
        raise ValueError("bulletin date not found")
    day, month, year = (int(g) for g in header.groups())
    date = datetime(2000 + year, month, day).date().isoformat()

    dams = {}
    for line in text.splitlines():
        stripped = line.strip()
        for printed, name in BULLETIN_NAMES.items():
            if not stripped.upper().startswith(printed) or name in dams:
                continue
            numbers = [float(n.replace(",", ".")) for n in NUMBER.findall(stripped[len(printed):])]
            if len(numbers) < 14:
                continue
            fill, stock, last_year = numbers[-3:]
            if 0 <= fill <= 100 and stock >= 0:
                dams[name] = {"fill_pct": fill, "stock_mm3": stock, "stock_year_ago_mm3": last_year}
    return {"date": date, "dams": dams}


def read_pdf(path):
    from pypdf import PdfReader

    return "\n".join(
        page.extract_text(extraction_mode="layout") for page in PdfReader(path).pages[:1]
    )


def points_from_folder(folder):
    points = []
    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        try:
            bulletin = parse_bulletin(read_pdf(path))
        except Exception as exc:
            print("skipped", os.path.basename(path), str(exc)[:60], file=sys.stderr)
            continue
        for dam, row in bulletin["dams"].items():
            points.append({"dam": dam, "date": bulletin["date"], "official_fill_pct": row["fill_pct"],
                           "stock_mm3": row["stock_mm3"], "source": "ONAGRI bulletin " + os.path.basename(path)})
    return points


def save_points(points, path=POINTS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    unique = {(p["dam"], p["date"]): p for p in points}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(unique.values(), key=lambda p: (p["dam"], p["date"])), f, ensure_ascii=False, indent=1)
    return list(unique.values())


def load_points(path=POINTS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    collected = save_points(points_from_folder(sys.argv[1]))
    print(f"{len(collected)} official points from {len({p['date'] for p in collected})} bulletins")
