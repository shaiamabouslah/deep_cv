import pytest

from src import official_fill as of

BULLETIN = """
 MINISTERE DE L'AGRICULTURE      APPORTS ET LACHERS DES BARRAGES
 DIRECTION GENERALE DES BARRAGES        du    16-07-25    observée le  17-07-25
 MELLEGUE      0,005  7,017  4,561  130,639  14,962  169,690   p   0,020  0,020  0,320  112,649  12,775   59,5   30,404  12,304   قلاـــــم
 SIDI SALEM    0,000  0,482  4,534  121,795  124,136 396,651   T,p 0,841  0,841  13,851 120,058 135,407  32,7  189,851 153,383   ملاــــس يديس
 S. EL BARRAK  0,000  1,099  0,495  210,456 133,452 296,291   E,tr 0,737 0,420  11,774  162,540 121,658  53,4  152,969 116,172   قارــــبلا يديس
 BIR M'CHERGA  0,000  0,023  0,222  8,365   6,974   26,099    E   0,007  0,004  0,109   0,853   1,311    19,5    8,131   8,191   ةـــقراشملا رئب
 NEBHANA       0,007  0,174  0,347  6,480   1,834   22,505    E   0,041  0,000  0,631   2,301   1,281     6,5    4,218   2,192   ةــــــناهبن
 LAKHMESS      0,001  0,062  0,395  1,606   0,799   12,608    E   0,000  0,000  0,001   0,019   1,668    29,6    2,137   1,207   ساـــــــمخل
 Kebir         0,000  0,097  0,110  30,373  20,331  39,677        0,000  0,000  0,000  11,503  0,000    71,2   45,846  28,220   رــيبكلا
 Sidi el Barrak (transfert) 0,708 141,871 161,790
"""


def test_parses_the_six_dams_with_the_observed_date():
    parsed = of.parse_bulletin(BULLETIN)

    assert parsed["date"] == "2025-07-17"
    assert {n: d["fill_pct"] for n, d in parsed["dams"].items()} == {
        "Mellègue": 59.5, "Sidi Salem": 32.7, "Sidi El Barrak": 53.4,
        "Bir M'cherga": 19.5, "Nebhana": 6.5, "Lakhmes": 29.6,
    }
    assert parsed["dams"]["Sidi Salem"]["stock_mm3"] == 189.851
    assert parsed["dams"]["Mellègue"]["stock_year_ago_mm3"] == 12.304


def test_ignores_other_dams_and_the_short_transfer_line():
    parsed = of.parse_bulletin(BULLETIN)
    assert "Kebir" not in parsed["dams"] and len(parsed["dams"]) == 6


def test_rejects_a_bulletin_without_a_date():
    with pytest.raises(ValueError):
        of.parse_bulletin("SIDI SALEM 1,0 2,0")


def test_rows_with_an_impossible_fill_rate_are_dropped():
    broken = BULLETIN.replace("32,7", "320,7")
    assert "Sidi Salem" not in of.parse_bulletin(broken)["dams"]


def test_points_are_saved_deduplicated_and_reloadable(tmp_path):
    point = {"dam": "Nebhana", "date": "2025-07-17", "official_fill_pct": 6.5, "stock_mm3": 4.218, "source": "x"}
    path = str(tmp_path / "points.json")
    saved = of.save_points([point, dict(point), {**point, "date": "2025-07-18"}], path)
    assert len(saved) == 2
    assert of.load_points(path)[0]["date"] == "2025-07-17"
