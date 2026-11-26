from datetime import date

from src import fetch_bulletins as fb


def test_url_is_built_from_the_date_only_without_zero_padding():
    assert fb.bulletin_url(date(2025, 7, 8)).endswith("/BARRAGES-8-7-2025.pdf")


def test_only_real_pdfs_of_sane_size_are_accepted():
    assert fb.is_pdf(b"%PDF-1.5 ...")
    assert not fb.is_pdf(b"<html>not found</html>")
    assert not fb.is_pdf(b"%PDF-1.5" + b"0" * fb.MAX_BYTES)


def test_missing_dates_fall_back_to_a_neighbouring_day_then_are_reported(tmp_path, monkeypatch):
    available = {date(2025, 1, 9): b"%PDF-1.5 a", date(2025, 1, 23): b"%PDF-1.5 b"}
    monkeypatch.setattr(fb, "fetch", lambda day: available.get(day))
    monkeypatch.setattr(fb, "PAUSE_S", 0)

    got, missing = fb.download(str(tmp_path), date(2025, 1, 8), date(2025, 1, 22), step_days=7)

    assert got == [date(2025, 1, 9), date(2025, 1, 23)]
    assert missing == [date(2025, 1, 15)]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["BARRAGES-2025-01-09.pdf", "BARRAGES-2025-01-23.pdf"]


def test_existing_files_are_not_downloaded_again(tmp_path, monkeypatch):
    (tmp_path / "BARRAGES-2025-01-08.pdf").write_bytes(b"%PDF-1.5 old")
    calls = []
    monkeypatch.setattr(fb, "fetch", lambda day: calls.append(day))
    monkeypatch.setattr(fb, "PAUSE_S", 0)

    got, _ = fb.download(str(tmp_path), date(2025, 1, 8), date(2025, 1, 8))

    assert got == [date(2025, 1, 8)] and calls == []
