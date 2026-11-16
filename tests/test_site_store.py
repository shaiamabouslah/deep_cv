from src.site_store import add_custom_site, get_custom_sites


def test_add_and_list_custom_site(tmp_path):
    db_path = str(tmp_path / "sites.db")
    add_custom_site(db_path, "Mon Lac", 36.1, 9.2)
    sites = get_custom_sites(db_path)
    assert len(sites) == 1
    assert sites[0]["name"] == "Mon Lac"
    assert sites[0]["custom"] is True


def test_duplicate_name_is_ignored(tmp_path):
    db_path = str(tmp_path / "sites.db")
    add_custom_site(db_path, "Mon Lac", 36.1, 9.2)
    add_custom_site(db_path, "Mon Lac", 40.0, 10.0)
    sites = get_custom_sites(db_path)
    assert len(sites) == 1
    assert sites[0]["lat"] == 36.1


def test_add_custom_site_reports_name_taken_with_existing_coords(tmp_path):
    db_path = str(tmp_path / "sites.db")
    first = add_custom_site(db_path, "Mon Lac", 36.1, 9.2)
    assert first["name_taken"] is False

    second = add_custom_site(db_path, "Mon Lac", 40.0, 10.0)
    assert second["name_taken"] is True
    assert second["lat"] == 36.1
    assert second["lon"] == 9.2
