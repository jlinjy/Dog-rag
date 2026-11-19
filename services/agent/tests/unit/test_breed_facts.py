from app.tools.breed_facts import breed_database, load_breeds_df, normalize_quotes, run_sql


def test_csv_loads_with_numeric_ranges_and_no_duplicates():
    df = load_breeds_df()
    assert df["name"].is_unique
    beagle = df[df["name"] == "Beagle"].iloc[0]
    assert (beagle["lifespan_min_yrs"], beagle["lifespan_max_yrs"]) == (12, 15)


def test_select_returns_header_and_rows():
    out = run_sql("SELECT name, country FROM breeds WHERE lower(country) LIKE '%germany%'")
    lines = out.splitlines()
    assert lines[0] == "name | country"
    assert "German Shepherd | Germany" in lines


def test_tool_invocation_ranks_numerically():
    out = breed_database.invoke({"sql": "SELECT name FROM breeds ORDER BY height_max_in DESC LIMIT 1"})
    assert out.splitlines()[1] == "Irish Wolfhound"


def test_double_quoted_values_become_string_literals():
    assert normalize_quotes('SELECT "name" FROM breeds WHERE country = "Germany"') == \
        "SELECT \"name\" FROM breeds WHERE country = 'Germany'"
    assert normalize_quotes("WHERE name IN (“pug”)") == "WHERE name IN ('pug')"
    assert "21" in run_sql('SELECT COUNT(*) FROM breeds WHERE lower(country) LIKE "%england%"')


def test_writes_are_rejected():
    assert run_sql("DELETE FROM breeds").startswith("Error")
    out = run_sql("SELECT 1; DROP TABLE breeds")
    assert out.startswith("SQL error")
    assert "Beagle" in run_sql("SELECT name FROM breeds WHERE name = 'Beagle'")


def test_bad_sql_and_no_rows_return_guidance_not_exceptions():
    assert run_sql("SELECT nope FROM breeds").startswith("SQL error: no such column")
    assert run_sql("SELECT name FROM breeds WHERE name = 'Unicorn'").startswith("No rows matched")


def test_result_is_truncated(monkeypatch):
    from app import config
    monkeypatch.setattr(config, "SQL_MAX_ROWS", 3)
    out = run_sql("SELECT name FROM breeds")
    assert len(out.splitlines()) == 1 + 3 + 1
    assert "truncated" in out
