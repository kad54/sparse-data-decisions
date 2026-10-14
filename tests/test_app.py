from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_complete_minimal_research_demo():
    app = AppTest.from_file(Path("app.py").resolve(), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "How little information is enough?"
    assert [tab.label for tab in app.tabs] == [
        "Simulation",
        "Research Experiments",
        "Possible Extensions",
        "Data & Methodology",
    ]
    app.slider[0].set_value(20).run()
    assert not app.exception
    assert app.session_state["poc_result"]["n"] == 3
    app.button[0].click().run()
    assert app.session_state["poc_result"]["n"] == 20
    assert not app.exception
    app.selectbox[0].set_value("High · 50 min").run()
    app.button[1].click().run()
    assert app.session_state["poc_result"]["noise_sd"] == 50
    assert app.session_state["poc_result"]["seed"] == 43
    app.button[2].click().run()
    assert not app.exception
    records, noise = app.session_state["poc_experiments"]
    assert len(records) == 2100
    assert noise == 50
    assert len(app.get("image")) == 3
