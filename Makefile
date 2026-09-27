.PHONY: help venv install data test ask ui eval clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

help:
	@echo "Targets:"
	@echo "  venv     Create a Python 3.11 virtual environment in .venv"
	@echo "  install  Install the package and dependencies (editable)"
	@echo "  data     Build the bundled sample database"
	@echo "  test     Run the test suite"
	@echo "  ask      Ask a question, e.g. make ask Q='Which port had the most calls?'"
	@echo "  ui       Launch the Streamlit chat interface"
	@echo "  eval     Run the evaluation harness against the configured model"
	@echo "  clean    Remove caches and the generated database"

venv:
	python3.11 -m venv .venv

install:
	$(PIP) install -e ".[ui,dev]"

data:
	$(PY) -m text2sql.sample_data

test:
	$(PY) -m pytest -q

ask:
	$(PY) -m text2sql.cli "$(Q)"

ui:
	.venv/bin/streamlit run app/streamlit_app.py

eval:
	$(PY) eval/harness.py

clean:
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info
	rm -f data/sample.duckdb
