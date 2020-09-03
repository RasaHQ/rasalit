build:
	npm build rasalit/apps/demo/frontend

develop:
	python -m pip install -e ".[dev]"
	pre-commit install
