build:
	npm build rasalit/apps/demo/frontend

develop:
	python -m pip install -e ".[dev]"
	pre-commit install

flake:
	flake8 rasalit tests setup.py

black:
	black --check .

test:
	pytest tests

clean:
	rm -rf build
	rm -rf dist

check: black flake test clean
