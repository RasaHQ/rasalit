cr_url = ghcr.io
cr_owner = RasaHQ

git_version = $(shell git describe --tags --always)

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

build-images:
	docker build -t $(cr_url)/$(cr_owner)/rasalit-diet-explorer:latest \
		-t $(cr_url)/$(cr_owner)/rasalit-diet-explorer:$(git_version) \
		--target rasalit-diet-explorer .

	docker build -t $(cr_url)/$(cr_owner)/rasalit-live-nlu:latest \
		-t $(cr_url)/$(cr_owner)/rasalit-live-nlu:$(git_version) \
		--target rasalit-live-nlu .

	docker build -t $(cr_url)/$(cr_owner)/rasalit-nlu-cluster:latest \
		-t $(cr_url)/$(cr_owner)/rasalit-nlu-cluster:$(git_version) \
		--target rasalit-nlu-cluster .

	docker build -t $(cr_url)/$(cr_owner)/rasalit-overview:latest \
		-t $(cr_url)/$(cr_owner)/rasalit-overview:$(git_version) \
		--target rasalit-overview .

	docker build -t $(cr_url)/$(cr_owner)/rasalit-spelling:latest \
		-t $(cr_url)/$(cr_owner)/rasalit-spelling:$(git_version) \
		--target rasalit-spelling .

push-images:
	docker push $(cr_url)/$(cr_owner)/rasalit-diet-explorer:latest
	docker push $(cr_url)/$(cr_owner)/rasalit-diet-explorer:$(git_version)

	docker push $(cr_url)/$(cr_owner)/rasalit-live-nlu:latest
	docker push $(cr_url)/$(cr_owner)/rasalit-live-nlu:$(git_version)

	docker push $(cr_url)/$(cr_owner)/rasalit-nlu-cluster:latest
	docker push $(cr_url)/$(cr_owner)/rasalit-nlu-cluster:$(git_version)

	docker push $(cr_url)/$(cr_owner)/rasalit-overview:latest
	docker push $(cr_url)/$(cr_owner)/rasalit-overview:$(git_version)

	docker push $(cr_url)/$(cr_owner)/rasalit-spelling:latest
	docker push $(cr_url)/$(cr_owner)/rasalit-spelling:$(git_version)


clean:
	rm -rf build
	rm -rf dist

check: black flake test clean
