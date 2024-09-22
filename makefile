
# --------------------------------------------------------------- venv

.PHONY: init # initialize .venv
init:
	pip install pip --upgrade
	pip install pipreqs
	rm -rf requirements.txt requirements.in
	pipreqs . --mode no-pin --encoding utf-8 --ignore .venv
	mv requirements.txt requirements.in

	pip install pip-tools
	pip-compile requirements.in -o requirements.txt -vvv
	
	rm -rf .venv
	python -m venv .venv
	./.venv/bin/python3 -m pip install -r requirements.txt
	echo "to activate venv, run: source .venv/bin/activate"

.PHONY: lock # freeze and dump .venv
lock:
	$(PYTHON_PATH) -m pip freeze > requirements.in
	pip-compile requirements.in -o requirements.txt -vvv

# --------------------------------------------------------------- docker

.PHONY: docker-install # run docker container
docker-install:
	docker-compose up --detach
	echo "to exec into docker container, run: docker exec -it main bash"

.PHONY: docker-build # save changes to container
docker-build:
	docker-compose build

.PHONY: docker-clean # wipe everything in all docker containers
docker-clean:
	docker-compose down

	-docker stop $$(docker ps -a -q)
	-docker rm $$(docker ps -a -q)
	-docker rmi $$(docker images -q)
	yes | docker container prune
	yes | docker image prune
	yes | docker volume prune
	yes | docker network prune
	yes | docker system prune
	
	docker ps --all
	docker images
	docker system df
	docker volume ls
	docker network ls

# --------------------------------------------------------------- conda

.PHONY: reqs-to-yaml # generate environment.yml from requirements.txt (idempotent)
conda-req-to-yaml:
	./.venv/bin/python3 -m pip install pyyaml
	./.venv/bin/python3 -c "import re, yaml; \
	requirements_text = open('requirements.txt').read(); \
	pattern = r'^(\S+)==(\S+)'; \
	matches = re.findall(pattern, requirements_text, re.MULTILINE); \
	requirements_dict = {name: version for name, version in matches}; \
	conda_env = {'name': 'con', 'channels': ['rusty1s', 'pytorch', 'nvidia', 'anaconda', 'conda-forge', 'defaults'], 'dependencies': ['python=3.11'] + [f'{package}={version}' for package, version in requirements_dict.items()]}; \
	yaml.dump(conda_env, open('environment.yml', 'w'), sort_keys=False);"

.PHONY: conda-gen-yaml # install conda to generate environment.yml from requirements.txt (idempotent)
conda-gen-yaml:
	conda update -n base -c defaults conda
	# conda config --env --set subdir osx-64
	# conda config --env --set subdir osx-arm64
	conda config --set auto_activate_base false
	conda info
	bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda create --yes --name con python=3.11; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate con; \
		\
		pip install -r requirements.txt; \
		\
		conda env export --no-builds | grep -v "prefix:" > environment.yml; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
		conda remove --yes --name con --all; \
	'

.PHONY: conda-install # install conda from environment.yml file
conda-install:
	bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda env create --file environment.yml --solver=libmamba; \
	'

.PHONY: conda-clean # wipe conda environment
conda-clean:
	# conda clean --all
	bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda remove --yes --name con --all; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
	'

# --------------------------------------------------------------- nohup

.PHONY: monitor # create nohup with restart on failure
monitor:
	if [ "$(filepath)" = "" ]; then echo "missing 'filepath' argument"; exit 1; fi
	bash -c '\
		monitor() { \
			while true; do \
				if ! ps -p $$(cat "monitor-process.pid" 2>/dev/null) > /dev/null 2>&1; then \
					echo "$$(date): process not running or died, (re)starting..." >> monitor.log; \
					nohup python "$(filepath)" > "monitor-process.log" 2>&1 & \
					echo $$! > "monitor-process.pid"; \
					echo "$$(date): started process with PID $$(cat monitor-process.pid)" >> monitor.log; \
				fi; \
				sleep 5; \
			done; \
		}; \
		monitor >> "monitor.log" 2>&1 & \
		echo $$! > "monitor.pid"; \
		echo "$$(date): monitor started" >> "monitor.log"; \
	'

.PHONY: monitor-tail # tail log of nohup process
monitor-tail:
	while true; do clear; tail -n 100 monitor-process.log; sleep 0.1; done
	# watch -n 0.1 "tail -n 100 monitor-process.log"

.PHONY: monitor-kill # kill nohup process
monitor-kill:
	-kill -9 $$(cat monitor.pid)
	rm -rf monitor.pid
	rm -rf monitor.log
	-kill -9 $$(cat monitor-process.pid)
	rm -rf monitor-process.pid
	rm -rf monitor-process.log

# --------------------------------------------------------------- utils

.PHONY: fmt # format codebase
fmt:
	./.venv/bin/python3 -m pip install isort
	./.venv/bin/python3 -m pip install ruff
	./.venv/bin/python3 -m pip install autoflake

	./.venv/bin/python3 -m isort .
	./.venv/bin/python3 -m autoflake --remove-all-unused-imports --recursive --in-place .
	./.venv/bin/python3 -m ruff format --config line-length=500 .

.PHONY: sec # check for vulns
sec:
	./.venv/bin/python3 -m pip install bandit
	./.venv/bin/python3 -m pip install safety
	
	./.venv/bin/python3 -m bandit -r .
	./.venv/bin/python3 -m safety check --full-report

.PHONY: up # pull and push changes
up:
	git pull
	git add .
	if [ -z "$(msg)" ]; then git commit -m "up"; else git commit -m "$(msg)"; fi
	git push

.PHONY: help # generate help message
help:
	@echo "Usage: make [target]\n"
	@grep '^.PHONY: .* #' makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1	\2/' | expand -t20
