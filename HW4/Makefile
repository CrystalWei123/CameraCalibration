init:
	python3 -m pip install --user pipenv
	pipenv --python 3.7
	pipenv install --dev --skip-lock

LIB_LINT = data
RUN_LINT = run_sift.py
PYTHON_LINT = $(LIB_LINT) $(RUN_LINT)

lint: ending flake8 pylint mypy

ending:
	! grep -rHnP --include="*.py" --include="*.json" --include="*.md" "\x0D" ${PYTHON_LINT}

flake8:
	pipenv run flake8 ${PYTHON_LINT}

pylint:
	pipenv run pylint ${PYTHON_LINT}

mypy:
	pipenv run mypy --ignore-missing-imports ${PYTHON_LINT}

clean:
	rm -rf .mypy_cache/

run:
	pipenv run python3 run_sift.py
