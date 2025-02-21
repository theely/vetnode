

# Development


## Set-up Python Virtual environement

Create a virtual environment:
```console
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the CLI

```
cd src
python -m shrike diagnose ../templates/simple-config.yaml
```


## Running Tests
From the FirecREST root folder run pytest to execute all unit tests.
```console
source .venv/bin/activate
pip install -r ./requirements.txt -r ./requirements-testing.txt
pytest
```
