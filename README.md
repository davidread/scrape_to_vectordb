# Scrape website to a vector db


## Setup

```sh
python3 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Find your Claude API key: https://console.anthropic.com/settings/keys
```sh
echo "my-api-key" > ~/.anthropic/apikey
```

## Run

```sh
. venv/bin/activate
python scrape.py
python query.py
python rag.py
```
