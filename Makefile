.PHONY: install nltkdata test validate-personas preprocess baseline evaluate summary clean

PYTHON ?= python3
PAN_XML ?= data/pan2012/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml
PAN_PRED ?= data/pan2012/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt
PAN_PARQUET ?= data/processed/pan2012_train.parquet
SYNTH_PARQUET ?= data/processed/synthetic.parquet
SYNTH_TRAIN ?= data/processed/synthetic_train.parquet
SYNTH_TEST ?= data/processed/synthetic_test.parquet
RESULTS_DIR ?= reports/final

install:
	pip install -r requirements.txt

nltkdata:
	$(PYTHON) -c "import nltk; nltk.download('stopwords')"

test:
	pytest tests/ -v

validate-personas:
	$(PYTHON) src/persona_validator.py personas

preprocess:
	$(PYTHON) -m src.preprocess --train $(PAN_XML) --predators $(PAN_PRED) --output $(PAN_PARQUET)

baseline:
	$(PYTHON) -m src.classifier --input $(PAN_PARQUET) --experiment baseline_full
	cp reports/baseline_results.csv $(RESULTS_DIR)/baseline_results.csv

evaluate:
	$(PYTHON) -m src.evaluate --pan $(PAN_PARQUET) --synth $(SYNTH_PARQUET) --synth-train $(SYNTH_TRAIN) --synth-test $(SYNTH_TEST) --e2 --e3 --e4 --e5 --top-n 20 --export-knime --reports-dir $(RESULTS_DIR)

summary:
	$(PYTHON) -m src.summarize_results --results-dir $(RESULTS_DIR) --baseline-csv $(RESULTS_DIR)/baseline_results.csv --output $(RESULTS_DIR)/summary.md

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
