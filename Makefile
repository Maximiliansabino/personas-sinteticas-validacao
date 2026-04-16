.PHONY: install deps nltkdata test preprocess baseline evaluate clean

install: deps nltkdata

deps:
	pip install -r requirements.txt

nltkdata:
	python -c "import nltk; nltk.download('stopwords')"

test:
	pytest tests/ -v --cov=src

preprocess:
	python -m src.preprocess \
	  --train data/pan2012/train/corpus.xml \
	  --predators data/pan2012/train/predators.txt

baseline:
	python -m src.classifier \
	  --input data/processed/pan2012_train.parquet

evaluate:
	python -m src.evaluate --all --export-knime

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
