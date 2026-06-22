#!/usr/bin/env bash
set -euo pipefail

# Script publico de reexecucao do artigo.
#
# Este script executa somente etapas offline/reprodutiveis quando os dados
# locais necessarios estao disponiveis. Ele nao dispara geracao LLM, nao exige
# credenciais e nao acessa dados privados versionados fora deste repositorio.

PAN_XML="${PAN_XML:-data/pan2012/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml}"
PAN_PRED="${PAN_PRED:-data/pan2012/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt}"
PAN_PARQUET="${PAN_PARQUET:-data/processed/pan2012_train.parquet}"
SYNTH_DIR="${SYNTH_DIR:-data/synthetic}"
SYNTH_PARQUET="${SYNTH_PARQUET:-data/processed/synthetic.parquet}"
SYNTH_TRAIN="${SYNTH_TRAIN:-data/processed/synthetic_train.parquet}"
SYNTH_TEST="${SYNTH_TEST:-data/processed/synthetic_test.parquet}"
RESULTS_DIR="${RESULTS_DIR:-reports/final}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

run_cmd() {
  if command -v rtk >/dev/null 2>&1; then
    rtk "$@"
  else
    "$@"
  fi
}

section() {
  printf "\n== %s ==\n" "$1"
}

warn() {
  printf "AVISO: %s\n" "$1" >&2
}

mkdir -p data/processed "$RESULTS_DIR"

section "Validacao das personas"
run_cmd python src/persona_validator.py personas

section "Pre-processamento PAN 2012"
if [[ -f "$PAN_XML" && -f "$PAN_PRED" ]]; then
  run_cmd python -m src.preprocess \
    --train "$PAN_XML" \
    --predators "$PAN_PRED" \
    --output "$PAN_PARQUET"
else
  warn "PAN 2012 nao encontrado. Mantendo etapa de pre-processamento pulada."
  warn "Esperado: $PAN_XML"
  warn "Esperado: $PAN_PRED"
fi

section "Corpus sintetico"
if compgen -G "${SYNTH_DIR}/*.xml" >/dev/null; then
  run_cmd python -m src.evaluate \
    --create-synth "$SYNTH_DIR" \
    --synth "$SYNTH_PARQUET" \
    --split-synth \
    --synth-train "$SYNTH_TRAIN" \
    --synth-test "$SYNTH_TEST"
else
  warn "Nenhum XML sintetico encontrado em $SYNTH_DIR. Mantendo etapa pulada."
fi

section "Baseline E1"
if [[ -f "$PAN_PARQUET" ]]; then
  run_cmd python -m src.classifier \
    --input "$PAN_PARQUET" \
    --experiment baseline_full
  cp reports/baseline_results.csv "$RESULTS_DIR/baseline_results.csv"
else
  warn "Parquet PAN ausente: $PAN_PARQUET. Baseline E1 pulado."
fi

section "Experimentos E2-E5"
if [[ -f "$PAN_PARQUET" && -f "$SYNTH_PARQUET" ]]; then
  run_cmd python -m src.evaluate \
    --pan "$PAN_PARQUET" \
    --synth "$SYNTH_PARQUET" \
    --synth-train "$SYNTH_TRAIN" \
    --synth-test "$SYNTH_TEST" \
    --e2 --e3 --e4 --e5 \
    --top-n 20 \
    --export-knime \
    --reports-dir "$RESULTS_DIR"
else
  warn "E2-E5 pulado. Requer $PAN_PARQUET e $SYNTH_PARQUET."
fi

section "Resumo dos resultados"
if [[ -d "$RESULTS_DIR" ]]; then
  run_cmd python -m src.summarize_results \
    --results-dir "$RESULTS_DIR" \
    --baseline-csv "$RESULTS_DIR/baseline_results.csv" \
    --output "$RESULTS_DIR/summary.md"
fi

section "Concluido"
printf "Resultados: %s\n" "$RESULTS_DIR"
