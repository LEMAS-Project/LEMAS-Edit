#!/usr/bin/env bash

# Autoregressive codec-based speech editing demo wrapper for LEMAS-Edit HF.
# Runs the AR edit model on the LEMAS-Edit demo set using Azure align JSONs.

set -e

ROOT_DIR=$(pwd)
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

DATA_ROOT="${ROOT_DIR}/pretrained_models/demos/lemas_edit_test"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m lemas_edit.scripts.inference_lemas_editing \
  --wav_dir "${DATA_ROOT}/vocals" \
  --align_dir "${DATA_ROOT}/align" \
  --save_dir "${DATA_ROOT}/Autoregressive_Edit" \
  "$@"
