#!/usr/bin/env bash
# Download the public datasets used by the evaluation suite into data/raw and unpack them.
# ~2.3 GB of downloads. All sets are freely redistributable for research (see evaluation/README.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW" "$ROOT/data"
cd "$RAW"

curl -L -o esc50.zip https://github.com/karolpiczak/ESC-50/archive/master.zip
curl -L -o donateacry.zip https://github.com/gveres/donateacry-corpus/archive/master.zip
curl -L -o ravdess_speech.zip "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
curl -L -o vivae.zip "https://zenodo.org/api/records/4066235/files/VIVAE.zip/content"
curl -L -o librispeech_dev_clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz
mkdir -p cremad fleurs_he
for s in test validation; do
  curl -L -o "cremad/$s.parquet" "https://huggingface.co/datasets/confit/cremad-parquet/resolve/main/data/$s-00000-of-00001.parquet"
done
curl -L -o fleurs_he/dev.tsv "https://huggingface.co/datasets/google/fleurs/resolve/main/data/he_il/dev.tsv"
curl -L -o fleurs_he/dev.tar.gz "https://huggingface.co/datasets/google/fleurs/resolve/main/data/he_il/audio/dev.tar.gz"

cd "$ROOT/data"
unzip -q -o raw/esc50.zip "ESC-50-master/audio/*" "ESC-50-master/meta/esc50.csv" && rm -rf esc50 && mv ESC-50-master esc50
unzip -q -o raw/donateacry.zip
unzip -q -o raw/ravdess_speech.zip -d ravdess
unzip -q -o raw/vivae.zip -x "__MACOSX/*" "*.DS_Store"
tar -xzf raw/librispeech_dev_clean.tar.gz
mkdir -p fleurs_he && tar -xzf raw/fleurs_he/dev.tar.gz -C fleurs_he && cp raw/fleurs_he/dev.tsv fleurs_he/
echo "done: $(ls)"
echo "CREMA-D wavs are materialised from the parquet files on first use of the evaluation suite."
