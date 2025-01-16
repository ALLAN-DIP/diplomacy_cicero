#!/usr/bin/env bash

set -euo pipefail

SCRIPT_FILE=$(realpath "$0")
SCRIPT_NAME=$(basename "$SCRIPT_FILE")

function show_usage() {
  {
    echo "usage: bash $SCRIPT_NAME [--help] BOT_TAG BOT_TYPE BOT_ARGS..."
    echo
    echo 'Run CICERO from an OCI image using Docker.'
    echo '- BOT_TAG is the tag assigned when building the image.'
    echo '- BOT_TYPE is the type of bot to run and must be either "advisor" or "player".'
    echo '- BOT_ARGS are passed as arguments to the script within the running container.'
  } >&2
}

if [[ $# -lt 2 ]] || [[ ${1:-} = --help ]]; then
  show_usage
  exit 1
fi

BOT_TAG=$1
BOT_TYPE=$2
shift 2

if [[ $BOT_TYPE != advisor ]] && [[ $BOT_TYPE != player ]]; then
  {
    echo 'Invalid BOT_TYPE given: must be "advisor" or "player".'
    echo
  } >&2
  show_usage
  exit 1
fi

set -x

CICERO_DIR=/media/volume/cicero-base-models

GAME_COMMAND=(
  python fairdiplomacy_external/mila_api_"$BOT_TYPE".py
  --game_type 2
  "$@"
)

mkdir -p data/logs/
NOW=$(date -u +'%Y_%m_%d_%H_%M_%S')
LOG_FILE=data/logs/$NOW.txt

time docker run \
  --rm \
  --gpus all \
  --name cicero_"$BOT_TAG"_"$RANDOM" \
  --volume "$CICERO_DIR"/agents:/diplomacy_cicero/conf/common/agents:ro \
  --volume "$CICERO_DIR"/gpt2:/usr/local/lib/python3.7/site-packages/data/gpt2:ro \
  --volume "$CICERO_DIR"/models:/diplomacy_cicero/models:ro \
  --workdir /diplomacy_cicero \
  ghcr.io/allan-dip/diplomacy_cicero:"$BOT_TAG" \
  "${GAME_COMMAND[@]}" |&
  tee "$LOG_FILE"
