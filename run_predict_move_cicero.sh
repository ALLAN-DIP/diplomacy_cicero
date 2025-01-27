#!/usr/bin/env bash

set -euo pipefail

mkdir -p logs/
NOW=$(date -u +'%Y_%m_%d_%H_%M_%S')
LOG_FILE=logs/"$NOW"_denis_cicero.txt

CICERO_DIR=/home/exouser/cicero

GAME_COMMAND=(
  python fairdiplomacy_external/predict_move_from_state.py
)

time docker run \
  --rm \
  --gpus all \
  --name cicero_latest_"$RANDOM" \
  --volume /home/exouser/denis:/diplomacy_cicero/data/denis:rw \
  --volume /home/exouser/diplomacy_cicero/fairdiplomacy_external:/diplomacy_cicero/fairdiplomacy_external:ro \
  --volume "$CICERO_DIR"/agents:/diplomacy_cicero/conf/common/agents:ro \
  --volume "$CICERO_DIR"/gpt2:/usr/local/lib/python3.7/site-packages/data/gpt2:ro \
  --volume "$CICERO_DIR"/models:/diplomacy_cicero/models:ro \
  --workdir /diplomacy_cicero \
  ghcr.io/allan-dip/diplomacy_cicero:latest \
  "${GAME_COMMAND[@]}" |&
  tee "$LOG_FILE"
