#!/usr/bin/env bash

set -euo pipefail

mkdir -p logs/
NOW=$(date -u +'%Y_%m_%d_%H_%M_%S')
LOG_FILE=logs/"$NOW"_ctrld_cicero.txt

CICERO_DIR=/home/exouser/cicero

GAME_COMMAND=(
  python fairdiplomacy_external/mila_api_friction_advisor.py
  --game_type 2
  "$@"
)

time docker run \
  --rm \
  --gpus all \
  --name cicero_latest_"$RANDOM" \
  --volume /home/exouser/diplomacy_cicero/fairdiplomacy_external:/diplomacy_cicero/fairdiplomacy_external:rw \
  --volume "$CICERO_DIR"/agents:/diplomacy_cicero/conf/common/agents:ro \
  --volume "$CICERO_DIR"/gpt2:/usr/local/lib/python3.7/site-packages/data/gpt2:ro \
  --volume "$CICERO_DIR"/models:/diplomacy_cicero/models:ro \
  --volume "$CICERO_DIR"/AMR:/diplomacy_cicero/fairdiplomacy_external/AMR:ro \
  --workdir /diplomacy_cicero \
  ghcr.io/allan-dip/ctrld_cicero:latest \
  "${GAME_COMMAND[@]}" |&
  tee "$LOG_FILE"
