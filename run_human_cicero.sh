#!/bin/bash
module load tacc-apptainer
export WORK=/work/08801/wwongkam/ls6/ALLAN
export CICERO=/work/08801/wwongkam/ls6/CICERO/cicero
# export GAME_COMMAND="python fairdiplomacy_external/mila_api.py --game_id $1 --host $2 --power $3"
export GAME_COMMAND="python fairdiplomacy_external/mila_api.py --game_id $1 --host $2 --power $3 --game_type 2"
export CUDA_VISIBLE_DEVICES=$4
cd $CICERO

apptainer run --nv --cleanenv --ipc --no-eval --no-init --no-umask --pid --env CUDA_VISIBLE_DEVICES=$4  \
  --bind "$WORK"/diplomacy_cicero/fairdiplomacy_external:/diplomacy_cicero/fairdiplomacy_external \
  --bind "$CICERO"/agents:/diplomacy_cicero/conf/common/agents \
  --bind "$CICERO"/models:/diplomacy_cicero/models \
  --bind "$CICERO"/gpt2:/usr/local/lib/python3.7/site-packages/data/gpt2 \
  --pwd /diplomacy_cicero cicero_latest.sif $GAME_COMMAND
