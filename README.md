# Diplomacy Cicero and Diplodocus

This code contains checkpoints and training code the following papers:

* ["Human-Level Play in the Game of Diplomacy by Combining Language Models with Strategic Reasoning"](https://www.science.org/doi/10.1126/science.ade9097) published in Science, November 2022.
* ["Mastering the Game of No-Press Diplomacy via Human-Regularized Reinforcement Learning and Planning"](https://arxiv.org/abs/2210.05492) in review at ICLR 2023.

### Code
A very brief orientation:
- Most of the language modeling and generation code is in [parlai_diplomacy](parlai_diplomacy), and leverages the [ParlAI framework](https://github.com/facebookresearch/ParlAI) for running and finetuning the language models involved.
- Within the [agents](fairdiplomacy/agents) directory, the central logic for Cicero's strategic planning lives [here](fairdiplomacy/agents/br_corr_bilateral_search.py) and [here](fairdiplomacy/agents/bqre1p_agent.py). The latter also contains the core logic for Diplodocus's strategic planning. "bqre1p" was the internal dev name for DiL-piKL, and "br_corr_bilateral" the internal dev name for Cicero's bilateral and correlated planning components.
- The dialogue-free model architectures for RL are [here](fairdiplomacy/models/base_strategy_model/base_strategy_model.py), and the bulk of the training logic lives [here](fairdiplomacy/models/base_strategy_model/train_sl.py)
- The RL training code for both Cicero and Diplodocus is [here](fairdiplomacy/selfplay)
- The [conf](conf) directory contains various configs for Cicero, Diplodocus, benchmark agents, and training configs for RL.
- A separately licensed subfolder of this repo [here](fairdiplomacy_external) contains some utilities for visually rendering games, or connecting agents to be run online.

### Game info
Diplomacy is a strategic board game set in 1914 Europe.
The board is divided into fifty-six land regions and nineteen sea regions.
Forty-two of the land regions are divided among the seven Great Powers of the game: Austria-Hungary, England, France, Germany, Italy, Russia, and Turkey.
The remaining fourteen land regions are neutral at the start of the game.

Each power controls some regions and some units.
The number of the units controlled depends on the number of the controlled key regions called Supply Centers (SCs).
Simply put, more SCs means more units.
The goal of the game is to control more than half of all SCs by moving units into these regions and convincing other players to support you.

You can find the full rules [here](https://en.wikibooks.org/wiki/Diplomacy/Rules).
To get the game's spirit, watch [some](https://www.youtube.com/c/diplostrats) [games](https://www.youtube.com/playlist?list=PLmbDtCxqXA5CyFoBmB5dJHHOHeLQ0Nd-Y) with comments.
You can play the game online on [webDiplomacy](https://webdiplomacy.net/) either against bots or humans.

### Installation

To limit differences when used on various systems, CICERO is built in and run from an OCI image. Use the following commands to build the image: 

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/ALLAN-DIP/diplomacy_cicero.git
cd diplomacy_cicero/

# Build OCI image
make build
```

In addition, <https://js2.jetstream-cloud.org/project/shares/534a349e-9d9a-4757-b820-12d2cb30c76c/> should be mounted at `/media/volume/cicero-base-models`. If it is mounted elsewhere, change the hardcoded path for `CICERO_DIR` in [`run_cicero.sh`](run_cicero.sh) accordingly.

### Run CICERO on Jetstream2

If you have not already done so, follow the instructions in the "[Installation](#installation)" section of this documentation.

To use the advisor, run a command similar to the following:

```bash
./run_cicero.sh \
  latest \
  advisor \
  --host diplomacy.example.org \
  --use-ssl \
  --game_id test_game \
  --human_powers AUSTRIA ENGLAND \
  --advice_levels 'MESSAGE|MOVE|OPPONENT_MOVE' 'MOVE|OPPONENT_MOVE'
```

To use the player, run a command similar to the following:

```bash
./run_cicero.sh \
  latest \
  player \
  --host diplomacy.example.org \
  --use-ssl \
  --game_id test_game \
  --power AUSTRIA
```

Use the following guidance when modifying arguments:

- `--host` should be set to whichever domain the server is hosted at.
- If your server is not using SSL, then drop the `--use-ssl` argument.
- `--game_id` must match an existing game.
- `--advice_levels` is the joining of the values for each advice type with `|`. For example, `'MESSAGE|MOVE|OPPONENT_MOVE` means to provide advice for messages, moves, and opponent moves. The values need to be quoted in the shell so `|` is not interpreted as a shell pipe. For more information, see the output of `./run_cicero.sh latest advisor --help`.

### Downloading model files

Please email <diplomacyteam@meta.com> to request the password. Then run `bash bin/download_model_files.sh <PASSWORD>`. This will download and decrypt all relevant model files into `./models`. This might take awhile. Please note the model files have their own license separate from the code in this repository. More details on this [can be found below](#license-for-model-weights).

### Accessing Cicero's experiment games

JSON data and visualizations for games that Cicero played in are located in [data/cicero_redacted_games](data/cicero_redacted_games). Only conversations with players who have consented to having their dialogue released are included. Please refer to the (separately-licensed) [fairdiplomacy_external](fairdiplomacy_external) subdirectory for details on HTML visualizations.

### Getting started

The front-end for most tasks is `run.py`, which can run various tasks specified by a protobuf config. The config schema can be found at `conf/conf.proto`, and example configs for different tasks can be found in the `conf` folder. This can be used for most tasks (except training parlai models): training no-press models, comparing agents, profiling things, launching an agent on webdip, etc.

The config specification framework, called HeyHi, [is explained here](heyhi/README.md)

A core abstraction is an `Agent`, which is specified by an `Agent` config whose schema lives in `conf/agents.proto`.

### Simulating games between agents

To simulate 1v6 games between a pair of agents, you can run the `compare_agents` task. For example, to play one Cicero agent as Turkey against six full-press imitation agents, you can run

`python run.py --adhoc --cfg conf/c01_ag_cmp/cmp.prototxt Iagent_one=agents/cicero.prototxt Iagent_six=agents/ablations/cicero_imitation_only.prototxt power_one=TURKEY`

If you don't have sufficient memory to load two agents, you can load a single agent in self-play with the `use_shard_agent=1` flag:

`python run.py --adhoc --cfg conf/c01_ag_cmp/cmp.prototxt Iagent_one=agents/cicero.prototxt use_shared_agent=1 power_one=TURKEY`

### Training models in RL

To run the training for Cicero and/or Diplodocus:

```
python run.py —adhoc —cfg conf/c04_exploit/research_20221001_paper_cicero.prototxt launcher.slurm.num_gpus=256

python run.py —adhoc —cfg conf/c04_exploit/research_20221001_paper_diplodocus_high.prototxt launcher.slurm.num_gpus=256
```

The above training commands are designed for running on an appropriately configured Slurm cluster with a fast cross-machine shared filesystem. One can also instead pass `launcher.local.use_local=true` to run them on locally, e.g. on an individual 8-GPU-or-more GPU machine but training may be very slow.

### Other tasks
See [here](fairdiplomacy_external) for some separately-licensed code for rendering game jsons with HTML, as well as connecting agents to run on [webdiplomacy.net](https://webdiplomacy.net).

### Supervised training of baseline models
Supervised training and/or behavioral cloning for various dialogue-conditional models as well as pre-RL baseline dialogue-free models involves some of the scripts in [parlai_diplomacy](parlai_diplomacy) via the ParlAI framework, and on the dialogue-free side, some of the configs [conf/c02_sup_train](conf/c02_sup_train) and [train_sl.py](fairdiplomacy/models/base_strategy_model/train_sl.py). However the dataset of human games and/or dialogue is NOT available here, so the relevant code and configs are likely to be of limited use. They are provided here mostly as documentation for posterity.

However, as mentioned above pre-trained models are available, and with sufficient compute power, re-running the RL on top of these pre-trained models is also possible without any external game data.


### Pre-commit hooks

Run `pre-commit install` to install pre-commit hooks that will auto-format python code before commiting it.

Or you can do this manually. Use [black](https://github.com/psf/black) auto-formatter to format all python code.
For protobufs use `clang-format-8 conf/*.proto -i`.

### Tests

To run tests locally run `make test`.

We have 2 level of tests: fast, unit tests (run with `make test_fast`) and slow, integration tests (run with `make test_integration`).
The latter aims to use the same entry point as users do, i.e., `run.py` for the HeyHi part and `diplom` for the ParlAi.

We use `pytest` to run and discover tests. Some useful [pytest](https://docs.pytest.org/en/stable/) commands.

To run all tests in your current directory, simply run:
```
pytest
```

To run tests from a specific file, run:
```
pytest <filepath>
```

To use name-based filtering to run tests, use the flag `-k`. For example, to only run tests with `parlai` in the name, run:
```
pytest -k parlai
```

For verbose testing logs, use `-v`:
```
pytest -v -k parlai
```

To print the output from a test or set of tests, use `-s`; this also allows you to set breakpoints:
```
pytest -s
```

To view the durations of all tests, run with the flag `--durations=0`, e.g.:
```
pytest --durations=0 unit_tests/
```

## License for Code
The following license, which is also available [here](LICENSE.md), covers the content in this repo *except* for the [fairdiplomacy_external](fairdiplomacy_external) directory. The content of fairdiplomacy_external is separately licenced under a version of the AGPL, see the license file within that directory for details.

```
(covers this repo except for the fairdiplomacy_external directory)
MIT License

Copyright (c) Meta, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## License for Model Weights

We are releasing model weights under a separate license: [CC-BY-NC (version 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode). This license is copied into this repository for convenience: [LICENSE_FOR_MODEL_WEIGHTS.txt](LICENSE_FOR_MODEL_WEIGHTS.txt).
