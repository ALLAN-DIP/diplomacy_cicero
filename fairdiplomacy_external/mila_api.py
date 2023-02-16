from collections import defaultdict
from fnmatch import fnmatch
import gc
import html
import http.client
import math
import pickle
import traceback
from requests.models import Response
import requests
import socket
import urllib3.exceptions
from fairdiplomacy.agents.parlai_message_handler import (
    ParlaiMessageHandler,
    pseudoorders_initiate_sleep_heuristics_should_trigger,
)
from fairdiplomacy.agents.player import Player
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import (
    Json,
    MessageDict,
    MessageHeuristicResult,
    OutboundMessageDict,
    Phase,
    Power,
    Timestamp,
    Context,
)
import random
import hashlib
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import getpass
import itertools
import json
import logging
import os
import pathlib
import time
from fairdiplomacy.data.build_dataset import (
    DRAW_VOTE_TOKEN,
    UNDRAW_VOTE_TOKEN,
    DATASET_DRAW_MESSAGE,
    DATASET_NODRAW_MESSAGE,
)
from fairdiplomacy.utils.agent_interruption import ShouldStopException, set_interruption_condition
from fairdiplomacy.utils.atomicish_file import atomicish_open_for_writing_binary
from fairdiplomacy.utils.slack import GLOBAL_SLACK_EXCEPTION_SWALLOWER
from fairdiplomacy.utils.typedefs import build_message_dict, get_last_message
from fairdiplomacy.viz.meta_annotations.annotator import MetaAnnotator
from parlai_diplomacy.utils.game2seq.format_helpers.misc import POT_TYPE_CONVERSION
import torch
from fairdiplomacy.utils.game import game_from_view_of
from fairdiplomacy.viz.meta_annotations import api as meta_annotations
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.webdip.message_approval_cache_api import (
    PRESS_COUNTRY_ID_TO_POWER,
    ApprovalStatus,
    MessageReviewData,
    compute_phase_message_history_state_with_power,
    flag_proposal_as_stale,
    gen_id,
    get_message_review,
    get_redis_host,
    get_should_run_backup,
    get_kill_switch,
    maybe_get_phase_message_history_state,
    set_message_review,
    delete_message_review,
    botgame_fp_to_context,
    update_phase_message_history_state,
)
from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.build_dataset import (
    GameVariant,
    TERR_ID_TO_LOC_BY_MAP,
    COUNTRY_ID_TO_POWER_OR_ALL_MY_MAP,
    COUNTRY_POWER_TO_ID,
    get_valid_coastal_variant,
)
from fairdiplomacy.webdip.utils import turn_to_phase
from fairdiplomacy.utils.slack import send_slack_message
import heyhi
from conf import conf_cfgs
from parlai_diplomacy.wrappers.classifiers import INF_SLEEP_TIME

from fairdiplomacy.agents.searchbot_agent import SearchBotAgentState

import argparse
import asyncio
import json as json
import random
import sys
import time
from pathlib import Path
from typing import Optional

from diplomacy import connect
from diplomacy.client.network_game import NetworkGame
from diplomacy.utils.export import to_saved_game_format


def play_mila(
    hostname: str,
    port: int,
    game_id: str,
    power_name: str,
    bot_type: str,
    discount_factor: float,
    gamedir: Path,
) -> None:


    logging.info(f"CICERO joining game: {game_id} as {power_name}")
    connection = await connect(hostname, port)
    channel = await connection.authenticate(
        f"CICERO_{power_name}", "password"
    )
    game: NetworkGame = await channel.join_game(game_id=game_id, power_name=power_name)

    # Wait while game is still being formed
    logging.info(f"Waiting for game to start")
    while game.is_game_forming:
        await asyncio.sleep(2)
    #     print("", end=".")
    # print()

    # Playing game
    logging.info(f"Started playing")

    dipcc_game = start_dipcc_game()
    
    # {phase: number of incoming messages for agent}
    dialogue_state = {}
    # {phase: timestamp}
    last_sent_message_time = {}
    # {phase: time_sent (from Mila json)}
    last_received_message_time = {}

    while not game.is_game_done:
        # extract game state from Mila to dipcc game
        if not game.powers[power_name].is_eliminated():
            # see the code flow in _play_webdip_without_retries
            # message phase - see run_message_approval_flow from webdip_api
            # gen message - see generate_message_for_approval
            while not get_should_stop():
                msg = generate_message(agent, dipcc_game, game)
            
            # order phase 
            agent_orders = agent.get_orders(dipcc_game)
            game.set_orders(power_name=power_name, orders=agent_orders, wait=False)

            while not has_phase_changed():
                await asyncio.sleep(2)
            if has_phase_changed(game, power_name):
                update_and_process_dipcc_game(game, dipcc_game)
                init_phase()

def init_phase():
    """     
    update new phase to the following Dict:
    - dialogue_state
    - last_sent_message_time
    - last_received_message_time
    """

def has_phase_changed():
    """ 
    check for game phase 
    """

def has_state_changed():
    """ 
    check from dialogue_state 
    """

def get_should_stop():
    """ 
    check for state change 1. new message 
    stop if 1. close to deadline! (time to submit order)
    call update_press_dipcc_game() 
    """

def update_press_dipcc_game():
    """ 
    check new messages in current phase from Mila 
    receiver == power_name and message timesent > last_received_message_time[current phase]
    update message in dipcc game 
    update last_received_message_time 
    """

def update_order_and_process_dipcc_game():
    """     
    check orders in current phase (of dipcc) from Mila
    get orders for power != power_name 
    submit orders and process game 
    """

def generate_message():
    """     
    handle with sleep time (if we want to send start new message)
    reply back to recipient for new message 
    gen message to dipcc and Mila 
    """

def start_dipcc_game() -> Game:
        # with open(gamedir / f"{power_name}_output.json", mode="w") as file:
        #     json.dump(
        #         to_saved_game_format(game), file, ensure_ascii=False, indent=2
        #     )
        #     file.write("\n")
        # game = Game.from_json(gamedir+f"/{power_name}_output.json")
        game = Game()
        game.set_scoring_system(Game.SCORING_SOS)
        game.set_metadata("phase_minutes", str(1))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="host IP address (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8432,
        help="port to connect to the game (default: %(default)s)",
    )
    parser.add_argument(
        "--game_id",
        type=str,
        required=True,
        help="game id of game created in DATC diplomacy game",
    )
    parser.add_argument(
        "--power",
        choices=POWERS,
        required=True,
        help="power name",
    )
    parser.add_argument(
        "--outdir", type=Path, help="output directory for game json to be stored"
    )
    args = parser.parse_args()
    host: str = args.host
    port: int = args.port
    game_id: str = args.game_id
    power: str = args.power
    agent = build_agent_from_cfg('agents/cicero.prototxt')
    outdir: Optional[Path] = args.outdir

    if outdir is not None and not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    
    play_mila(
            hostname=host,
            port=port,
            game_id=game_id,
            agent=agent
            power_name=power,
            outdir=outdir,
    )

if __name__ == "__main__":
    main()
    
