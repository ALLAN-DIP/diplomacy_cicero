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
import math
from pathlib import Path
from typing import Optional

from diplomacy import connect
from diplomacy.client.network_game import NetworkGame
from diplomacy.utils.export import to_saved_game_format



class milaWrapper:

    def __init__(
        self, 
        cicero_path: Path
    ):
        self.game: NetworkGame = None
        self.dipcc_game: Game = None
        self.prev_state = 0                                         # number of number received messages in the current phase
        self.dialogue_state = {}                                    # {phase: number of all (= received + new) messages for agent}
        self.last_sent_message_time = 0                             # {phase: timestamp} 
        self.last_received_message_time = 0                         # {phase: time_sent (from Mila json)}
        self.agent = build_agent_from_cfg(cicero_path)              # build cicero from config 
        self.dipcc_current_phase = None

    async def play_mila(
        hostname: str,
        port: int,
        game_id: str,
        power_name: str,
        bot_type: str,
        discount_factor: float,
        gamedir: Path,
    ) -> None:

        logging.info(f"CICERO joining game: {game_id} as {power_name}")
        connection = connect(hostname, port)
        channel = connection.authenticate(
            f"CICERO_{power_name}", "password"
        )
        self.game: NetworkGame = await channel.join_game(game_id=game_id, power_name=power_name)

        # Wait while game is still being formed
        logging.info(f"Waiting for game to start")
        while self.game.is_game_forming:
            await asyncio.sleep(2)

        # Playing game
        logging.info(f"Started playing")

        self.dipcc_game = self.start_dipcc_game(game.deadline)

        while not self.game.is_game_done:
            self.phase_start_time = time.time()
            self.dipcc_current_phase = self.game.get_current_phase()
            # extract game state from Mila to dipcc game
            if not self.game.powers[power_name].is_eliminated():
                # press 
                # see the code flow in _play_webdip_without_retries
                # message phase - see run_message_approval_flow from webdip_api
                # gen message - see generate_message_for_approval
                while not self.get_should_stop():
                    # if there is new message 
                    if self.has_state_changed():
                        self.update_press_dipcc_game()

                    # reply/gen new message
                    msg = self.generate_message(self.agent, self.dipcc_game, self.game)
                    # update message in both game
                
                # order  
                agent_orders = self.agent.get_orders(self.dipcc_game)
                self.game.set_orders(power_name=power_name, orders=agent_orders, wait=False)

                while not self.has_phase_changed():
                    await asyncio.sleep(2)
                if self.has_phase_changed(dipcc_current_phase):
                    self.phase_end_time = time.time()
                    self.update_order_and_process_dipcc_game()
                    self.init_phase()

    def init_phase():
        """     
        update new phase to the following Dict:
        """
        self.dipcc_current_phase = self.game.get_current_phase()
        self.prev_state = 0

    def has_phase_changed()->bool:
        """ 
        check for game phase 
        """
        return current_phase == game.get_current_phase()

    def has_state_changed()->bool:
        """ 
        check from dialogue_state 
        """
        mila_phase = self.game.get_current_phase()

        phase_messages = self.get_messages(
            messages=self.game.messages, power=power_name
        )

        phase_num_messages = len(phase_messages.values())
        self.dialogue_state[mila_phase] = phase_num_messages

        has_state_changed = self.prev_state == self.dialogue_state[mila_phase]
        self.prev_state = self.dialogue_state[mila_phase]

        return has_state_changed

    def get_should_stop(deadline: int)->bool:
        """ 
        stop if 
        1. it is close to deadline! (time to submit order)
        2. stale pseudo orders
        """
        close_to_deadline = deadline - 15
        current_time = time.time()

        # TODO: add stale pseudo order similar to get_should_stop_condition
        if  current_time - self.phase_start_time >= close_to_deadline:
            return True   
        else:
            return False

    def update_press_dipcc_game(power_name: POWERS):
        """ 
        1. check new messages in current phase from Mila 
        2. update message in dipcc game 
        3. update last_received_message_time 
        """
        mila_phase = self.game.get_current_phase()

        phase_messages = self.get_messages(
            messages=self.game.messages, power=power_name
        )
        most_recent = self.last_received_message_time

        for timesent, dict_message in phase_messages:
            if timesent > self.last_received_message_time:
                if timesent > most_recent:
                    most_recent = timesent

                self.dipcc_game.add_message(
                    dict_message['sender'],
                    power_name,
                    dict_message['message'],
                    time_sent=Timestamp.from_seconds(int(timesent * 1e-6)),
                    increment_on_collision=True,
                )

        self.last_received_message_time = most_recent

    def update_and_process_dipcc_game(self, power_name):
        """     
        Inputs orders from the bygone phase into the dipcc game and process the dipcc game.
        """

        dipcc_game = self.dipcc_game
        mila_game = self.game
        dipcc_phase = dipcc_game.get_state()['name'] # short name for phase
        orders_from_prev_phase = mila_game.order_history[dipcc_phase] 
        
        # gathering orders from other powers from the phase that just ended
        for power, orders in orders_from_prev_phase:
            if power != power_name: 
                dipcc_game.set_orders(power, orders)

        dipcc_game.process() # processing the orders set and moving on to the next phase of the dipcc game

    def generate_message():
        """     
        1. handle with sleep time (if we want to send start new message)
        2. reply back to recipient for new message 
        3. gen message to dipcc and Mila 
        """
    def get_messages(
        self, 
        messages: SortedDict,
        power: str,
        ):

        return {message.time_sent: {'message': message, 'sender': message.sender}
                for message in messages
                if message.recipient in [power]}

    def start_dipcc_game(
        deadline: int
        ) -> Game:
            # with open(gamedir / f"{power_name}_output.json", mode="w") as file:
            #     json.dump(
            #         to_saved_game_format(game), file, ensure_ascii=False, indent=2
            #     )
            #     file.write("\n")
            # game = Game.from_json(gamedir+f"/{power_name}_output.json")
            if deadline ==0:
                deadline = 1
            else:
                deadline = int(ceil(deadline/60))
            game = Game()
            game.set_scoring_system(Game.SCORING_SOS)
            game.set_metadata("phase_minutes", str(deadline))

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
        "--agent",
        type=Path,
        required=True,
        default ="agents/cicero.prototxt",
        help="path to prototxt with agent's configurations (default: %(default)s)",
    )
    parser.add_argument(
        "--outdir", type=Path, help="output directory for game json to be stored"
    )
    args = parser.parse_args()
    host: str = args.host
    port: int = args.port
    game_id: str = args.game_id
    power: str = args.power
    outdir: Optional[Path] = args.outdir

    if outdir is not None and not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    milaWrapper()
    asyncio.run(
        play_mila(
    milaWrapper()
    asyncio.run(
        play_mila(
            hostname=host,
            port=port,
            game_id=game_id,
            power_name=power,
            outdir=outdir,
        )
    )

async def test_mila_function():
    """ 
    The function is to test ability that we can access Mila game on TACC 
    Manually replace GAMEID and USERNAME to test accessing Mila game features
    """

    game_id = GAMEID
    connection = await connect('shade.tacc.utexas.edu', 8432)
    channel = await connection.authenticate(
        USERNAME, "password"
    )
    game: NetworkGame = await channel.join_game(game_id=game_id, power_name="ENGLAND")

    logging.info(f"Waiting for game to start")
    # while game.is_game_forming:
    #     await asyncio.sleep(2)
    curr_phase = game.get_current_phase()
    while not game.is_game_done:
        print(f" game history {game.state_history}")
        print(f" message history {game.message_history}")
        print(f" order history {game.order_history}")
        possible_orders = game.get_all_possible_orders()
        ENG_orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations('ENGLAND')
                    if possible_orders[loc]]
        await game.set_orders(power_name='ENGLAND', orders=ENG_orders, wait=False)
        while curr_phase == game.get_current_phase():
            print(f" message in current phase {game.messages}")
            await asyncio.sleep(1)
        curr_phase = game.get_current_phase()
        )
    )

async def test_mila_function():
    """ 
    The function is to test ability that we can access Mila game on TACC 
    Manually replace GAMEID and USERNAME to test accessing Mila game features
    """

    game_id = GAMEID
    connection = await connect('shade.tacc.utexas.edu', 8432)
    channel = await connection.authenticate(
        USERNAME, "password"
    )
    game: NetworkGame = await channel.join_game(game_id=game_id, power_name="ENGLAND")

    logging.info(f"Waiting for game to start")
    # while game.is_game_forming:
    #     await asyncio.sleep(2)
    curr_phase = game.get_current_phase()
    while not game.is_game_done:
        print(f" game history {game.state_history}")
        print(f" message history {game.message_history}")
        print(f" order history {game.order_history}")
        possible_orders = game.get_all_possible_orders()
        ENG_orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations('ENGLAND')
                    if possible_orders[loc]]
        await game.set_orders(power_name='ENGLAND', orders=ENG_orders, wait=False)
        while curr_phase == game.get_current_phase():
            print(f" message in current phase {game.messages}")
            await asyncio.sleep(1)
        curr_phase = game.get_current_phase()

if __name__ == "__main__":
    main()
    
