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
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent as PyBQRE1PAgent

from conf.agents_pb2 import *
import google.protobuf.message
import heyhi


import argparse
import asyncio
import json as json
import sys
import time
import math
from pathlib import Path
from typing import Optional

from diplomacy import connect
from diplomacy import Message
from diplomacy.client.network_game import NetworkGame
from diplomacy.utils.export import to_saved_game_format

MESSAGE_DELAY_IF_SLEEP_INF = Timestamp.from_seconds(60)
ProtoMessage = google.protobuf.message.Message

class milaWrapper:

    def __init__(self):
        self.game: NetworkGame = None
        self.dipcc_game: Game = None
        self.prev_state = 0                                         # number of number received messages in the current phase
        self.dialogue_state = {}                                    # {phase: number of all (= received + new) messages for agent}
        self.last_sent_message_time = 0                             # {phase: timestamp} 
        self.last_received_message_time = 0                         # {phase: time_sent (from Mila json)}                                      
        self.dipcc_current_phase = None                             
        self.last_successful_message_time = None                    # timestep for last message successfully sent in the current phase                       
        self.reuse_stale_pseudo_after_n_seconds = 45                # seconds to reuse pseudo order to generate message
        
        agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
        print(f"successfully load cicero config")

        self.agent = PyBQRE1PAgent(agent_config.bqre1p)

    async def play_mila(
        self,
        hostname: str,
        port: int,
        game_id: str,
        power_name: str,
        gamedir: Path,
    ) -> None:
        
        print(f"CICERO joining game: {game_id} as {power_name}")
        connection = await connect(hostname, port)
        channel = await connection.authenticate(
            f"CICERO_{power_name}", "password"
        )
        self.game: NetworkGame = await channel.join_game(game_id=game_id, power_name=power_name)

        # Wait while game is still being formed
        print(f"Waiting for game to start")
        while self.game.is_game_forming:
            await asyncio.sleep(2)

        # Playing game
        print(f"Started playing")

        self.dipcc_game = self.start_dipcc_game(power_name)
        print(f"Started dipcc game")

        self.player = Player(self.agent, power_name)

        while not self.game.is_game_done:
            self.phase_start_time = time.time()
            self.dipcc_current_phase = self.game.get_current_phase()

            # While agent is not eliminated
            if not self.game.powers[power_name].is_eliminated():
                logging.info(f"Press in {self.dipcc_current_phase}")
                # PRESS
                while not self.get_should_stop():
                    # if there is new message incoming
                    if self.has_state_changed(power_name):
                        # update press in dipcc
                        self.update_press_dipcc_game(power_name)
                        
                    # reply/gen new message
                    msg = self.generate_message(power_name)
                    # send message in dipcc and Mila
                    if msg is not None:
                        self.send_message(msg)
                    await asyncio.sleep(0.1)
        
                # ORDER
                print(f"Submit orders in {self.dipcc_current_phase}")
                agent_orders = self.player.get_orders(self.dipcc_game)

                # set order in Mila
                self.game.set_orders(power_name=power_name, orders=agent_orders, wait=False)
                
                # wait until the phase changed
                print(f"wait until {self.dipcc_current_phase} is done", end=" ")
                while not self.has_phase_changed():
                    print("", end=".")
                    await asyncio.sleep(2)
                
                # when the phase has changed, update submitted orders from Mila to dipcc
                if self.has_phase_changed():
                    self.phase_end_time = time.time()
                    self.update_and_process_dipcc_game()
                    self.init_phase()
                    print(f"Process to new phase")
        
        with open(gamedir / f"{power_name}_{game_id}_output.json", mode="w") as file:
            json.dump(
                to_saved_game_format(game), file, ensure_ascii=False, indent=2
            )
            file.write("\n")

    def init_phase(self):
        """     
        update new phase to the following Dict:
        """
        self.dipcc_current_phase = self.game.get_current_phase()
        self.prev_state = 0
        self.num_stop = 0
        self.last_successful_message_time = None

    def has_phase_changed(self)->bool:
        """ 
        check game phase 
        """
        return self.dipcc_current_phase != self.game.get_current_phase()

    def has_state_changed(self, power_name)->bool:
        """ 
        check dialogue state 
        """
        mila_phase = self.game.get_current_phase()

        phase_messages = self.get_messages(
            messages=self.game.messages, power=power_name
        )

        # update number of messages coming in
        phase_num_messages = len(phase_messages.values())
        self.dialogue_state[mila_phase] = phase_num_messages

        # check if previous state = current state
        has_state_changed = self.prev_state != self.dialogue_state[mila_phase]
        self.prev_state = self.dialogue_state[mila_phase]

        # print(f'current phase state: {self.dialogue_state}')

        return has_state_changed

    def get_should_stop(self)->bool:
        """ 
        stop when:
        1. close to deadline! (make sure that we have enough time to submit order)
        2. reuse stale pseudoorders
        """

        deadline = self.game.deadline
        if deadline ==0:
            deadline = 2*60
        close_to_deadline = deadline - 15
        no_message_second = 30

        assert close_to_deadline > 0, "Press period is less than zero"

        current_time = time.time()

        # PRESS allows in movement phase (ONLY)
        if not self.game.get_current_phase().endswith("M"):
            return True
        if self.has_phase_changed():
            return True
        if current_time - self.phase_start_time >= close_to_deadline:
            return True   
        if self.last_received_message_time != 0 and current_time - self.last_received_message_time >=no_message_second:
            print(f'no incoming message for {current_time - self.last_received_message_time} seconds')
            return True
        
        # check if reuse state psedo orders for too long
        if self.reuse_stale_pseudo():
            return True

        return False

    def update_press_dipcc_game(self, power_name: POWERS):
        """ 
        update new messages that present in Mila to dipcc
        """
        assert self.game.get_current_phase() == self.dipcc_current_phase, "Phase in two engines are not synchronized"

        mila_phase = self.game.get_current_phase()
        
        # new messages in current phase from Mila 
        phase_messages = self.get_messages(
            messages=self.game.messages, power=power_name
        )
        most_recent = self.last_received_message_time

        # update message in dipcc game
        for timesent, message in phase_messages.items():
            if timesent > self.last_received_message_time:

                dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)

                if timesent > most_recent:
                    most_recent = dipcc_timesent

                print(f'update a message from: {message.sender} to: {message.recipient} timesent: {timesent} and body: {message.message}')
                self.dipcc_game.add_message(
                    message.sender,
                    message.recipient,
                    message.message,
                    time_sent=dipcc_timesent,
                    increment_on_collision=True,
                )

        # update last_received_message_time 
        self.last_received_message_time = most_recent

    def update_and_process_dipcc_game(self):
        """     
        Inputs orders from the bygone phase into the dipcc game and process the dipcc game.
        """

        dipcc_game = self.dipcc_game
        mila_game = self.game
        dipcc_phase = dipcc_game.get_state()['name'] # short name for phase
        orders_from_prev_phase = mila_game.order_history[dipcc_phase] 
        
        # gathering orders from other powers from the phase that just ended
        for power, orders in orders_from_prev_phase.items():
            dipcc_game.set_orders(power, orders)

        # dipcc_game_state = dipcc_game.to_json()
        # print(f'dipcc game state: {dipcc_game_state}')

        dipcc_game.process() # processing the orders set and moving on to the next phase of the dipcc game

    def generate_message(self, power_name: POWERS)-> MessageDict:
        """     
        call CICERO to generate message (reference from generate_message_for_approval function - webdip_api.py)
        """
        
        # get last message timesent
        if self.last_successful_message_time == None:
            self.last_successful_message_time = Timestamp.now()
        
        # timestamp condition
        last_timestamp_this_phase = self.get_last_timestamp_this_phase(default=Timestamp.now())
        sleep_time = self.player.get_sleep_time(self.dipcc_game, recipient=None)
        wakeup_time = last_timestamp_this_phase + sleep_time

        sleep_time_for_conditioning = (
            sleep_time if sleep_time < INF_SLEEP_TIME else MESSAGE_DELAY_IF_SLEEP_INF
        )

        if get_last_message(self.dipcc_game) is None:
            timestamp_for_conditioning = sleep_time_for_conditioning
        else:
            timestamp_for_conditioning = last_timestamp_this_phase + sleep_time_for_conditioning

        # generate message using pseudo orders
        pseudo_orders = None
        # if isinstance(self.player.state, SearchBotAgentState):
        #     pseudo_orders = self.player.state.pseudo_orders_cache.maybe_get(
        #         self.dipcc_game, self.player.power, True, True, None
        #     ) 

        msg = self.player.generate_message(
            game=self.dipcc_game,
            timestamp=timestamp_for_conditioning,
            pseudo_orders=pseudo_orders,
        )

        self.last_successful_message_time = Timestamp.now()
        return msg

    def reuse_stale_pseudo(self):
        last_msg_time = self.last_successful_message_time
        if last_msg_time:
            delta = Timestamp.now() - last_msg_time
            logging.info(f"reuse_stale_pseudo: delta= {delta / 100:.2f} s")
            return delta > Timestamp.from_seconds(self.reuse_stale_pseudo_after_n_seconds)
        else:
            return False

    def get_last_timestamp_this_phase(
        self, default: Timestamp = Timestamp.from_seconds(0)
    ) -> Timestamp:
        """
        Looks for most recent message in this phase and returns its timestamp, returning default otherwise
        """
        all_timestamps = self.dipcc_game.messages.keys()
        return max(all_timestamps) if len(all_timestamps) > 0 else default

    def send_message(self, msg: MessageDict):
        """ 
        send message in dipcc and mila games 
        """ 
        timesend = Timestamp.now()
        self.dipcc_game.add_message(
                    msg['sender'], 
                    msg['recipient'], 
                    msg['message'], 
                    time_sent=timesend,
                    increment_on_collision=True,
                )

        mila_msg = Message(
            sender=msg["sender"],
            recipient=msg["recipient"],
            message=msg["message"],
            phase=self.game.get_current_phase(),
            )
        self.game.send_game_message(message=mila_msg)

        print(f'update a message from: { msg["sender"] } to: { msg["recipient"] } timesent: {timesend} and body: {msg["message"]}')

    def get_messages(
        self, 
        messages,
        power: POWERS,
        ):

        return {message.time_sent: message
                for message in messages.values()
                if message.recipient == power}

    def start_dipcc_game(self, power_name: POWERS) -> Game:

        deadline = self.game.deadline

        if deadline ==0:
            deadline = 2
        else:
            deadline = int(ceil(deadline/60))
        
        game = Game()

        game.set_scoring_system(Game.SCORING_SOS)
        game.set_metadata("phase_minutes", str(deadline))

        while game.get_state()['name'] != self.game.get_current_phase():
            self.update_past_phase(game,  game.get_state()['name'], power_name)

        return game
    
    def update_past_phase(self, dipcc_game: Game, phase: str, power_name: str):
        if phase not in self.game.message_history:
            dipcc_game.process()
            return

        phase_message = self.game.message_history[phase]
        for timesent, message in phase_message.items():
                dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)
                dipcc_game.add_message(
                    message.sender,
                    message.recipient,
                    message.message,
                    time_sent=dipcc_timesent,
                    increment_on_collision=True,
                )

        phase_order = self.game.order_history[phase] 

        for power, orders in phase_order.items():
            dipcc_game.set_orders(power, orders)
        
        dipcc_game.process()

            
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
    # parser.add_argument(
    #     "--agent",
    #     type=Path,
    #     # default ="/diplomacy_cicero/conf/common/agents/cicero.prototxt",
    #     default ="/diplomacy_cicero/conf/agents/bqre1p_parlai_20220819_cicero_2.prototxt",
    #     help="path to prototxt with agent's configurations (default: %(default)s)",
    # )
    parser.add_argument(
        "--outdir", type=Path, help="output directory for game json to be stored"
    )
    
    args = parser.parse_args()
    host: str = args.host
    port: int = args.port
    game_id: str = args.game_id
    power: str = args.power
    outdir: Optional[Path] = args.outdir

    print(f"settings:")
    print(f"host: {host}, port: {port}, game_id: {game_id}, power: {power}")

    if outdir is not None and not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    mila = milaWrapper()

    asyncio.run(
        mila.play_mila(
            hostname=host,
            port=port,
            game_id=game_id,
            power_name=power,
            gamedir=outdir,
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
    
