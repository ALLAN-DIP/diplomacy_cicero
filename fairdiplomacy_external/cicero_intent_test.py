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
    BilateralConditionalValueTable,
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
from fairdiplomacy.pseudo_orders import PseudoOrders

from conf.agents_pb2 import *
import google.protobuf.message
import heyhi

import sys
import argparse
import asyncio
import json as json
import sys
import time
import math
from pathlib import Path

from diplomacy import connect
from diplomacy import Message
from diplomacy.client.network_game import NetworkGame
from diplomacy.utils.export import to_saved_game_format, from_saved_game_format
from diplomacy.utils import strings
from daide2eng.utils import gen_English, create_daide_grammar, is_daide
from stance_vector import ActionBasedStance, ScoreBasedStance

default_is_bot ={
            'AUSTRIA': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'ENGLAND': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'FRANCE': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'GERMANY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'ITALY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'RUSSIA': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'TURKEY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False}
        }
default_stance ={
            'AUSTRIA': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'ENGLAND': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'FRANCE': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'GERMANY': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'ITALY': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'RUSSIA': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'TURKEY': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0}
        }
default_order_log ={00000:'default'}  
default_deceiving = {
            'AUSTRIA': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'ENGLAND': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'FRANCE': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'GERMANY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'ITALY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'RUSSIA': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'TURKEY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False}
        }

MESSAGE_DELAY_IF_SLEEP_INF = Timestamp.from_seconds(60)
DEFAULT_DEADLINE = 5
GAME_PATH = "./fairdiplomacy_external/out/test_deceptive_cicero_7.json"
YEAR = 'S1905M'

def update_past_phase(mila_game, dipcc_game: Game, phase: str, power: Power):
    if phase not in mila_game.message_history:
        dipcc_game.process()
        return

    phase_message = mila_game.message_history[phase]
    for timesent, message in phase_message.items():
        dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)

        if message.recipient != power or message.sender != power:
            continue
        # print(f'update message: {message}')

        dipcc_game.add_message(
                message.sender,
                message.recipient,
                message.message,
                time_sent=dipcc_timesent,
                increment_on_collision=True,
            )

    phase_order = mila_game.order_history[phase] 

    for power, orders in phase_order.items():
        dipcc_game.set_orders(power, orders)
    
    dipcc_game.process()

def get_last_timestamp_this_phase(dipcc_game: Game, default: Timestamp = Timestamp.from_seconds(0)) -> Timestamp:
    """
    Looks for most recent message in this phase and returns its timestamp, returning default otherwise
    """
    all_timestamps = dipcc_game.messages.keys()
    return max(all_timestamps) if len(all_timestamps) > 0 else default

def generate_message(
    dipcc_game: Game, 
    player: Player, 
    recipient: Power = None,
    pseudo_orders: PseudoOrders = None
    )-> MessageDict:
    """     
    call CICERO to generate message (reference from generate_message_for_approval function - webdip_api.py)
    """
    
    # timestamp condition
    last_timestamp_this_phase = get_last_timestamp_this_phase(dipcc_game, default=Timestamp.now())
    sleep_time = player.get_sleep_time(dipcc_game, recipient=recipient)

    sleep_time_for_conditioning = (
        sleep_time if sleep_time < INF_SLEEP_TIME else MESSAGE_DELAY_IF_SLEEP_INF
    )

    if get_last_message(dipcc_game) is None:
        timestamp_for_conditioning = sleep_time_for_conditioning
    else:
        timestamp_for_conditioning = last_timestamp_this_phase + sleep_time_for_conditioning

    # generate message
    msg = player.generate_message(
        game=dipcc_game,
        timestamp=timestamp_for_conditioning,
        recipient=recipient,
        pseudo_orders=pseudo_orders,
    )
    return msg


def load_game(stop_at_phase: str, power: Power):
    #load game from mila json
    f = open(GAME_PATH)
    saved_game = json.load(f)
    mila_game = from_saved_game_format(saved_game)
    f.close()

    game = Game()

    game.set_scoring_system(Game.SCORING_SOS)
    game.set_metadata("phase_minutes", str(DEFAULT_DEADLINE))
    #avoid having info. leak!
    game = game_from_view_of(game, power)

    #load each phase to mila and dipcc
    #stop at phase
    stance_vector = ActionBasedStance(power ,mila_game)
    # print(f'start phase mila: {mila_game.get_current_phase()}')
    # print(f'start phase dipcc: {game.get_state()["name"]}')
    while game.get_state()['name'] != mila_game.get_current_phase() and game.get_state()['name'] !=stop_at_phase:
        # print(f'updating phase: {game.get_state()["name"]}')
        update_past_phase(mila_game, game, game.get_state()['name'], power)
    # print(f'let\'s play at phase: {game.get_state()["name"]}')

    #quick fix for error in diplomacy/engine/game.py
    for phase in mila_game.order_history:
        if phase not in mila_game.order_log_history:
            mila_game.order_log_history.put(mila_game._phase_wrapper_type(phase), default_order_log)
        if phase  not in mila_game.is_bot_history:
            mila_game.is_bot_history.put(mila_game._phase_wrapper_type(phase), default_is_bot)
        if phase  not in mila_game.deceiving_history:
            mila_game.deceiving_history.put(mila_game._phase_wrapper_type(phase), default_deceiving)
    # print(len(list(mila_game.order_history.keys())))
    # print(len(list(mila_game.order_log_history.keys())))
    # print(len(list(mila_game.is_bot_history.keys())))
    # print(len(list(mila_game.deceiving_history.keys())))

    #get stance vector
    first_turn = True if mila_game.get_current_phase()=='S1901M' else False
    if not first_turn:
        # update stance vector 
        _, _ = stance_vector.get_stance(mila_game, verbose=True)
    

    
    return mila_game, game, stance_vector

def load_cicero():
    agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
    agent = PyBQRE1PAgent(agent_config.bqre1p)
    return agent


async def test_intent_from_game(sender: Power, recipient: Power):
    K=20
    #load agent
    for i in range(K):
        mila_game, dipcc_game, stance_vector = load_game(YEAR, sender)
        if i ==0:
            agent = load_cicero()

        sender_player = Player(agent, sender)
        sender_player.agent.set_stance_vector(stance_vector)
        sender_player.agent.set_mila_game(mila_game)

        dipcc_game.add_message(
                    recipient, 
                    sender, 
                    "What's the plan this turn?", 
                    time_sent=Timestamp.now(),
                    increment_on_collision=True,
                )

        pre_orders = sender_player.get_orders(dipcc_game)
        print(f'-------- sample {i} --------')
        print(f'{sender} (main) first order of this turn w/o communication {pre_orders}')

        # get msg from cicero and log intent 
        # for i in range(K):
        #     msg = generate_message(dipcc_game, cicero_player, recipient=recipient)
        #     real_pseudo = cicero_player.state.pseudo_orders_cache.maybe_get(
        #                 dipcc_game, cicero_player.power, True, True, recipient) 
        #     print(f'-------- sample {i} --------')
        #     # print(f'PO to comm log: {real_pseudo}')
        #     print(f'with message {msg}')

        # change intent -> new K intents (by self) and get msg 2... K
        #ours
        #get from Searchbot for the best lie intent if possible? #edit in searchbot too
        # test_intent1 = PseudoOrders({'S1902M': {'GERMANY': ('F DEN - NTH', 'A HOL H', 'A RUH - MUN', 'F KIE - HEL', 'A MUN - TYR'), 'FRANCE': ('F BRE - ENG', 'A BUR - BEL', 'A PAR - PIC', 'F POR - MAO', 'A SPA - GAS')}, 
        #         'S1902R': {'GERMANY': (), 'FRANCE': ()}, 'F1902M': {'GERMANY': ('A HOL H', 'A MUN - TYR', 'F HEL S F DEN - NTH', 'F DEN - NTH', 'A TYR - PIE'), 'FRANCE': ('A BEL S A HOL', 'F BRE - ENG', 'F MAO S F BRE - ENG', 'A PIC S A BEL', 'A GAS - BUR')}})
        # print(f'fake PO1 to comm log: {test_intent1}')
        
        msg1 = generate_message(dipcc_game, sender_player, recipient=recipient, pseudo_orders=None)
        if msg1 == None:
            print(f'msg is none, we are skipping this {i} sample')
            continue
        
        
        print(f'with message {msg1}')

        # #theirs

        # #future

        # #generate outcomes
        # msg = generate_message(dipcc_game, cicero_player, recipient=recipient, pseudo_orders=test_intent1)

        rec_mila_game, rec_dipcc_game, stance_vector = load_game(YEAR, recipient)
        rec_player = Player(agent, recipient)
        rec_player.agent.set_stance_vector(stance_vector)
        rec_player.agent.set_mila_game(mila_game)
        # rec_pre_orders = rec_player.get_orders(dipcc_game)
        bp_policy = rec_player.agent.maybe_get_incremental_bp(
            rec_dipcc_game,
            agent_power=recipient,
            agent_state=rec_player.state,
            extra_plausible_orders=extra_plausible_orders,
        )
        print(f'{recipient} (recipient) first order of this turn w/o communication {bp_policy}')
        rec_dipcc_game.add_message(
                    recipient, 
                    sender, 
                    "What's the plan this turn?", 
                    time_sent=Timestamp.now(),
                    increment_on_collision=True,
                )
        rec_dipcc_game.add_message(
                    msg1['sender'], 
                    msg1['recipient'], 
                    msg1['message'], 
                    time_sent=Timestamp.now(),
                    increment_on_collision=True,
                )
        # later_rec_pre_orders = rec_player.agent.get_orders(rec_dipcc_game)
        bp_policy = rec_player.agent.maybe_get_incremental_bp(
            rec_dipcc_game,
            agent_power=recipient,
            agent_state=rec_player.state,
            extra_plausible_orders=extra_plausible_orders,
        )
        #or get their bp_policies to see diff in prob
        print(f'{recipient} (recipient) order of this turn after communication with {sender}: {bp_policy}')
        print(f'-------- end --------')
        await asyncio.sleep(5)
# def lie_score():

def test_val_table(power: Power, recipient: Power):
    #ref: https://github.com/facebookresearch/diplomacy_cicero/blob/main/fairdiplomacy/agents/br_corr_bilateral_search.py#L358
    mila_game, dipcc_game = load_game(YEAR, power)
    cicero_player = load_cicero(power)
    search_result = cicero_player.agent.run_best_response_against_correlated_bilateral_search(game=dipcc_game, agent_power=power,agent_state=cicero_player.state)

    print(f'search result at S1902M for {power} save to file')
    print(f'{search_result.power_value_matrices[recipient]}')
    new_table_dict = {}
    for key, values in search_result.power_value_matrices[recipient].items():
        key1, key2 = key
        if str(key1) not in new_table_dict:
            new_table_dict[str(key1)] = dict()
        new_table_dict[str(key1)][str(key2)] = values.numpy().tolist()

    out_file = open(f'./fairdiplomacy_external/out/value_table_{YEAR}_{power[:3]}_{recipient[:3]}.json', "w") 
    json.dump(new_table_dict, out_file, indent = 6) 
    out_file.close() 

asyncio.run(test_intent_from_game('RUSSIA','TURKEY'))
# test_val_table('GERMANY','FRANCE')