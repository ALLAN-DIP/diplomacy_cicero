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

sys.path.insert(0, '/diplomacy_cicero/fairdiplomacy/AMR/DAIDE/DiplomacyAMR/code')
from amrtodaide import AMR
sys.path.insert(0, '/diplomacy_cicero/fairdiplomacy/AMR/penman')
# import penman
import regex
sys.path.insert(0, '/diplomacy_cicero/fairdiplomacy/AMR/amrlib')
from amrlib.models.parse_xfm.inference import Inference

MESSAGE_DELAY_IF_SLEEP_INF = Timestamp.from_seconds(60)
DEFAULT_DEADLINE = 5
GAME_PATH = "./fairdiplomacy_external/out/AIGame_0.json"
YEAR = 'S1901M'

num_beams   = 4
batch_size  = 16

device = 'cuda:0'
# model_dir  = './drive/MyDrive/cicero_experiments/AMR/amrlib/amrlib/data/model_parse_xfm/checkpoint-9920/'
model_dir  = '/diplomacy_cicero/fairdiplomacy/AMR/personal/SEN_REC_MODEL/'
inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)

def eng_to_amr(english,sender,recipient,inference):
    print('---------------------------')
    # gen_graphs = inference.parse_sents([english], disable_progress=False)
    gen_graphs = inference.parse_sents(['SEN'+' send to '+'REC'+' that '+english.replace(sender,'SEN').replace(recipient,'REC')], disable_progress=False)
    for graph in gen_graphs:
        graph = graph.replace('SEN',sender).replace('REC',recipient)
        amr = AMR()
        amr_node, s, error_list, snt_id, snt, amr_s = amr.string_to_amr(graph)
        if amr_node:
            amr.root = amr_node
        try:
            amr_s2 = amr.amr_to_string()
            return amr_s2
        except RecursionError:
            return '(a / amr-empty)'

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
    game = game_from_view_of(game, power)

    #load each phase to mila and dipcc
    #stop at phase
    print(f'start phase mila: {mila_game.get_current_phase()}')
    print(f'start phase dipcc: {game.get_state()["name"]}')
    while game.get_state()['name'] != mila_game.get_current_phase() and game.get_state()['name'] !=stop_at_phase:
        print(f'updating phase: {game.get_state()["name"]}')
        update_past_phase(mila_game, game, game.get_state()['name'], power)

    print(f'let\'s play at phase: {game.get_state()["name"]}')
    #avoid having info. leak!
    
    return mila_game, game

def load_cicero(power_name: Power):
    agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
    agent = PyBQRE1PAgent(agent_config.bqre1p)
    player = Player(agent, power_name)
    return player

def test_intent_from_game(sender: Power, recipient: Power):
    K=20
    #load agent
    mila_game, dipcc_game = load_game(YEAR, sender)
    cicero_player = load_cicero(sender)

    pre_orders = cicero_player.get_orders(dipcc_game)
    print(f'first order of this turn w/o communication {pre_orders}')

    for i in range(K):
        msg1 = generate_message(dipcc_game, cicero_player, recipient=recipient, pseudo_orders=None)
        print(f'-------- sample {i} --------')
        
        print(f'with message {msg1}')

def test_reply_from_game(sender: Power, recipient: Power, test_list):
    log = []
    for test in test_list:
        test_message = test
        K=10
        #load agent
        mila_game, dipcc_game = load_game(YEAR, sender)
        cicero_player = load_cicero(sender)
        dipcc_game.add_message(
                    recipient, 
                    sender, 
                    test_message, 
                    time_sent=Timestamp.now(),
                    increment_on_collision=True,
                )
        print(f'------ test_message: {test} ------')
        log_test = {'test_message': test, 'reply': []}
        for i in range(K):
            msg = generate_message(dipcc_game, cicero_player, recipient=recipient, pseudo_orders=None)
            if msg == None:
                print(f'msg is none, we are skipping this {i} sample')
                continue
            try:
                daide_string = eng_to_amr(msg['message'],msg['sender'],msg['recipient'],inference)
            except:
                daide_string = '(a / amr-empty)'
            log_test['reply'].append({'message':msg['message'], 'daide':daide_string})
        
        log.append(log_test)
        # print(f'sample {i}: message: {msg["message"]}, DAIDE: {daide_string}')
    out_file = open(f'./fairdiplomacy_external/out/log_reply_test.json', "w") 
    json.dump(log, out_file, indent = 4) 
    out_file.close() 

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

# test_intent_from_game('GERMANY','FRANCE')
# test_val_table('GERMANY','FRANCE')

# test format of proposal.
test_list = []
test_list += ["Please move VIE to GAL."]
test_list += ["Please move VIE to GAL. Reply with Yes if you agree or No if you won't or haven't decided."]
test_list += ["Please move VIE to GAL. Say agree if you are fine or say no if you won't or haven't decided."]
test_list += ["Please move VIE to GAL. Reply with Yes or No only"]
test_list += ["Can move army in VIE to GAL?"]
test_list += ["Can move army in VIE to GAL? Reply with Yes if you agree or No if you won't or haven't decided."]
test_list += ["Can move army in VIE to GAL? Say agree if you are fine or say no if you won't or haven't decided."]
test_list += ["Can move army in VIE to GAL? Reply with Yes or No only"]
test_list += ["Will you move VIE to GAL?"]
test_list += ["Will you move VIE to GAL? Reply with Yes if you agree or No if you won't or haven't decided."]
test_list += ["Will you move VIE to GAL? Say agree if you are fine or say no if you won't or haven't decided."]
test_list += ["Will you move VIE to GAL? Reply with Yes or No only"]
test_reply_from_game('AUSTRIA','ITALY', test_list)