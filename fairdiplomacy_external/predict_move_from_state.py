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
from fairdiplomacy.utils.order_idxs import is_action_valid
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

# from stance_vector import ActionBasedStance, ScoreBasedStance
# from cicero_stance import CiceroStance
from diplomacy import Game as milaGame


import json
    

def load_game_file(input_file):
    """Load the JSON game file."""
    with open(input_file, 'r') as f:
        return json.load(f)

def save_game_file(output_file, game_data):
    """Save the modified game data to a new file."""
    with open(output_file, 'w') as f:
        json.dump(game_data, f, indent=4)

def generate_stance_vector(game, stance_vector):
    """
    Generate a stance vector dictionary.
    Nested dictionary structure: {power1: {power2: vector_value}}.
    This is just a mockup. Replace with your actual stance prediction logic.
    """
    stance_vector.is_rollout = True
    stance_vector.rollout_dipcc_game = game
    mila =milaGame()
    new_stance = stance_vector.get_stance(mila)
    return new_stance

def add_stance_vectors_to_phases(game_data):
    """Add stance vectors to each phase of the game."""
    
    from diplomacy import Game as milaGame
    powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
    mila = milaGame()
    stance_vector = CiceroStance('AUSTRIA' ,
                                mila,
                                invasion_coef = 0.1,
                                conflict_coef = 0.05,
                                invasive_support_coef = 0.1,
                                conflict_support_coef = 0.05,
                                friendly_coef = 0.1,
                                unrealized_coef = 0.05,
                discount_factor=0.9, random_betrayal=False)
    game = Game()
    
    for phase in game_data['phases']:
        # print(f'phase: {phase["name"]}')
        if phase["name"]!= 'S1901M':
            # get stance from prev m orders
            stance_vectors = generate_stance_vector(game, stance_vector)
            # Add the stance vector to the phase
        else:
            stance_vectors = stance_vector.stance
        phase['stance_vectors'] = stance_vectors
        # print(f'stance: {stance_vectors}')
        # print('==============================================')    
        
        if game.is_game_done:
            break
        
        # set order and process
        # denis dataset has 'moves' instead of 'orders'
        phase_orders = phase['moves']
        for power in powers:
            game.set_orders(power, phase_orders[power] if power in phase_orders else [])
        game.process()

    return game_data

def max_abs_scaling_global(phases):
    def clip_value(value):
        if value < -1.0:
            return -1.0
        elif value > 1.0:
            return 1.0
        else:
            return round(value, 2)
    # all_values = []

    # Collect all stance values from all phases
    # for phase in phases:
    #     if 'stance_vectors' in phase:
    #         for stances in phase['stance_vectors'].values():
    #             all_values.extend(stances.values())

    # # Find the global maximum absolute value
    # global_max_abs = max(abs(value) for value in all_values)

    # Scale each stance value by the global maximum absolute value
    for phase in phases:
        if 'stance_vectors' in phase:
            for power, stances in phase['stance_vectors'].items():
                phase['stance_vectors'][power] = {other_power: clip_value(value) for other_power, value in stances.items()}


def update_game_with_stance_vectors(input_file, output_file):
    """Main function to load, modify, and save the game file."""
    # Load the game file
    game_data = load_game_file(input_file)
    
    # Add stance vectors to the game phases
    game_data = add_stance_vectors_to_phases(game_data)
    max_abs_scaling_global(game_data['phases'])
    
    # Save the updated game file
    save_game_file(output_file, game_data)
    
def load_cicero():
    agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
    agent = PyBQRE1PAgent(agent_config.bqre1p)
    return agent
    
def load_state_and_predict_move(input_file, output_file):
    # work with Sadra (input = one phase data)
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    dipcc_game = Game()
    dipcc_json = json.loads(dipcc_game.to_json())
    # print(data)
    dipcc_json['phases'][0]['state']['units'] = data['state']['units']
    dipcc_json['phases'][0]['state']['homes'] = data['state']['homes']
    dipcc_json['phases'][0]['state']['centers'] = data['state']['centers']
    dipcc_game = Game.from_json(json.dumps(dipcc_json))
    
    data["moves"] = {power : {} for power in POWERS}
    cicero = load_cicero()
    for power in POWERS:
        cicero_player = Player(cicero, power)
        cicero_policy = cicero_player.agent.get_plausible_orders_policy(game=dipcc_game, agent_power=power, agent_state=cicero_player.state)
        data["moves"][power] = [[list(tuple_order), prob] for tuple_order, prob in cicero_policy[power].items()]
    # print(data["moves"])

    # Save the updated file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        
import os

def get_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

def load_state_sadra_data(input_directory):
    json_files_list = get_json_files(input_directory)
    # Example usage:
    for input_file in json_files_list:
        output_file = f"{os.path.splitext(input_file)[0]}_moves.json"  # Replace with your desired output file path
        if not os.path.exists(input_file):
            continue
        if os.path.exists(output_file):
            continue
        # update_game_with_stance_vectors(input_file, output_file)
        load_state_and_predict_move(input_file, output_file)
        print(f'Finished processing {input_file}.')

def load_whole_game_and_predict_move(input_file, target_phase='S1901M', target_message='None', our_power='AUSTRIA',target_power='AUSTRIA', stance_power_orders=None, proposal=None):
    import fairdiplomacy.action_generation
    # work with Sadra (input = one phase data)
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    dipcc_game = Game()
    
    found = False
    for phase in data['phases']:
        while dipcc_game.get_current_phase() != phase['name']:
            print(f"catch up phase {dipcc_game.get_current_phase()} to game data {phase['name']}")
            dipcc_game.process()
        else:
            print(phase['name'])
        for msg in phase['messages']:
            dipcc_game.add_message(
                        msg['sender'],
                        msg['recipient'],
                        msg['message'],
                        time_sent=Timestamp.now(),
                        increment_on_collision=True,
                    )
            if phase['name'] == target_phase and target_message in msg['message']:
                print("FOUND")
                found = True
                break
        if found:
            break
        else:
            for power, orders in phase['moves'].items():
                print(f'set order {power}\'s {orders}')
                dipcc_game.set_orders(power, orders)
            dipcc_game.process()
        
    dipcc_game = game_from_view_of(dipcc_game, our_power)
    agent = load_cicero()
    power=our_power
    cicero_player = Player(agent, power)
    bp_policy = cicero_player.agent.get_plausible_orders_policy(
                dipcc_game, agent_power=power, agent_state=cicero_player.state
            )
    for order, prob in stance_power_orders.items():
        bp_policy[target_power][order] = prob
    
    our_power_possible_orders = fairdiplomacy.action_generation.get_all_possible_orders(dipcc_game, our_power, max_actions=100)
    proposal_orders= dict()
    for action in our_power_possible_orders:
        # print('possible orders')
        # print(action)
        for unit_order in list(action):
            if proposal in unit_order:
                proposal_orders[action]=0.2
                print(f'found align action! {action}')
    
    for order, prob in proposal_orders.items():
        bp_policy[our_power][order] = prob
    # stance_orders[target_power] = [stance_power_orders] # order that stance controlled SL return for target power
    # PlausibleOrders = Dict[Power, List[Action]]
    # search_result = cicero_player.agent.run_best_response_against_correlated_bilateral_search(game=dipcc_game, agent_power=power, agent_state=cicero_player.state)
    search_result = cicero_player.agent.run_best_response_against_correlated_bilateral_search(game=dipcc_game, agent_power=power,bp_policy=bp_policy, agent_state=cicero_player.state)
    print(f'expect to see stance-controlled order {stance_power_orders} in the following policy')
    print(f'cicero {our_power}\'s policy: {search_result.get_agent_policy()[our_power]}')
    print(f'cicero {target_power}\'s policy: {search_result.get_agent_policy()[target_power]}')
    our_policy = search_result.get_agent_policy()[our_power]
    our_best_action = max(our_policy.items(), key=lambda item: item[1])[0]
    target_policy = search_result.get_agent_policy()[target_power]
    target_best_action = max(target_policy.items(), key=lambda item: item[1])[0]
    
    joint_table = search_result.power_value_matrices[target_power]
    for our_action in list(proposal_orders.keys()) + [our_best_action]:
        for target_action in list(stance_power_orders.keys()) + [target_best_action]:
            print(f'cicero bilateral condition ({our_action}, {target_action}) : {joint_table[our_action,target_action]}')

# load_state_sadra_data('/diplomacy_cicero/data/lr5')
game_file = '/diplomacy_cicero/data/denis/game1.json'
target_phase = 'F1902M'
our_power = 'ENGLAND'
target_power = 'FRANCE'
stance_power_orders = {tuple(['F ENG - MAO', 'A BEL H', 'F MAR H','F MAO - POR', 'A BUR S A GAS - MAR', 'A GAS - MAR']): 0.2, tuple(['F ENG - MAO', 'A BEL H','F MAR H', 'F MAO - POR', 'A BUR - MAR', 'A GAS - SPA']): 0.2}
proposal = 'A YOR - HOL'
target_message = "Germany's told me that he'll just support his centers"
load_whole_game_and_predict_move(game_file, target_phase=target_phase, target_message=target_message, our_power= our_power,target_power=target_power, stance_power_orders=stance_power_orders,proposal=proposal)