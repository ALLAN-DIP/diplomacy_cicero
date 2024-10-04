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

from stance_vector import ActionBasedStance, ScoreBasedStance
from cicero_stance import CiceroStance
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
    stance_vector = CiceroStance('AUSTRIA', mila, conflict_coef=1.0, conflict_support_coef=1.0, unrealized_coef = 0.0
                , discount_factor=1.0, random_betrayal=False)
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
        phase_orders = phase['orders']
        for power in powers:
            game.set_orders(power, phase_orders[power] if power in phase_orders else [])
        game.process()

    return game_data

def max_abs_scaling_global(phases):
    all_values = []

    # Collect all stance values from all phases
    for phase in phases:
        if 'stance_vectors' in phase:
            for stances in phase['stance_vectors'].values():
                all_values.extend(stances.values())

    # Find the global maximum absolute value
    global_max_abs = max(abs(value) for value in all_values)

    # Scale each stance value by the global maximum absolute value
    for phase in phases:
        if 'stance_vectors' in phase:
            for power, stances in phase['stance_vectors'].items():
                phase['stance_vectors'][power] = {other_power: value / global_max_abs if global_max_abs != 0 else 0
                                                  for other_power, value in stances.items()}


def update_game_with_stance_vectors(input_file, output_file):
    """Main function to load, modify, and save the game file."""
    # Load the game file
    game_data = load_game_file(input_file)
    
    # Add stance vectors to the game phases
    game_data = add_stance_vectors_to_phases(game_data)
    max_abs_scaling_global(game_data['phases'])
    
    # Save the updated game file
    save_game_file(output_file, game_data)

# Example usage:
for game_number in range(102527, 200000):
    input_file = f'/data/games/all_games/game_{game_number}.json'  # Replace with your input file path
    output_file = f'/data/games_stance/normalized_game_{game_number}.json'  # Replace with your desired output file path
    if not os.path.exists(input_file):
        continue
    if os.path.exists(output_file):
        continue
    update_game_with_stance_vectors(input_file, output_file)