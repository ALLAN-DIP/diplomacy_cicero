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

from diplomacy import connect
from diplomacy import Message
from diplomacy import Game as milaGame
from diplomacy.client.network_game import NetworkGame
from diplomacy.utils.export import to_saved_game_format, from_saved_game_format
from diplomacy.utils import strings
from daide2eng.utils import gen_English, create_daide_grammar, is_daide
from stance_vector import ActionBasedStance, ScoreBasedStance
from cicero_stance import CiceroStance

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
YEAR = 'F1902M'

def update_past_phase(mila_game, dipcc_game: Game, phase: str, power: Power):
    if phase in mila_game.message_history:
        phase_message = mila_game.message_history[phase]
        
        for timesent, message in phase_message.items():
            dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)
            if message.recipient != power and message.sender != power:
                continue
            
            print(f'update message: {message}')

            dipcc_game.add_message(
                    message.sender,
                    message.recipient,
                    message.message,
                    time_sent=dipcc_timesent,
                    increment_on_collision=True,
                )
    if phase in mila_game.order_history:
        phase_order = mila_game.order_history[phase] 
        
        print(phase_order)
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


def load_game(gamepath, stop_at_phase: str, power: Power):
    #load game from mila json
    f = open(gamepath)
    saved_game = json.load(f)
    mila_game = from_saved_game_format(saved_game)
    f.close()

    game = Game()

    game.set_scoring_system(Game.SCORING_SOS)
    game.set_metadata("phase_minutes", str(DEFAULT_DEADLINE))
    #avoid having info. leak!
    # game = game_from_view_of(game, power)

    #quick fix for error in diplomacy/engine/game.py
    for phase in mila_game.order_history:
        if phase not in mila_game.order_log_history:
            mila_game.order_log_history.put(mila_game._phase_wrapper_type(phase), default_order_log)
        if phase  not in mila_game.is_bot_history:
            mila_game.is_bot_history.put(mila_game._phase_wrapper_type(phase), default_is_bot)
        if phase  not in mila_game.deceiving_history:
            mila_game.deceiving_history.put(mila_game._phase_wrapper_type(phase), default_deceiving)

    #load each phase to mila and dipcc
    #stop at phase
    stance_vector = start_stance(mila_game, stop_at_phase, power)
    # print(f'start phase mila: {mila_game.get_current_phase()}')
    # print(f'start phase dipcc: {game.get_state()["name"]}')
    while game.get_state()['name'] != mila_game.get_current_phase() and game.get_state()['name'] !=stop_at_phase:
        print(f'updating phase: {game.get_state()["name"]}')
        update_past_phase(mila_game, game, game.get_state()['name'], power)
    print(f'let\'s play at phase: {game.get_state()["name"]}')
    
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
    
def start_stance(game, stop_phase, power_name):
    stance_vector = CiceroStance(power_name ,game,conflict_coef=1.0, conflict_support_coef=1.0, unrealized_coef = 0.0
                    , discount_factor=1.0, random_betrayal=False)
    
    # let's recalculate stance vector from begining if this is not the first m turn
    if game.get_current_phase() != 'S1901M':
        for state in game.state_history.values():
            phase = state['name']
            if phase == stop_phase:
                break
            if phase[-1] == 'M':
                #set for prev_m_phase to specific mphase data
                stance_vector.set_mphase(phase)
                curr_stance, stance_log = stance_vector.get_stance(game, verbose=True)
        print(f'most updated stance {phase}: {stance_log[power_name]}')
                
    stance_vector.set_mphase(None)  
    return stance_vector

def load_prev_messages(extracted_pair_file, dipcc_game, year, cutoff_timesent):
    # messages is a list of message that needed to load
    f = open(extracted_pair_file)
    extracted_pair = json.load(f)
    list_msg = []
    
    for pair, pair_conv in extracted_pair[year].items():
        for msg in pair_conv:
            if msg['time_sent'] > cutoff_timesent:
                break
            list_msg.append(msg)
            dipcc_game.add_message(
                msg['sender'], 
                msg['recipient'], 
                msg['message'], 
                time_sent=Timestamp.now(),
                increment_on_collision=True,
            )
    return list_msg

def load_mila_game(mila_game, stop_phase):
    game = milaGame()
    for phase in mila_game.order_history:
        if phase == stop_phase:
            break
        phase_order = mila_game.order_history[phase] 
        for power, orders in phase_order.items():
            game.set_orders(power_name=power, orders=orders)
        
        game.process()
    logging.info(f'load mila game to phase {game.get_current_phase()}')
    return game

def test_lie_message(game_id, gamepath, convpath, year, cutoff, sender: Power, recipient: Power):
    extracted_game_file = convpath
    K=3
    list_msg = []
    date_str = datetime.today().strftime('%Y-%m-%d')
    # load agent
    # new_mila_game to feed in agents
    logging.basicConfig(filename=f'./fairdiplomacy_external/lie/{date_str}_{game_id}_possible_lie_{year}_{sender}_{cutoff}.log', format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

    # logging.info(f'=================== test_lie sample {i} ==================')
    mila_game, dipcc_game, stance_vector = load_game(gamepath, year, sender)
    new_mila_game = load_mila_game(mila_game, year)
    new_mila_game.role = sender
    agent = load_cicero()

    sender_player = Player(agent, sender)
    sender_player.state.set_stance_vector(stance_vector)
    sender_player.state.set_mila_game(new_mila_game, game_id)
    sender_player.agent.set_mila_game(new_mila_game, game_id)
    
    sender_player.agent.local_game = True
    
    if cutoff>0:
        list_msg = load_prev_messages(extracted_game_file, dipcc_game, year, cutoff)
    dipcc_game = game_from_view_of(dipcc_game, sender)
    for i in range(K):
        msg1 = generate_message(dipcc_game, sender_player, recipient=recipient, pseudo_orders=None)
        
        if msg1 == None:
            print(f'msg is none, we are skipping this {i} sample')
            logging.info(f'Lie or Deception None')
            continue
        
        dipcc_game.add_message(
                msg1['sender'], 
                msg1['recipient'], 
                msg1['message'], 
                time_sent=Timestamp.now(),
                increment_on_collision=True,
            )
        logging.info(f'sample {i}: Lie or Deception  {msg1["sender"]} -> {msg1["recipient"]}: {msg1["message"]}')
        # msg1['test_lie'] = True
        # list_msg.append(msg1)       
        # out_file = open(f"{save_path}/{game_id}_possible_lie_{year}_{cutoff}.json", "w") 
        # json.dump(list_msg, out_file, indent = 6) 
        # out_file.close() 

def test_st_value_table(game_id, gamepath, year, sender: Power, recipient: Power):
    import pathlib
    import heyhi
    import numpy as np
    import sys
    from fairdiplomacy.utils.game import game_from_two_party_view
    from diplomacy import Game, GamePhaseData
    from fairdiplomacy_external.cicero_stance import CiceroStance
    from datetime import datetime, timedelta
    from fairdiplomacy.agents.br_corr_bilateral_search import (
        extract_bp_policy_for_powers,
        compute_weights_for_opponent_joint_actions,
        compute_best_action_against_reweighted_opponent_joint_actions,
        sample_joint_actions,
        filter_invalid_actions_from_policy,
        rescore_bp_from_bilateral_views,
        compute_payoff_matrix_for_all_opponents,
        BRCorrBilateralSearchResult,
    )

    # logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)
    date_str = datetime.today().strftime('%Y-%m-%d')
    logging.basicConfig(filename=f'./fairdiplomacy_external/lie/{date_str}_{game_id}_test_stance_br_foes.log', format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

    np.random.seed(0)  # type:ignore
    torch.manual_seed(0)

    mila_game, game, stance_vector = load_game(gamepath, year, sender)
    # cfg = heyhi.load_config(
    #     pathlib.Path(__file__).resolve().parents[2]
    #     / "conf/common/agents/bqre1p_20210723.prototxt",
    #     overrides=["bqre1p.base_searchbot_cfg.n_rollouts=64"] + sys.argv[1:],
    # )
    agent_power = sender
    recipient = recipient
    
    cfg = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
    print(cfg.bqre1p)
    agent = PyBQRE1PAgent(cfg.bqre1p)
    agent_state = agent.initialize_state(power=agent_power)
    
    stance_vector = CiceroStance(agent_power ,mila_game,conflict_coef=1.0, conflict_support_coef=1.0, unrealized_coef = 0.0
                , discount_factor=1.0, random_betrayal=False)
    logging.info(f'stance vector before game starting : {stance_vector.stance}')
    
    stance_vector.is_rollout = True
    
    agent_state.set_stance_vector(stance_vector)
    agent_state.set_opponent(recipient)
    
    # print(agent.get_orders(game, power="AUSTRIA", state=agent.initialize_state(power="AUSTRIA")))
    
    logging.info('------ test stance vector in rollout and results ------')
    game_2p = game_from_two_party_view(game, agent_power, recipient, add_message_to_all=False)
    # base_policy = agent.maybe_get_incremental_bp(
    #                 game_2p, agent_power=agent_power, agent_state=agent_state,
    #                 )
    # if base_policy is None:
    #     base_policy = agent.order_sampler.sample_orders(
    #         game_2p,
    #         agent_power=agent_power,
    #         speaking_power=agent_power,
    #         player_rating=agent.player_rating if agent.set_player_rating else None
    #     )
    # base_policy = extract_bp_policy_for_powers(base_policy, [agent_power, recipient])
    
    # recipient_value_table = compute_payoff_matrix_for_all_opponents(
    #                     game_2p,
    #                     agent.all_power_base_strategy_model_executor,
    #                     base_policy,
    #                     agent_power,
    #                     agent.br_corr_bilateral_search_cfg.bilateral_search_num_cond_sample,
    #                     agent.has_press,
    #                     None,
    #                     None,
    #                     )[recipient]
    # logging.info('------ recipient_value_table ------')
    # logging.info(f'{recipient_value_table}')
    
    # ally_recipient_value_table = compute_payoff_matrix_for_all_opponents(
    #                     game_2p,
    #                     agent.all_power_base_strategy_model_executor,
    #                     base_policy,
    #                     agent_power,
    #                     agent.br_corr_bilateral_search_cfg.bilateral_search_num_cond_sample,
    #                     agent.has_press,
    #                     None,
    #                     None,
    #                     agent_state = agent_state,
    #                     stance_vector_mode = 'ally',
    #                 )[recipient]
    
    # logging.info('------ ally - recipient_value_table ------')
    # logging.info(f'{ally_recipient_value_table}')
    
    # foes_recipient_value_table = compute_payoff_matrix_for_all_opponents(
    #                 game_2p,
    #                 agent.all_power_base_strategy_model_executor,
    #                 base_policy,
    #                 agent_power,
    #                 agent.br_corr_bilateral_search_cfg.bilateral_search_num_cond_sample,
    #                 agent.has_press,
    #                 None,
    #                 None,
    #                 agent_state = agent_state,
    #                 stance_vector_mode = 'foes',
    #             )[recipient]
    
    # logging.info('------ foes - recipient_value_table ------')
    # logging.info(f'{foes_recipient_value_table}')
    
    model_pseudo_orders = agent.message_handler.model_pseudo_orders
    old_args = model_pseudo_orders.set_generation_args("greedy")
    pseudo_orders = model_pseudo_orders.produce_joint_action_bilateral(
            game, agent_power, recipient=recipient
        )
    extra_plausible_orders = {
            pwr: ([a] if is_action_valid(game, pwr, a) else []) for pwr, a in pseudo_orders.items()
        }

    bp_policy = agent.order_sampler.sample_orders(
        game,
        agent_power=agent_power,
        speaking_power=agent_power,
        player_rating=agent.player_rating if agent.set_player_rating else None,
        extra_plausible_orders=extra_plausible_orders,
    )
    
#     deceptive_search_result = agent.run_best_response_against_correlated_bilateral_search(
#     game,
#     agent_state=agent_state,
#     bp_policy=bp_policy,
#     agent_power=agent_power,
# )
#     logging.info('------ BR deceptive / persuasion for those moves ------')
#     logging.info(f'value_to_me: {deceptive_search_result.value_to_me}')
#     logging.info(f'sample action: {deceptive_search_result.sample_action(agent_power)}')
    
    
    foes_deceptive_search_result = agent.run_best_response_against_correlated_bilateral_search(
        game,
        agent_state=agent_state,
        bp_policy=bp_policy,
        agent_power=agent_power,
        stance_vector_mode = 'foes',
    )
    
    logging.info('------ BR deceptive / persuasion for those moves IF assuming that rec is foes! ------')
    logging.info(f'value_to_me: {foes_deceptive_search_result.value_to_me}')
    logging.info(f'sample action: {foes_deceptive_search_result.sample_action(agent_power)}')
    

def test_stab_message(game_id, game_path, stab_path, save_path, year, sender, recipient):
    K=3 # rounds
    M=5 # number of interactions (from each side)
    #load sender agent
    agent = load_cicero()
    sender_player = Player(agent, sender)
    sender_player.agent.set_stance_vector(sen_stance_vector)
    
    #load recipient agent
    agent = load_cicero()
    recipient_player = Player(agent, recipient)
    recipient_player.agent.set_stance_vector(rec_stance_vector)
    
    for i in range(K):
        # load game,phase that we want to test (from file)
        mila_game, dipcc_game, sen_stance_vector = load_game(game_path, year, sender)
        _, _, rec_stance_vector = load_game(game_path, year, recipient)
        
        sender_player.agent.set_mila_game(mila_game,game_id)
        recipient_player.agent.set_mila_game(mila_game, game_id)
        
        # pair conversation to see if one actually stab (if possible)
        for j in range(M):
            dipcc_game = game_from_view_of(dipcc_game, sender)
            msg1 = generate_message(dipcc_game, sender_player, recipient=recipient, pseudo_orders=None)
            if msg1 != None:
                send_message(msg = msg1, engine='dipcc', dipcc_game = dipcc_game)
                send_message(msg = msg1, engine='mila', mila_game = mila_game)
            
            dipcc_game = game_from_view_of(dipcc_game, recipient)
            msg2 = generate_message(dipcc_game, recipient_player, recipient=sender, pseudo_orders=None)
            if msg2 != None:
                send_message(msg = msg2, engine='dipcc', dipcc_game = dipcc_game)
                send_message(msg = msg2, engine='mila', mila_game = mila_game)
                
        # submit order and save game file
        sender_orders = sender_player.get_orders(dipcc_game)
        recipient_orders = recipient_player.get_orders(dipcc_game)
        
        mila_game.set_orders(power_name=sender, orders=sender_orders, wait=False)
        mila_game.set_orders(power_name=recipient, orders=recipient_orders, wait=False)
        
        mila_game.process()
        
        with open(f"{save_path}/{game_id}_stab_{K}.json", mode="w") as file:
            json.dump(
                to_saved_game_format(mila_game), file, ensure_ascii=False, indent=4
            )
            file.write("\n")

def detect_stab(game_path, save_path):
    #load game from mila json
    f = open(game_path)
    saved_game = json.load(f)
    mila_game = from_saved_game_format(saved_game)
    mila_game.role = strings.OMNISCIENT_TYPE
    f.close()
    
    # init stance and detect update stance in each m season
    stance_vector = CiceroStance('AUSTRIA' ,mila_game,conflict_coef=1.0, conflict_support_coef=1.0, unrealized_coef = 0.0
                , discount_factor=1.0, random_betrayal=False)
    print('read game done')
    #for each phase
    #if ally->enemy and not lie then let's stab!
    for phase in mila_game.order_history: 
        #quick fix for error in diplomacy/engine/game.py
        if phase not in mila_game.order_log_history:
            mila_game.order_log_history.put(mila_game._phase_wrapper_type(phase), default_order_log)
        if phase  not in mila_game.is_bot_history:
            mila_game.is_bot_history.put(mila_game._phase_wrapper_type(phase), default_is_bot)
        if phase  not in mila_game.deceiving_history:
            mila_game.deceiving_history.put(mila_game._phase_wrapper_type(phase), default_deceiving)
            
    for state in mila_game.state_history.values():
        phase = state['name']
        print(phase)
        if phase!= 'S1901M' and phase[-1] == 'M' and phase != 'COMPLETED':
            #set for prev_m_phase to specific mphase data
            stance_vector.set_mphase(phase)
            prev_stance = stance_vector.stance.copy()
            curr_stance, stance_log = stance_vector.get_stance(mila_game, verbose=True)
            # print(curr_stance)
            power_orders = mila_game.order_log_history[mila_game._phase_wrapper_type(phase)]
            stab_by_pair(prev_stance, curr_stance, phase, power_orders, save_path)

    stance_vector.set_mphase(None)  

def stab_by_pair(prev_stance, curr_stance, phase, power_orders, save_path):
    list_stab = []
    for power, st_to_powers in prev_stance.items():
        print(f'------{power}------')
        # if there is a change in prev - curr >= 1.0, then this is a stab! save up year and pair
        for other_power, prev_st in st_to_powers.items():
            if prev_st - curr_stance[power][other_power] >= 1.0 and prev_st >= 0.0 :
                print(f'stab by {other_power} ')
                stab = {'phase': phase, 'stabber': other_power, 'stabbee': power, 'stabber_move': power_orders[other_power]}
                list_stab.append(stab.copy())
    append_to_json_file(list_stab, save_path)

def append_to_json_file(data, filename):
    try:
        # Read existing data from the file
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist yet, create an empty list
        existing_data = []
    
    # Append new data to the existing list
    existing_data.extend(data)
    
    # Write the combined data back to the file
    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4)
        
def send_message(msg: MessageDict, engine: str, dipcc_game=None, mila_game=None):
        """ 
        send message in dipcc and mila games 
        """ 
        
        if engine =='dipcc':
            timesend = Timestamp.now()
            dipcc_game.add_message(
                        msg['sender'], 
                        msg['recipient'], 
                        msg['message'], 
                        time_sent=timesend,
                        increment_on_collision=True,
                    )

        if engine =='mila':
            mila_msg = Message(
                sender=msg["sender"],
                recipient=msg["recipient"],
                message=msg["message"],
                phase=mila_game.get_current_phase(),
                )
            mila_game.send_game_message(message=mila_msg)
            timesend = mila_msg.time_sent

        print(f'update a message in {engine}, {msg["sender"] }->{ msg["recipient"]}: {msg["message"]}')
        
POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
# PHASES = ['S1902M', 'F1902M', 'S1903M', 'F1903M'] 
PHASES = ['S1903M', 'F1903M', 'S1904M', 'F1904M', 'S1905M', 'F1905M']       
# PHASES = [ 'S1905M', 'F1905M']      
# asyncio.run(test_intent_from_game('RUSSIA','TURKEY'))
# test_val_table('GERMANY','FRANCE')
# test two situations
# bad lie -> better lie
# possible stab -> lie + persuade with our method!
gameid = 'AIGame_21'
game_path = f'./fairdiplomacy_external/human_games/{gameid}.json'
extract_path = f'./fairdiplomacy_external/human_games/extracted_moves/{gameid}.json'
save_path = './fairdiplomacy_external/stab/'

# for phase in PHASES:
#     for power in POWERS:
#         test_lie_message(gameid, game_path, extract_path, phase, -1, power, None)
# test_lie_message(gameid, game_path, extract_path, 'F1901M', -1, 'AUSTRIA', 'TURKEY')
test_st_value_table(gameid, game_path, 'F1904M', 'AUSTRIA', 'TURKEY')
# detect_stab('./fairdiplomacy_external/human_games/AIGame_21.json', './fairdiplomacy_external/stab/AIGame_21.json')
# test_stab_message(game_id, game_path, stab_path, save_path)