from abc import ABC
import argparse
import asyncio
import copy
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import List, Optional, Sequence

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
import chiron_utils.game_utils
from chiron_utils.utils import return_logger
from conf.agents_pb2 import *
from diplomacy import connect
from diplomacy.client.network_game import NetworkGame
from diplomacy.utils import strings
from diplomacy.utils.constants import SuggestionType
from diplomacy.utils.export import to_saved_game_format

from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent as PyBQRE1PAgent
from fairdiplomacy.agents.player import Player
from fairdiplomacy.data.build_dataset import (DATASET_DRAW_MESSAGE, DATASET_NODRAW_MESSAGE, DRAW_VOTE_TOKEN,
                                              UNDRAW_VOTE_TOKEN)
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import (
    MessageDict,
    Timestamp,
)
from fairdiplomacy.utils.game import game_from_view_of
from fairdiplomacy.utils.sampling import normalize_p_dict, sample_p_dict
from fairdiplomacy.utils.typedefs import get_last_message
import heyhi
from heyhi import setup_logging
from parlai_diplomacy.wrappers.classifiers import INF_SLEEP_TIME
from fairdiplomacy_external.amr_utils.amr_moves import parse_single_message_to_amr
from fairdiplomacy_external.amr_utils.amr_to_dict import amr_single_message_to_dict, is_move_in_order_set, is_prov_in_units, is_power_unit, get_move_in_order_set
import fairdiplomacy.action_generation


logger = return_logger(__name__)

MESSAGE_DELAY_IF_SLEEP_INF = Timestamp.from_seconds(60)

DEFAULT_DEADLINE = 5

POWER_TO_INDEX = {power: index for index, power in enumerate(POWERS)}


def get_proposal_move_dict(dipcc_game, msg):
    """mainly checking move dicts whether they are proposal to do such move in the current turn.
    The code is partly from deception/persuasion detection in our prior paper (https://aclanthology.org/2024.acl-long.672.pdf)
    """
    # check if extracted_moves relate to sender or recipient
    # copy from deception detection
    extracted_moves = msg['extracted_moves']
    curr_phase = dipcc_game.get_current_phase()
    curr_state = dipcc_game.get_state()
    sender = msg['sender']
    recipient = msg['recipient']
    
    sender_units = curr_state['units'][sender]
    recipient_units = curr_state['units'][recipient]
    
    last_m_phase = next(
                (phase for phase in reversed(dipcc_game.get_phase_history()) if phase.name.endswith("M")),None)
    rec_prev_m_orders = last_m_phase.orders[recipient] if last_m_phase!=None else []
    sen_prev_m_orders = last_m_phase.orders[sender] if last_m_phase!=None else []
    
    all_possible_orders = dipcc_game.get_all_possible_orders()
    rec_possible_orders = [order for unit in recipient_units for order in all_possible_orders[unit[2:]]]
    sen_possible_orders = [order for unit in sender_units for order in all_possible_orders[unit[2:]]]
    
    proposals = {sender: [], recipient: []}
    for move in extracted_moves:
        if 'action' not in move or ('from' not in move and 'to' not in move):
            continue

        country =  move.get('support_country', move.get('transport_country', move.get('country', '')))

        if country not in [sender, recipient]:
            continue

        if 'year' in move and move['year']!=curr_phase[1:-1]:
            continue

        # it should be about sender/recipient 
        if move['concept'] in ['bounce-03','demilitarize-01'] and (move['country'] != recipient or move['country'] != sender):
            continue

        #get unit's power from move dict
        unit_loc_key = ''
        to_key = ''
        if 'support_from' in move:
            unit_loc_key = move['support_from']
        elif 'transport_from' in move:
            unit_loc_key = move['transport_from']
        elif 'from' in move:
            unit_loc_key = move['from']

        possible = True

        power_units = curr_state['units'][recipient]
        # unit is in recipient's unit and move country is for recipient
        possible = is_power_unit(power_units, unit_loc_key) and country == recipient

        #if move is about recipient:
        if possible:
            
            if  'support_action' in move or'transport_action' in move:
                # if recipient is asked to support/convoy any units
                new_move_info = dict()
                for key, value in move.items():
                    if key in ['unit', 'from','to','action','country']:
                        new_move_info[key] = value
                for key in ['concept','variable','polarity']:
                    new_move_info[key] = ''
                
                # whole support/convoy is possible
                in_possible_orders = is_move_in_order_set(rec_possible_orders, move, recipient)
                if in_possible_orders:
                    #  let's count move
                    # A VIE S A TRI - VEN --- first assign to recipient
                    proposals[recipient].append(move)

                # support/convoy rec's own unit or sender unit to do possible move
                in_sen_possible_orders = is_move_in_order_set(sen_possible_orders, new_move_info, sender)
                in_rec_possible_orders = is_move_in_order_set(rec_possible_orders, new_move_info, recipient)

                if in_sen_possible_orders== 'not enough info' and in_rec_possible_orders=='not enough info':
                    continue

                # if it is possible, let's count new_move to proposal!
                if in_sen_possible_orders == True or in_rec_possible_orders ==True:
                    # then assign A TRI - VEN --- to sender/recipient that A TRI belongs to
                    if  is_power_unit(sender_units, new_move_info['from']):
                        proposals[sender].append(new_move_info)
                    elif is_power_unit(recipient_units, new_move_info['from']):
                        proposals[recipient].append(new_move_info)
            else:
                # if recipient is asked to do any move
                in_units = False if 'to' not in move else is_prov_in_units(power_units, move['to'])
            
                if in_units:
                    continue
                
                # if move is about previous m orders 
                in_prev_m_move = is_move_in_order_set(rec_prev_m_orders, move, recipient) and move['action'] =='-'
                # if move is even possible to do
                in_possible_orders = is_move_in_order_set(rec_possible_orders, move, recipient)

                # if move has not enough info, just disregard it
                if in_possible_orders== 'not enough info':
                    continue

                # if in previous m orders, also disregard it
                if in_prev_m_move:
                    continue

                # if it is possible, then let's count as proposal!
                if in_possible_orders:
                    proposals[recipient].append(move)

        
        else:
            #if move is about sender doing something
            
            #if it is not sender and not recipient then we not considering it in deception detection (it's hard to tell if it is lie about the third country or the sender is being lied to)
            if not is_power_unit(sender_units, unit_loc_key):
                continue
            
            if  'support_action' in move or'transport_action' in move:
                # if sender promise to support/convoy recipient units
                new_move_info = dict()
                for key, value in move.items():
                    if key in ['unit', 'from','to','action','country']:
                        new_move_info[key] = value
                for key in ['concept','variable','polarity']:
                    new_move_info[key] = ''
                
                # whole support/convoy is possible
                in_possible_orders = is_move_in_order_set(sen_possible_orders, move, sender)
                if in_possible_orders:
                    #  let's count move
                    # A VIE S A TRI - VEN --- first assign to sender
                    proposals[sender].append(move)


                # support/convoy rec's own unit or sender unit
                in_sen_possible_orders = is_move_in_order_set(sen_possible_orders, new_move_info, sender)
                in_rec_possible_orders = is_move_in_order_set(rec_possible_orders, new_move_info, recipient)

                if in_sen_possible_orders== 'not enough info' and in_rec_possible_orders=='not enough info':
                    continue

                # if it is possible, then let's count new_move to proposal!
                if in_sen_possible_orders == True or in_rec_possible_orders ==True:
                    # then assign A TRI - VEN --- to sender/recipient that A TRI belongs to
                    if  is_power_unit(sender_units, new_move_info['from']):
                        proposals[sender].append(new_move_info)
                    elif is_power_unit(recipient_units, new_move_info['from']):
                        proposals[recipient].append(new_move_info)
            
            else: 
                # if sender promise to do a regular move
                in_units = False if 'to' not in move else is_prov_in_units(sender_units, move['to'])
        
                if in_units:
                    continue
                
                # if move is about previous m orders 
                in_prev_m_move = is_move_in_order_set(sen_prev_m_orders, move, recipient) and move['action'] =='-'
                # if move is even possible to do
                in_possible_orders = is_move_in_order_set(sen_possible_orders, move, sender)

                # if move has not enough info, just disregard it
                if in_possible_orders== 'not enough info':
                    continue

                # if in previous m orders, also disregard it
                if in_prev_m_move:
                    continue

                # if it is possible, then let's count as proposal!
                if in_possible_orders:
                    proposals[sender].append(move)
    
    msg['proposals'] = proposals
    logger.info(f"get proposals from extracted_moves {msg['extracted_moves']}")
    logger.info(f"proposals: {msg['proposals']}")
    return msg

def msg_to_move_dict(msg, prev_extracted_moves, prev_messages):
    """
    a function to parse NL to AMR then capture AMR to extract moves
    """
    msg = msg_to_AMR(msg)
    msg = AMR_to_move_dict(msg, prev_extracted_moves, prev_messages)
    return msg

def msg_to_AMR(msg):
    msg = parse_single_message_to_amr(msg)
    logger.info(f"parsing msg to AMR {msg['message']}")
    # logger.info(f"{msg['parsed-amr']}")
    return msg

def AMR_to_move_dict(msg, prev_extracted_moves, prev_messages):
    msg = amr_single_message_to_dict(msg, prev_extracted_moves, prev_messages)
    logger.info(f"parsing AMR to move dict {msg['parsed-amr']}")
    logger.info(f"{msg['extracted_moves']}")
    return msg

def is_friction_in_proposal(dipcc_game, cicero_player, msg, power):
    # arrange move_dict_list into proposal (function to filter proposal?)
    msg = get_proposal_move_dict(dipcc_game, msg)
    
    # retrieving proposal orders by matching move in dict to any move in possible orders
    # matching reversing- if action move/hold -> 1. to 2. from , if support_action/convoy_action in dict 1. to 2. from 3. S/C 4. support_from/convoy_from
    our_power = power
    target_power = msg['sender']
    
    # we need proposal two ways - agreement of two parties
    if our_power != msg['recipient'] or msg['proposals'][our_power] == [] or msg['proposals'][target_power] == []:
        logger.info(f"current power to advise ({our_power}) should be the same msg recipient (msg['recipient'])")
        logger.info(f"we need proposal two ways {msg['proposals']}")
        return False, {}
    
    logger.info(f"checking friction in proposal {msg['proposals']}...")
    our_power_dipcc_game = game_from_view_of(dipcc_game, our_power)
    
    bp_policy = cicero_player.agent.get_plausible_orders_policy(
        our_power_dipcc_game, agent_power=our_power, agent_state=cicero_player.state
    )
    
    proposal = msg['proposals']
    
    try_i = 0
    our_proposal_orders= dict()
    our_proposal_unit_orders = dict()
    while len(our_proposal_orders.keys()) ==0 and try_i <=3:
        our_power_possible_orders = fairdiplomacy.action_generation.get_all_possible_orders(our_power_dipcc_game, our_power, max_actions=300)
        # do RL part using those proposal from msg['extract_moves']
        for action in our_power_possible_orders:
            # print('possible orders')
            # print(action)
            not_found = [True for i in range (len(proposal[our_power]))]
            for i in range (len(proposal[our_power])):
                for unit_order in list(action):
                    is_valid = is_move_in_order_set([unit_order], proposal[our_power][i])
                    if is_valid and is_valid != 'not enough info':
                        not_found[i] = False
                        
            if all(not x for x in not_found):         
                our_proposal_orders[action]=0.2
                logger.info(f'found align action! {action} to our power proposal {proposal[our_power][i]}')
        try_i+=1
        
    try_i = 0
    target_proposal_orders= dict()
    while len(target_proposal_orders.keys()) ==0 and try_i <=3:
        target_power_possible_orders = fairdiplomacy.action_generation.get_all_possible_orders(our_power_dipcc_game, target_power, max_actions=300)
        
        for action in target_power_possible_orders:
            # print('possible orders')
            # print(action)
            not_found = [True for i in range (len(proposal[target_power]))]
            for i in range (len(proposal[target_power])):
                for unit_order in list(action):
                    # logger.info(f"comparing {unit_order} and {proposal[target_power][i]}")
                    is_valid = is_move_in_order_set([unit_order], proposal[target_power][i])
                    if is_valid and is_valid != 'not enough info':
                        not_found[i] = False
                        
            if all(not x for x in not_found):        
                target_proposal_orders[action]=0.2
                logger.info(f'found align action! {action} to target power proposal {proposal[target_power][i]}')
        try_i+=1
    
    # we need proposal in Diplomacy orders
    if len(our_proposal_orders.keys()) ==0 or len(target_proposal_orders.keys()) ==0:
        logger.info(f"we need proposal in Diplomacy orders our: {our_proposal_orders} and target: {target_proposal_orders}")
        return False, {}
    
    for order, prob in our_proposal_orders.items():
        bp_policy[our_power][order] = prob
    for order, prob in target_proposal_orders.items():
        bp_policy[target_power][order] = prob
        
    search_result = cicero_player.agent.run_best_response_against_correlated_bilateral_search(game=our_power_dipcc_game, agent_power=our_power,bp_policy=bp_policy, agent_state=cicero_player.state)
    our_policy = search_result.get_agent_policy()[our_power]
    # V_best = max(our_policy.items(), key=lambda item: item[1])[0]
    target_policy = search_result.get_agent_policy()[target_power]
    # D_best = max(target_policy.items(), key=lambda item: item[1])[0]
    
    joint_table = search_result.power_value_matrices[target_power]
    
    for our_action in list(our_proposal_orders.keys()):
        for target_action in list(target_proposal_orders.keys()):
            for V_best in list(our_policy.keys()):
                for D_best in list(target_policy.keys()):
                    if our_action == V_best and target_action == D_best:
                        continue
                    V_prime = our_action
                    D_lie = target_action
                    # val_to_vic(D_lie, V_prime)
                    val_vic_D_lie_V_prime = joint_table[V_prime, D_lie].squeeze().tolist()[POWER_TO_INDEX[our_power]]
                    # val_to_vic(D_best, V_prime)
                    val_vic_D_best_V_prime = joint_table[V_prime, D_best].squeeze().tolist()[POWER_TO_INDEX[our_power]]
                    # val_to_vic(D_lie, V_best)
                    val_vic_D_lie_V_best = joint_table[V_best, D_lie].squeeze().tolist()[POWER_TO_INDEX[our_power]]
                    # val_to_dec(D_best, V_prime)
                    val_dec_D_best_V_prime = joint_table[V_prime, D_best].squeeze().tolist()[POWER_TO_INDEX[target_power]]
                    # val_to_dec(D_best, V_best)
                    val_dec_D_best_V_best = joint_table[V_best, D_best].squeeze().tolist()[POWER_TO_INDEX[target_power]]
                    if (val_vic_D_lie_V_prime - val_vic_D_best_V_prime > 0 and
                        val_dec_D_best_V_prime - val_dec_D_best_V_best >= 0 and
                        val_vic_D_lie_V_prime - val_vic_D_lie_V_best >=0):
                        logger.info(f"Friction detected! passing all criteria:")
                        logger.info(f"too good to be true {val_vic_D_lie_V_prime - val_vic_D_best_V_prime} > 0")
                        logger.info(f"deceiver receives better value when stab {val_dec_D_best_V_prime - val_dec_D_best_V_best} >= 0")
                        logger.info(f"a lie that increases victim's value {val_vic_D_lie_V_prime - val_vic_D_lie_V_best} >= 0")
                        return True, {'V_best': V_best, 'D_best': D_best, 'V_prime': V_prime, 'D_lie': D_lie, 'proposals': proposal}
                    # else:
                    #     logger.info(f"Friction not detected! not passing some criteria:")
                    #     logger.info(f"too good to be true {val_vic_D_lie_V_prime - val_vic_D_best_V_prime} > 0")
                    #     logger.info(f"deceiver receives better value when stab {val_dec_D_best_V_prime - val_dec_D_best_V_best} >= 0")
                    #     logger.info(f"a lie that increases victim's value {val_vic_D_lie_V_prime - val_vic_D_lie_V_best} >= 0")   
    return False, {}

def load_cicero(power):
    agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
    agent = PyBQRE1PAgent(agent_config.bqre1p)
    cicero_player = Player(agent, power)
    return cicero_player

def load_offline_game_and_test_utils(input_file, target_phase='S1901M', target_message='None', our_power='AUSTRIA',target_power='AUSTRIA'):
    cicero_player = load_cicero(our_power)
    
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    dipcc_game = Game()
    
    found_msg = None
    for phase in data['phases']:
        while dipcc_game.get_current_phase() != phase['name']:
            print(f"catch up phase {dipcc_game.get_current_phase()} to game data {phase['name']}")
            dipcc_game.process()
        else:
            print(phase['name'])
        prev_messages = {
            '-'.join(sorted([power1, power2])): dict()
            for power1 in POWERS for power2 in POWERS if power1 != power2
            }
        prev_extracted_moves = {
            '-'.join(sorted([power1, power2])): []
            for power1 in POWERS for power2 in POWERS if power1 != power2
        }
        for msg in phase['messages']:
            dipcc_game.add_message(
                        msg['sender'],
                        msg['recipient'],
                        msg['message'],
                        time_sent=Timestamp.now(),
                        increment_on_collision=True,
                    )
            if phase['name'] == target_phase:
                sender = msg['sender']
                recipient = msg['recipient']
                pair_power_str = '-'.join(sorted([sender, recipient]))
                msg = msg_to_move_dict(msg, prev_extracted_moves, prev_messages)
                prev_extracted_moves[pair_power_str] = copy.deepcopy(msg['extracted_moves'])
                prev_messages[pair_power_str] = copy.deepcopy(msg)
            
            if phase['name'] == target_phase and target_message in msg['message']:
                print("FOUND")
                found_msg = msg
                break
        if found_msg:
            break
        else:
            for power, orders in phase['moves'].items():
                print(f'set order {power}\'s {orders}')
                dipcc_game.set_orders(power, orders)
            dipcc_game.process()
            
    if found_msg:   
        msg_tuple = {'sender': found_msg['sender'], 'recipient': found_msg['recipient'], 'message': found_msg['message']}
        msg_tuple = msg_to_move_dict(msg_tuple, prev_extracted_moves, prev_messages)
        is_friction, what_friction = is_friction_in_proposal(dipcc_game, cicero_player, found_msg, our_power)
        if is_friction:
            print(f"Friction detected! {what_friction}")

def test():    
    game_file = '/denis/game1.json'
    target_phase = 'F1902M'
    our_power = 'ENGLAND'
    target_power = 'FRANCE'
    target_message = "Germany's told me that he'll just support his centers"          
    load_offline_game_and_test_utils(game_file, target_phase=target_phase, target_message=target_message, our_power= our_power,target_power=target_power)

test()