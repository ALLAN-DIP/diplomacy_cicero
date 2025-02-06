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
    rec_prev_m_orders = last_m_phase.orders[recipient] if last_m_phase!=None and recipient in last_m_phase.orders else []
    sen_prev_m_orders = last_m_phase.orders[sender] if last_m_phase!=None and sender in last_m_phase.orders else []
    
    all_possible_orders = dipcc_game.get_all_possible_orders()
    rec_possible_orders = [order for unit in recipient_units for order in all_possible_orders[unit[2:]]]
    sen_possible_orders = [order for unit in sender_units for order in all_possible_orders[unit[2:]]]
    # print(f"sender and rec {sender} {recipient}")
    proposals = {sender: [], recipient: []}
    # print(f"extracted_moves {extracted_moves}")
    for move in extracted_moves:
        # print(f"move {move}")
        if 'propose-01' not in move['parsed-amr']:
            continue
        if move['polarity']:
            continue
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
                # if move has not enough info, just disregard it
                if in_possible_orders== 'not enough info':
                    continue
                if in_possible_orders:
                    #  let's count move
                    # A VIE S A TRI - VEN --- first assign to recipient
                    proposals[recipient].append(copy.deepcopy(move))

                # support/convoy rec's own unit or sender unit to do possible move
                in_sen_possible_orders = is_move_in_order_set(sen_possible_orders, new_move_info, sender)
                in_rec_possible_orders = is_move_in_order_set(rec_possible_orders, new_move_info, recipient)

                if in_sen_possible_orders== 'not enough info' and in_rec_possible_orders=='not enough info':
                    continue

                # if it is possible, let's count new_move to proposal!
                if in_sen_possible_orders == True or in_rec_possible_orders ==True:
                    # then assign A TRI - VEN --- to sender/recipient that A TRI belongs to
                    if new_move_info['country'] == sender:
                        proposals[sender].append(copy.deepcopy(new_move_info))
                    elif new_move_info['country'] == recipient:
                        proposals[recipient].append(copy.deepcopy(new_move_info))
                    elif  is_power_unit(sender_units, new_move_info['from']):
                        proposals[sender].append(copy.deepcopy(new_move_info))
                    elif is_power_unit(recipient_units, new_move_info['from']):
                        proposals[recipient].append(copy.deepcopy(new_move_info))
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
                    proposals[recipient].append(copy.deepcopy(move))

        
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
                # if move has not enough info, just disregard it
                if in_possible_orders== 'not enough info':
                    continue
                
                if in_possible_orders:
                    #  let's count move
                    # A VIE S A TRI - VEN --- first assign to sender
                    proposals[sender].append(copy.deepcopy(move))


                # support/convoy rec's own unit or sender unit
                in_sen_possible_orders = is_move_in_order_set(sen_possible_orders, new_move_info, sender)
                in_rec_possible_orders = is_move_in_order_set(rec_possible_orders, new_move_info, recipient)

                if in_sen_possible_orders== 'not enough info' and in_rec_possible_orders=='not enough info':
                    continue

                # if it is possible, then let's count new_move to proposal!
                if in_sen_possible_orders == True or in_rec_possible_orders ==True:
                    # then assign A TRI - VEN --- to sender/recipient that A TRI belongs to
                    if new_move_info['country'] == sender:
                        proposals[sender].append(copy.deepcopy(new_move_info))
                    elif new_move_info['country'] == recipient:
                        proposals[recipient].append(copy.deepcopy(new_move_info))
                    elif  is_power_unit(sender_units, new_move_info['from']):
                        proposals[sender].append(copy.deepcopy(new_move_info))
                    elif is_power_unit(recipient_units, new_move_info['from']):
                        proposals[recipient].append(copy.deepcopy(new_move_info))
            
            else: 
                # if sender promise to do a regular move
                in_units = False if 'to' not in move else is_prov_in_units(sender_units, move['to'])
        
                if in_units:
                    continue
                
                # if move is about previous m orders 
                in_prev_m_move = is_move_in_order_set(sen_prev_m_orders, move, recipient) and move['action'] =='-'
                # print(f'in_prev_m_move {in_prev_m_move} with {move}')
                # if move is even possible to do
                in_possible_orders = is_move_in_order_set(sen_possible_orders, move, sender)
                # print(f'in_possible_orders {in_possible_orders} with {move}')
                
                # if move has not enough info, just disregard it
                if in_possible_orders== 'not enough info':
                    # print('not enough info')
                    continue

                # if in previous m orders, also disregard it
                if in_prev_m_move:
                    # print('in prev m move')
                    continue

                # if it is possible, then let's count as proposal!
                if in_possible_orders:
                    proposals[sender].append(copy.deepcopy(move))
    
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
    dipcc_game.set_metadata("phase_minutes", str(10))
    
    # retrieving proposal orders by matching move in dict to any move in possible orders
    # matching reversing- if action move/hold -> 1. to 2. from , if support_action/convoy_action in dict 1. to 2. from 3. S/C 4. support_from/convoy_from
    our_power = power
    target_power = msg['sender']
    
    curr_state = dipcc_game.get_state()
    
    our_units = curr_state['units'][our_power]
    target_units = curr_state['units'][target_power]
    
    # we need proposal two ways - agreement of two parties
    if our_power != msg['recipient'] or (msg['proposals'][our_power] == [] and msg['proposals'][target_power] == []):
        logger.info(f"current power to advise ({our_power}) should be the same msg recipient {msg['recipient']}")
        logger.info(f"no proposals ! {msg['proposals']}")
        return False, {}
    
    logger.info(f"checking friction in proposal {msg['proposals']}...")
    our_power_dipcc_game = game_from_view_of(dipcc_game, our_power)
    
    bp_policy = cicero_player.agent.get_plausible_orders_policy(
        our_power_dipcc_game, agent_power=our_power, agent_state=cicero_player.state
    )
    
    proposal = msg['proposals']
    
    try_i = 0
    our_proposal_orders= dict()
    while len(our_proposal_orders.keys()) ==0 and try_i <=3:
        our_power_possible_orders = list(bp_policy[our_power].keys())
        if try_i > 0:
            our_power_possible_orders += fairdiplomacy.action_generation.get_all_possible_orders(our_power_dipcc_game, our_power, max_actions=100)
            
        # do RL part using those proposal from msg['extract_moves']
        for action in our_power_possible_orders:
            # print('possible orders')
            # print(action)
            not_found = [True for i in range (len(proposal[our_power]))]
            for i in range (len(proposal[our_power])):
                for unit_order in list(action):
                    # if province in unit order is not country in move's units then invalid
                    #make sure we support the right country units either support its own units or target's units
                    order_tokens = unit_order.split()
                    # A VIE S A TRI H
                    # A VIE S A TRI - VEN
                    if order_tokens[2] == 'S' or order_tokens[2] == 'C':
                        is_power = is_power_unit(our_units, order_tokens[4]) or is_power_unit(target_units, order_tokens[4])
                    else:
                        is_power = True

                    is_valid = is_move_in_order_set([unit_order], proposal[our_power][i])==True
                    if is_power and is_valid and is_valid != 'not enough info':
                        not_found[i] = False
                        
            if all(not x for x in not_found):         
                our_proposal_orders[action]=1.0
                logger.info(f'found align action! {action} to our power proposal {proposal[our_power]}')
            if len(our_proposal_orders.keys()) >= 5:
                try_i = 5
                break
        try_i+=1
        
    try_i = 0
    target_proposal_orders= dict()
    while len(target_proposal_orders.keys()) ==0 and try_i <=3:
        target_power_possible_orders = list(bp_policy[target_power].keys()) 
        if try_i > 0:
            target_power_possible_orders += fairdiplomacy.action_generation.get_all_possible_orders(our_power_dipcc_game, target_power, max_actions=100)
        
        for action in target_power_possible_orders:
            # print('possible orders')
            # print(action)
            not_found = [True for i in range (len(proposal[target_power]))]
            for i in range (len(proposal[target_power])):
                for unit_order in list(action):
                    # logger.info(f"comparing {unit_order} and {proposal[target_power][i]}")
                    order_tokens = unit_order.split()
                    # A VIE S A TRI H
                    # A VIE S A TRI - VEN
                    if order_tokens[2] == 'S' or order_tokens[2] == 'C':
                        is_power = is_power_unit(our_units, order_tokens[4]) or is_power_unit(target_units, order_tokens[4])
                    else:
                        is_power = True
                        
                    is_valid = is_move_in_order_set([unit_order], proposal[target_power][i])
                    if is_power and is_valid and is_valid != 'not enough info':
                        not_found[i] = False
                        
            if all(not x for x in not_found):        
                target_proposal_orders[action]=1.0
                logger.info(f'found align action! {action} to target power proposal {proposal[target_power]}')
            if len(target_proposal_orders.keys()) >= 5:
                try_i = 5
                break
        try_i+=1
    
    # we need proposal in Diplomacy orders
    if len(our_proposal_orders.keys()) ==0 and len(target_proposal_orders.keys()) ==0:
        logger.info(f"we need at least one proposal for our/target proposal orders in Diplomacy our: {our_proposal_orders} and target: {target_proposal_orders}")
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
    
    if len(our_proposal_orders.keys()) ==0:
        our_proposal_orders = our_policy
    elif len(target_proposal_orders.keys()) ==0:
        target_proposal_orders = target_policy
    
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
                    if (round(val_vic_D_lie_V_prime - val_vic_D_best_V_prime, 3) >= 0.001 and
                        round(val_dec_D_best_V_prime - val_dec_D_best_V_best, 3) >= 0.005 and
                        round(val_vic_D_lie_V_prime - val_vic_D_lie_V_best, 3) >= 0):
                        logger.info(f"Friction detected! passing all criteria:")
                        logger.info(f"too good to be true {val_vic_D_lie_V_prime - val_vic_D_best_V_prime} >= 0.001")
                        logger.info(f"deceiver receives better value when stab {val_dec_D_best_V_prime - val_dec_D_best_V_best} >= 0.005")
                        logger.info(f"a lie that increases victim's value {val_vic_D_lie_V_prime - val_vic_D_lie_V_best} >= 0")
                        return True, {'V_best': V_best, 'D_best': D_best, 'V_prime': V_prime, 'D_lie': D_lie, 'our_proposal_orders': our_proposal_orders, 'target_proposal_orders':target_proposal_orders}
                        # else:
                        #     logger.info(f"Friction not detected! not passing some criteria:")
                        #     logger.info(f"too good to be true {val_vic_D_lie_V_prime - val_vic_D_best_V_prime} > 0")
                        #     logger.info(f"deceiver receives better value when stab {val_dec_D_best_V_prime - val_dec_D_best_V_best} >= 0")
                        #     logger.info(f"a lie that increases victim's value {val_vic_D_lie_V_prime - val_vic_D_lie_V_best} >= 0")   
    return False, {}

def load_cicero():
    # agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero_llama.prototxt')
    agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
    agent = PyBQRE1PAgent(agent_config.bqre1p)
    return agent

def load_offline_game_and_save_extracted_moves(input_file, output_file):
    # agent = load_cicero()
    # cicero_player = Player(agent, our_power)
    
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    dipcc_game = Game()
    
    for phase in data['phases']:
        dipcc_game_year = int(dipcc_game.get_current_phase()[1:-1])
        phase_year = int(phase['name'][1:-1])
        while dipcc_game.get_current_phase() != phase['name'] and dipcc_game_year <= phase_year:
            print(f"catch up phase {dipcc_game.get_current_phase()} to game data {phase['name']}")
            dipcc_game.process()
            dipcc_game_year = int(dipcc_game.get_current_phase()[1:-1])
            
        if dipcc_game_year > phase_year:
            continue

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
           
            sender = msg['sender']
            recipient = msg['recipient']
            pair_power_str = '-'.join(sorted([sender, recipient]))
            msg = msg_to_move_dict(msg, prev_extracted_moves, prev_messages)
            prev_extracted_moves[pair_power_str] = copy.deepcopy(msg['extracted_moves'])
            prev_messages[pair_power_str] = copy.deepcopy(msg)
        
        for power, orders in phase['orders'].items():
            print(f'set order {power.upper()}\'s {orders}')
            dipcc_game.set_orders(power.upper(), orders)
        dipcc_game.process()
            
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4) 
    print(f'saved amr and extracted moves to {output_file}')

def load_offline_game_and_save_prediction_moves(input_file, output_file):
    agent = load_cicero()
    
    
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)


    for power in POWERS:
        dipcc_game = Game()
        dipcc_game.set_metadata("phase_minutes", str(3))
        view_dipcc_game = game_from_view_of(dipcc_game, power)
        cicero_player = Player(agent, power)
        
        for phase in data['phases']:
            # cache_predict_orders = {power: dict() for power in POWERS}
            if 'predicted_orders' not in phase:
                phase['predicted_orders'] = dict()

            while view_dipcc_game.get_current_phase() != phase['name']:
                print(f"catch up phase {view_dipcc_game.get_current_phase()} to game data {phase['name']}")
                view_dipcc_game.process()
            else:
                logger.info(phase['name'])
                
            if phase['name'][-1] == 'M':
                phase['predicted_orders'][power] = cicero_player.agent.get_orders(view_dipcc_game, power, cicero_player.state)
                logger.info(f"predict for {power} updating {phase['predicted_orders'][power]}")
            
            new_orders = dict()
            for power_order, orders in phase['orders'].items():
                logger.info(f'set order {power_order.upper()}\'s {orders}')
                new_orders[power_order.upper()] = copy.deepcopy(orders)
                view_dipcc_game.set_orders(power_order.upper(), orders)
            phase['orders'] = copy.deepcopy(new_orders)
            view_dipcc_game.process()
            
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4) 
    logger.info(f'saved predicted orders to {output_file}')

# def test_specific_message():    
#     game_file = '/denis/merged_game1.json'
#     target_phase = 'F1902M'
#     our_power = 'ENGLAND'
#     target_power = 'FRANCE'
#     target_message = "Germany's told me that he'll just support his centers"          
#     load_offline_game_and_test_utils(game_file, target_phase=target_phase, target_message=target_message, our_power= our_power,target_power=target_power)

def run_through_game(input_file, output_file):
    agent = load_cicero()
    
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    dipcc_game = Game()
    
    for phase in data['phases']:
        while dipcc_game.get_current_phase() != phase['name']:
            # print(f"catch up phase {dipcc_game.get_current_phase()} to game data {phase['name']}")
            dipcc_game.process()
        else:
            print(phase['name'])
            
        if phase['name'][-1] != 'M':
            continue

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
            
            sender = msg['sender']
            recipient = msg['recipient']
            pair_power_str = '-'.join(sorted([sender, recipient]))
            # msg = msg_to_move_dict(msg, prev_extracted_moves, prev_messages)
            assert 'extracted_moves' in msg, f"no extracted_moves in msg {msg}"
            logger.info(f"{sender}->{recipient} with msg {msg['message']} \nAMR {msg['parsed-amr']} \nextracted moves {msg['extracted_moves']}")
            cicero_player = Player(agent, recipient)
            is_friction, what_friction = is_friction_in_proposal(dipcc_game, cicero_player, msg, recipient)
            msg['friction'] = is_friction
            msg['friction_info'] = what_friction
            # if is_friction and not msg['sender_labels']:
            #     print(f"True positive")
            # elif is_friction and msg['sender_labels']:
            #     print(f"False positive")
            # elif not is_friction and not msg['sender_labels']:
            #     print(f"False negative")
            # else:
            #     print(f"True negative")
            prev_extracted_moves[pair_power_str] = copy.deepcopy(msg['extracted_moves'])
            prev_messages[pair_power_str] = copy.deepcopy(msg)
            

        for power, orders in phase['orders'].items():
            print(f'set order {power.upper()}\'s {orders}')
            dipcc_game.set_orders(power.upper(), orders)
        dipcc_game.process()
            
    # save tp fn fp (except tn too many)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4) 
    print(f'saved to {output_file}')
    
for i in range(1,13):
    game_file = f'/denis/merged_game{i}_proposals.json'
    new_file = f'/denis/proposal_only/merged_game{i}.json'
    if os.path.exists(new_file):
        print(f"The file {new_file} already exists.")
    else:
        run_through_game(game_file, new_file)

# for i in range(1,13):
#     game_file = f'/denis/merged_game{i}.json'
#     # new_file = f'/denis/merged_game{i}_proposals.json'
#     new_file = f'/denis/predict_orders_denis/game{i}_predict_orders.json'
#     if os.path.exists(new_file):
#         print(f"The file {new_file} already exists.")
#     else:
#         # load_offline_game_and_save_extracted_moves(game_file, new_file)
#         load_offline_game_and_save_prediction_moves(game_file, new_file)
        
        