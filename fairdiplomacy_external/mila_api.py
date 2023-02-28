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

from daidepp.utils import pre_process, gen_English, post_process, is_daide

MESSAGE_DELAY_IF_SLEEP_INF = Timestamp.from_seconds(60)
ProtoMessage = google.protobuf.message.Message

DEFAULT_DEADLINE = 5

import json
import sys
sys.path.insert(0, '/diplomacy_cicero/fairdiplomacy/AMR/DAIDE/DiplomacyAMR/code')
from amrtodaide import AMR
sys.path.insert(0, '/diplomacy_cicero/fairdiplomacy/AMR/penman')
# import penman
import regex
sys.path.insert(0, '/diplomacy_cicero/fairdiplomacy/AMR/amrlib')
from amrlib.models.parse_xfm.inference import Inference

power_dict = {'ENGLAND':'ENG','FRANCE':'FRA','GERMANY':'GER','ITALY':'ITA','AUSTRIA':'AUS','RUSSIA':'RUS','TURKEY':'TUR'}
af_dict = {'A':'AMY','F':'FLT'}

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
        self.sent_FCT = {'RUSSIA':set(),'TURKEY':set(),'ITALY':set(),'ENGLAND':set(),'FRANCE':set(),'GERMANY':set(),'AUSTRIA':set()}
        self.sent_PRP = {'RUSSIA':set(),'TURKEY':set(),'ITALY':set(),'ENGLAND':set(),'FRANCE':set(),'GERMANY':set(),'AUSTRIA':set()}
        self.last_PRP_review_timestamp = {'RUSSIA':0,'TURKEY':0,'ITALY':0,'ENGLAND':0,'FRANCE':0,'GERMANY':0,'AUSTRIA':0}
        
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

        num_beams   = 4
        batch_size  = 16
        device = 'cuda:0'
        model_dir  = '/diplomacy_cicero/fairdiplomacy/AMR/amrlib/amrlib/data/model_parse_xfm/checkpoint-9920/'
        self.inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)
        self.sent_self_intent = False

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

                    #TODO: Yanze check PRP message (you can follow some steps in update_press_dipcc_game 
                    # to get messages in current turn and check if it's daide)

                    proposal_response = self.check_PRP(msg,power_name)
                    #TODO: Yanze reply_to_proposal(proposal, cicero_response)

                    # send message in dipcc and Mila
                    if msg is not None and not proposal_response:
                        recipient_power = msg['recipient']
                        power_pseudo = self.player.state.pseudo_orders_cache.maybe_get(
                            self.dipcc_game, self.player.power, True, True, recipient_power) 
                        
                        power_po = power_pseudo[self.dipcc_current_phase]
                        for power in power_po.keys():
                            if power == power_name:
                                self_po = power_po[power]
                            else:
                                recp_po = power_po[power]
                        
                        if not self.sent_self_intent:
                            self_pseudo_log = f'CICERO_{power_name} intent: {self_po}'
                            await self.send_log(self_pseudo_log) 
                            self.sent_self_intent = True

                        # deceive_pseudo = self.player.agent.message_handler.get_deceive_orders()
                        # power_pseudo_log = f'CICERO_{power_name} random intent for {recipient_power}: {deceive_pseudo}'
                        # await self.send_log(power_pseudo_log) 

                        list_msg = self.to_daide_msg(msg)

                        #TODO: Konstantine deception_to_FCT(list_msg) -> list_msg

                        if len(list_msg)>0:

                            power_pseudo_log = f'CICERO_{power_name} search intent for {recipient_power}: {recp_po}'
                            await self.send_log(power_pseudo_log) 

                            nl_log = f"CICERO_{power_name} English message: {msg['message']}"
                            await self.send_log(nl_log) 

                            self.send_message(msg, 'dipcc')
                        for msg in list_msg:
                            self.send_message(msg, 'mila')
                            
                    await asyncio.sleep(0.25)
        
                # ORDER
                if not self.has_phase_changed():
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
                to_saved_game_format(self.game), file, ensure_ascii=False, indent=2
            )
            file.write("\n")


    def check_PRP(self,msg,power_name):
        phase_messages = self.get_messages(
                        messages=self.game.messages, power=power_name
                    )
        most_recent = self.last_PRP_review_timestamp.copy()
        for timesent,message in phase_messages.items():
            if message.message.startswith('PRP')
                if msg is not None and message is not None:
                    if msg['recipient'] == message.sender:
                        if int(str(timesent)[0:10]) > int(str(self.last_PRP_review_timestamp[message.sender])[0:10]):
                            dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)
                            if int(str(timesent)[0:10]) > int(str(most_recent[message.sender])[0:10]):
                                most_recent[message.sender] = dipcc_timesent
                            result = self.reply_to_proposal(message.message,msg)
                            if result is not None:
                                msg['message'] = result
                                self.send_message(msg, 'mila')
                                self.last_PRP_review_timestamp = most_recent
                                return True
            else:
                continue
        return False

    def reply_to_proposal(self, proposal, cicero_response):
        # Proposal: DAIDE Proposal from the speaker, for example RUSSIA-TURKEY here
        # cicero_response: Generated CICERO ENG sentences, for example TURKEY-RUSSIA here
        # return YES/REJ DAIDE response.
        positive_reply = 'YES ('
        negative_reply = 'REJ ('
        if any(item in cicero_response['message'] for item in ["reject","Idk","idk","do not agree","don't agree","refuse","rejection","not",'rather']):
            return negative_reply+proposal+')'
        elif any(item in cicero_response['message'] for item in ["yeah","okay","agree",'agreement','good','great',"I'm in",'like','down','perfect','Brilliant','ok','Ok','Good','Great']):
            return positive_reply+proposal+')'
        else:
            return None

    def to_daide_msg(self, msg: MessageDict):
        print('-----------------------')
        print(f'Parsing {msg} to DAIDE')

        try:
            daide_status,daide_s = self.eng_to_daide(msg, self.inference)
        except:
            daide_status,daide_s = 'NO-DAIDE',''
        # if isinstance(self.player.state, SearchBotAgentState):
        pseudo_orders = self.player.state.pseudo_orders_cache.maybe_get(
                self.dipcc_game, self.player.power, True, True, None
            ) 
        list_msg = []
        if daide_status == 'Full-DAIDE':
            print(daide_status)
            print(daide_s)
            daide_s = self.check_fulldaide(daide_s)
            daide_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': daide_s}
            list_msg.append(daide_msg)
        elif daide_status == 'Partial-DAIDE' or daide_status == 'Para-DAIDE':
            current_phase_code = pseudo_orders[msg["phase"]]
            PRP_DAIDE,FCT_DAIDE = self.psudo_code_gene(current_phase_code,msg,power_dict,af_dict)
            print(daide_status)
            print(daide_s)
            fct_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': FCT_DAIDE}
            prp_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': PRP_DAIDE}
            if fct_msg['message'] not in self.sent_FCT[fct_msg['recipient']]:
                list_msg.append(fct_msg)
                self.sent_FCT[fct_msg['recipient']].add(fct_msg['message'])
            if prp_msg['message'] not in self.sent_PRP[prp_msg['recipient']]:
                list_msg.append(prp_msg)
                self.sent_PRP[prp_msg['recipient']].add(prp_msg['message'])

        # elif daide_status == 'Para-DAIDE':
        #     current_phase_code = pseudo_orders[msg["phase"]]
        #     PRP_DAIDE,FCT_DAIDE = self.psudo_code_gene(current_phase_code,msg,power_dict,af_dict)
        #     print(daide_status)
        #     print(daide_s)
        #     fct_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': FCT_DAIDE}
        #     prp_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': PRP_DAIDE}

        #     if fct_msg['message'] not in self.sent_FCT:
        #         list_msg.append(fct_msg)
        #         self.sent_FCT[fct_msg['recipient']].add(fct_msg['message'])
        #     if prp_msg['message'] not in self.sent_PRP:
        #         list_msg.append(prp_msg)
        #         self.sent_PRP[prp_msg['recipient']].add(prp_msg['message'])
        # else:
        #     print(daide_status)
        #     print(daide_s)

        return list_msg

    def check_fulldaide(self,daide_message):
        if daide_message.count('PRP') >1:
            daide_message = daide_message.replace('PRP (','',1)
        return daide_message[0:-1]

    def psudo_code_gene(self,current_phase_code,message,power_dict,af_dict):
        string1 = 'FCT (ORR'
        string2 = 'PRP (ORR'
        for country in current_phase_code.keys():
            if country == message["sender"]:
            #FCT for sender
                for i in current_phase_code[country]:
                    sen_length = len(i)
                    if sen_length == 11:
                        string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') MTO '+i[8:11]+'))'
                    elif sen_length == 7:
                        if i[6] == 'H':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') HLD))'
                        elif i[6] == 'B':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') BLD))'
                        elif i[6] == 'R':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') REM))'
                    elif sen_length == 19:
                        if i[6] =='S':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') SUP ('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') MTO '+i[16:19]+'))'
                        elif i[6] == 'C':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') CVY ('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') CTO '+i[16:19]+'))'
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') CTO '+i[16:19]+' VIA ('+i[2:5]+')))'
            else:
            #PRP for recipient
                for i in current_phase_code[country]:
                    sen_length = len(i)
                    if sen_length == 11:
                        string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') MTO '+i[8:11]+'))'
                    elif sen_length == 7:
                        if i[6] == 'H':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') HLD))'
                        elif i[6] == 'B':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') BLD))'
                        elif i[6] == 'R':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') REM))'
                    elif sen_length == 19:
                        if i[6] =='S':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') SUP ('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') MTO '+i[16:19]+'))'
                        elif i[6] == 'C':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') CVY ('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') CTO '+i[16:19]+'))'
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') CTO '+i[16:19]+' VIA ('+i[2:5]+')))'
        string1 += ')'
        string2 += ')'
        return string1,string2

    def eng_to_daide(self,message:MessageDict,inference):
        gen_graphs = inference.parse_sents([message["sender"]+' send to '+message["recipient"]+' that '+message["message"]], disable_progress=False)
        for graph in gen_graphs:
            amr = AMR()
            amr_node, s, error_list, snt_id, snt, amr_s = amr.string_to_amr(graph)
            if amr_node:
                amr.root = amr_node
            try:
                amr_s2 = amr.amr_to_string()
            except RecursionError:
                return 'No-DAIDE',''
            if amr_s2 == '(a / amr-empty)':
                daide_s, warnings = '', []
            else:
                daide_s, warnings = amr.amr_to_daide()
            if regex.search(r'[A-Z]{3}', daide_s):
                if regex.search(r'[a-z]', daide_s):
                    daide_status = 'Partial-DAIDE'
                elif warnings:
                    daide_status = 'Para-DAIDE'
                else:
                    daide_status = 'Full-DAIDE'
            else:
                daide_status = 'No-DAIDE'

            return daide_status,daide_s

    def init_phase(self):
        """     
        update new phase to the following Dict:
        """
        self.dipcc_current_phase = self.game.get_current_phase()
        self.prev_state = 0
        self.num_stop = 0
        self.last_successful_message_time = None
        self.sent_self_intent = False
        self.sent_FCT = {'RUSSIA':set(),'TURKEY':set(),'ITALY':set(),'ENGLAND':set(),'FRANCE':set(),'GERMANY':set(),'AUSTRIA':set()}
        self.sent_PRP = {'RUSSIA':set(),'TURKEY':set(),'ITALY':set(),'ENGLAND':set(),'FRANCE':set(),'GERMANY':set(),'AUSTRIA':set()}
        self.last_PRP_review_timestamp = {'RUSSIA':0,'TURKEY':0,'ITALY':0,'ENGLAND':0,'FRANCE':0,'GERMANY':0,'AUSTRIA':0}

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
            deadline = DEFAULT_DEADLINE*60
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
            if int(str(timesent)[0:10]) > int(str(self.last_received_message_time)[0:10]):

                dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)

                if timesent > most_recent:
                    most_recent = dipcc_timesent

                # Excluding the parentheses, check if the message only contains three upper letters.
                # If so, go through daide++. If there is an error, then send 'ERROR parsing {message}' to global,
                # and don't add it to the dipcc game.
                # If it has at least one part that contains anything other than three upper letters,
                # then just keep message body as original

                if is_daide(message.message):
                    pre_processed = pre_process(message.message)
                    generated_English = gen_English(pre_processed, message.recipient, message.sender)

                    # if the message is invalid daide, send an error to paquette global
                    if generated_English.startswith("ERROR"):
                        self.game.add_message(Message(
                            sender=message.sender,
                            recipient='GLOBAL',
                            message=generated_English,
                            phase=self.dipcc_current_phase,
                            time_sent=dipcc_timesent))
                        
                        print(f'Error updating invalid daide from: {message.sender} to: {message.recipient} timesent: {timesent} and body: {message.message}, an error message is sent to global')

                    # if the message is valid daide, process and send it to dipcc recipient
                    else:
                        message_to_send = post_process(generated_English, message.recipient, message.sender)

                        self.dipcc_game.add_message(
                            message.sender,
                            message.recipient,
                            message_to_send,
                            time_sent=dipcc_timesent,
                            increment_on_collision=True)
                        
                        print(f'update a message from: {message.sender} to: {message.recipient} timesent: {timesent} and body: {message_to_send}')

                # if the message is english, just send it to dipcc recipient
                else:
                    self.dipcc_game.add_message(
                        message.sender,
                        message.recipient,
                        message.message,
                        time_sent=dipcc_timesent,
                        increment_on_collision=True,
                    )

                    print(f'update a message from: {message.sender} to: {message.recipient} timesent: {timesent} and body: {message.message}')

        # update last_received_message_time 
        self.last_received_message_time = most_recent

    def update_and_process_dipcc_game(self):
        """     
        Inputs orders from the bygone phase into the dipcc game and process the dipcc game.
        """

        dipcc_game = self.dipcc_game
        mila_game = self.game
        dipcc_phase = dipcc_game.get_state()['name'] # short name for phase
        if dipcc_phase in mila_game.order_history:
            orders_from_prev_phase = mila_game.order_history[dipcc_phase] 
            
            # gathering orders from other powers from the phase that just ended
            for power, orders in orders_from_prev_phase.items():
                dipcc_game.set_orders(power, orders)

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

    async def send_log(self, log: str):
        """ 
        send log to mila games 
        """ 

        log_data = self.game.new_log_data(body=log)
        await self.game.send_log_data(log=log_data)

    def send_message(self, msg: MessageDict, engine: str):
        """ 
        send message in dipcc and mila games 
        """ 
        timesend = Timestamp.now()
        if engine =='dipcc':
            self.dipcc_game.add_message(
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
                phase=self.game.get_current_phase(),
                )
            self.game.send_game_message(message=mila_msg)

        print(f'update a message in {engine}, {msg["sender"] }->{ msg["recipient"]}: {msg["message"]}')

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
            deadline = DEFAULT_DEADLINE
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

                # Excluding the parentheses, check if the message only contains three upper letters.
                # If so, go through daide++. If there is an error, then send 'ERROR parsing {message}' to global,
                # and don't add it to the dipcc game.
                # If it has at least one part that contains anything other than three upper letters,
                # then just keep message body as original
                if message.recipient == 'GLOBAL':
                    continue
                if is_daide(message.message):
                    pre_processed = pre_process(message.message)
                    generated_English = gen_English(pre_processed, message.recipient, message.sender)

                    # if the message is invalid daide, send an error to paquette global; do nothing
                    if not generated_English.startswith("ERROR"):
                    # if the message is valid daide, process and send it to dipcc recipient

                        message_to_send = post_process(generated_English, message.recipient, message.sender)
                        
                        dipcc_game.add_message(
                            message.sender,
                            message.recipient,
                            message_to_send,
                            time_sent=dipcc_timesent,
                            increment_on_collision=True)

                # if the message is english, just send it to dipcc recipient
                else:
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
    
