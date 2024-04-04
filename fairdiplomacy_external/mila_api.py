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

import sys
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
from diplomacy.utils import strings
from daide2eng.utils import gen_English, create_daide_grammar, is_daide
# from discordwebhook import Discord

MESSAGE_DELAY_IF_SLEEP_INF = Timestamp.from_seconds(60)
ProtoMessage = google.protobuf.message.Message

DEFAULT_DEADLINE = 5
PRE_DEADLINE = 4

import json
sys.path.insert(0, '/diplomacy_cicero/fairdiplomacy/AMR/DAIDE/DiplomacyAMR/code')
from amrtodaide import AMR
sys.path.insert(0, '/diplomacy_cicero/fairdiplomacy/AMR/penman')
# import penman
import regex
sys.path.insert(0, '/diplomacy_cicero/fairdiplomacy/AMR/amrlib')
from amrlib.models.parse_xfm.inference import Inference

power_dict = {'ENGLAND':'ENG','FRANCE':'FRA','GERMANY':'GER','ITALY':'ITA','AUSTRIA':'AUS','RUSSIA':'RUS','TURKEY':'TUR'}
af_dict = {'A':'AMY','F':'FLT'}
possible_positive_response = ["yeah","okay","agree",'agreement','good','great',"I'm in",'count me in','like','down','perfect','Brilliant','ok','Ok','Good','Great','positive','sure','Alright','yes','yep','Awesome','Done','Works for me','Will do','Perfect','I agree','Fine','Agreed','yup','Absolutely','Understood','That\'s the plan','Deal']


class milaWrapper:

    def __init__(self, is_deceptive):
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
        self.last_comm_intent={'RUSSIA':None,'TURKEY':None,'ITALY':None,'ENGLAND':None,'FRANCE':None,'GERMANY':None,'AUSTRIA':None,'final':None}
        self.prev_received_msg_time_sent = {'RUSSIA':None,'TURKEY':None,'ITALY':None,'ENGLAND':None,'FRANCE':None,'GERMANY':None,'AUSTRIA':None}
        self.deceptive = is_deceptive
        self.grammar = create_daide_grammar(level=130)
        
        if self.deceptive:
            agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero_lie.prototxt')
            print('CICERO deceptive')
        else:
            agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
            print('Cicero')
        print(f"successfully load cicero config")

        self.agent = PyBQRE1PAgent(agent_config.bqre1p)

    async def play_mila(self, args) -> None:
        hostname = args.host
        port = args.port
        game_id = args.game_id
        power_name = args.power
        gamedir = args.outdir
        self.daide_fallback = args.daide_fallback
        
        print(f"Cicero joining game: {game_id} as {power_name}")
        connection = await connect(hostname, port)
        dec = 'Deceptive_' if self.deceptive else ''
        channel = await connection.authenticate(
            f"admin", "password"
        )
        self.game: NetworkGame = await channel.join_game(game_id=game_id)

        # Wait while game is still being formed
        print(f"Waiting for game to start")
        while self.game.is_game_forming:
            await asyncio.sleep(2)

        # Playing game
        print(f"Started playing")

        self.dipcc_game = self.start_dipcc_game(power_name)
        print(f"Started dipcc game")

        self.player = Player(self.agent, power_name)
        self.game_type = args.game_type
        
        num_beams   = 4
        batch_size  = 16

        if self.game_type not in [2,5]:
            self.model = 'best_model'
            device = 'cuda:0'
            model_dir  = '/diplomacy_cicero/fairdiplomacy/AMR/personal/SEN_REC_MODEL/'
            self.inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)
        

        while not self.game.is_game_done:
            self.phase_start_time = time.time()
            self.dipcc_current_phase = self.dipcc_game.get_current_phase()
            self.presubmit = False

            # fix issue that there is a chance where retreat phase appears in dipcc but not mila 
            while self.has_phase_changed():
                self.send_log(f'process dipcc game {self.dipcc_current_phase} to catch up with a current phase in mila {self.game.get_current_phase()}') 
                agent_orders = self.player.get_orders(self.dipcc_game)
                self.dipcc_game.set_orders(power_name, agent_orders)
                self.dipcc_game.process()
                self.dipcc_current_phase = self.dipcc_game.get_current_phase()

            # While agent is not eliminated
            if not self.game.powers[power_name].is_eliminated():
                logging.info(f"Press in {self.dipcc_current_phase}")
                self.sent_self_intent = False
                
                # set wait to True: to avoid being skipped in R/A phase
                self.game.set_wait(power_name, wait=True)

                # PRESS
                has_deadline = self.game.deadline > 0 
                should_stop = await self.get_should_stop()
                
                # suggest move to human
                if self.game_type==5:
                    self.suggest_move(power_name)


                while not should_stop :
                    if has_deadline:
                        # if times almost up but still can do some press, let's presubmit order
                        should_presubmit = await self.get_should_presubmit()
                        if should_presubmit and not self.presubmit:
                            self.presubmit = True
                            print(f"Pre-submit orders in {self.dipcc_current_phase}")
                            agent_orders = self.player.get_orders(self.dipcc_game)
                            self.game.set_orders(power_name=power_name, orders=agent_orders, wait=True)

                    msg=None
                    # if there is new message incoming
                    if self.has_state_changed(power_name):
                        # update press in dipcc
                        await self.update_press_dipcc_game(power_name)

                    # if not a silent agent
                    if self.game_type!=3:
                    # reply/gen new message
                        msg = self.generate_message(power_name)
                        print(f'msg from cicero to dipcc {msg}')
                    
                    if msg is not None:
                        draw_token_message = self.is_draw_token_message(msg,power_name)
                        # proposal_response = self.check_PRP(msg,power_name)

                    # send message in dipcc and Mila
                    if msg is not None and not draw_token_message and msg['recipient'] in self.game.powers:
                        recipient_power = msg['recipient']
                        power_pseudo = self.player.state.pseudo_orders_cache.maybe_get(
                            self.dipcc_game, self.player.power, True, True, recipient_power) 

                        power_po = power_pseudo[self.dipcc_current_phase]
                        for power in power_po.keys():
                            if power == power_name:
                                self_po = power_po[power]
                            else:
                                recp_po = power_po[power]
                        
                        # if not self.sent_self_intent:
                        #     self_pseudo_log = f'At the start of this phase, I intend to do: {self_po}'
                        #     self.send_log(self_pseudo_log) 
                        #     self.sent_self_intent = True

                        # keep track of intent that we talked to each recipient
                        self.set_comm_intent(recipient_power, power_po)

                        if self.game_type==0:
                            list_msg = self.to_daide_msg(msg)
                            if len(list_msg)>0:
                                for daide_msg in list_msg:
                                    self.send_log(f'My external DAIDE response is: {daide_msg["message"]}')   
                                self.send_message(msg, 'dipcc')    
                            else:
                                self.send_log(f'No valid DIADE found / Attempt to send repeated FCT/PRP messages') 

                            for msg in list_msg:
                                self.send_message(msg, 'mila')

                        elif self.game_type==1:
                            list_msg = self.to_daide_msg(msg)
                            self.send_message(msg, 'dipcc')
                            self.send_message(msg, 'mila')
                            for daide_msg in list_msg:
                                self.send_log(f'My external DAIDE response is: {daide_msg["message"]}')    
                            else:
                                self.send_log(f'No valid DIADE found / Attempt to send repeated FCT/PRP messages') 

                            for msg in list_msg:
                                self.send_message(msg, 'mila')

                        elif self.game_type==2:
                            self.send_message(msg, 'dipcc')
                            mila_timesent = self.send_message(msg, 'mila')

                            self_pseudo_log = f'After I got the message (prev msg time_sent: {self.prev_received_msg_time_sent[msg["recipient"]]}) from {recipient_power}. \
                                My response is {msg["message"]} (msg time_sent: {mila_timesent}). I intend to do: {self_po}. I expect {recipient_power} to do: {recp_po}.'
                            self.send_log(self_pseudo_log) 

                            if 'deceptive' in msg:
                                self.send_log(msg['deceptive'])
                                print(f'Cicero logs if message is deceptive: {msg["deceptive"]}')
                            
                            # for daide_msg in list_msg:
                            #     await self.send_log(f'My DAIDE response is: {daide_msg["message"]}')    
                            # else:
                            #     await self.send_log(f'No valid DIADE found / Attempt to send repeated FCT/PRP messages') 
                        
                        elif self.game_type==4:
                            list_msg = self.eng_daide_eng_dipcc(msg)
                            self_pseudo_log = f'After I got the message (prev msg time_sent: {self.prev_received_msg_time_sent[msg["recipient"]]}) from {recipient_power}. \
                                My internal response is {msg["message"]}. I intend to do: {self_po}. I expect {recipient_power} to do: {recp_po}.'
                            self.send_log(self_pseudo_log) 

                            if len(list_msg)==0:
                                self.send_log(f'No valid DIADE found / Attempt to send repeated FCT/PRP messages') 
                            else:
                                for daide_msg in list_msg:
                                    self.send_log(f'My external DAIDE-ENG response is: {daide_msg["message"]}')    
                                    self.send_message(daide_msg, 'dipcc')
                                    mila_timesent = self.send_message(daide_msg, 'mila')

                        elif self.game_type==5:
                            msg['message'] = f"{power_name} Cicero suggests a message to {msg['recipient']}: {msg['message']}"
                            msg['recipient'] = 'GLOBAL'
                            msg['type'] = 'suggested_message'
                            mila_timesent = self.send_message(msg, 'mila')
                            self.suggest_move(power_name)

                            # self_pseudo_log = f'After I got the message (prev msg time_sent: {self.prev_received_msg_time_sent[msg["recipient"]]}) from {recipient_power}. \
                            #     My response is {msg["message"]} (msg time_sent: {mila_timesent}). I intend to do: {self_po}. I expect {recipient_power} to do: {recp_po}.'
                            # self.send_log(self_pseudo_log) 

                    should_stop = await self.get_should_stop()
                    randsleep = random.random()
                    await asyncio.sleep(1 + 10* randsleep)
        
                # ORDER

                if not self.has_phase_changed():
                    print(f"Submit orders in {self.dipcc_current_phase}")
                    agent_orders = self.player.get_orders(self.dipcc_game)

                    # keep track of our final order
                    self.set_comm_intent('final', agent_orders)
                    self.send_log(f'A record of intents in {self.dipcc_current_phase}: {self.get_comm_intent()}') 

                    # set order in Mila
                    if self.game_type!=5:
                        self.game.set_orders(power_name=power_name, orders=agent_orders, wait=False)

                # wait until the phase changed
                print(f"wait until {self.dipcc_current_phase} is done", end=" ")
                while not self.has_phase_changed():
                    print("", end=".")
                    await asyncio.sleep(0.5)
                
                # when the phase has changed, update submitted orders from Mila to dipcc
                if self.has_phase_changed():
                    self.phase_end_time = time.time()
                    self.update_and_process_dipcc_game()
                    self.init_phase()
                    print(f"Process to {self.game.get_current_phase()}")
        if gamedir:
            with open(gamedir / f"{power_name}_{game_id}_output.json", mode="w") as file:
                json.dump(
                    to_saved_game_format(self.game), file, ensure_ascii=False, indent=2
                )
                file.write("\n")

    def reset_comm_intent(self):
        self.last_comm_intent={'RUSSIA':None,'TURKEY':None,'ITALY':None,'ENGLAND':None,'FRANCE':None,'GERMANY':None,'AUSTRIA':None,'final':None}
        
    def get_comm_intent(self):
        return self.last_comm_intent

    def set_comm_intent(self, recipient, pseudo_orders):
        self.last_comm_intent[recipient] = pseudo_orders
        
    def check_PRP(self,msg,power_name,response):
        phase_messages = self.get_messages(
                        messages=self.game.messages, power=power_name
                    )
        most_recent = self.last_PRP_review_timestamp.copy()
        for timesent,message in phase_messages.items():
            if message.message.startswith('PRP'):
                if msg is not None and message is not None:
                    if msg['recipient'] == message.sender:
                        if int(str(timesent)[0:10]) > int(str(self.last_PRP_review_timestamp[message.sender])[0:10]):
                            dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)
                            if int(str(timesent)[0:10]) > int(str(most_recent[message.sender])[0:10]):
                                most_recent[message.sender] = dipcc_timesent
                            if response == True:
                                result = 'YES ('+message.message+')'
                            elif response == False:
                                result = 'REJ ('+message.message+')'
                            else:
                                result = None
                                #result = self.reply_to_proposal(message.message,msg)
                            if result is not None:
                                #msg['message'] = result
                                #self.send_message(msg, 'mila')
                                self.last_PRP_review_timestamp = most_recent
                                return result
                                #return True
            else:
                continue

    # def reply_to_proposal(self, proposal, cicero_response):
    #     # Proposal: DAIDE Proposal from the speaker, for example RUSSIA-TURKEY here
    #     # cicero_response: Generated CICERO ENG sentences, for example TURKEY-RUSSIA here
    #     # return YES/REJ DAIDE response.
    #     positive_reply = 'YES ('
    #     negative_reply = 'REJ ('
    #     # if any(item in cicero_response['message'] for item in ["reject","Idk","idk","do not agree","don't agree","refuse","rejection","not",'rather']):
    #     #     return negative_reply+proposal+')'
    #     if any(item in cicero_response['message'] for item in possible_positive_response):
    #         return positive_reply+proposal+')'
    #     else:
    #         return negative_reply+proposal+')'
            
    def suggest_move(self, power_name):
        msg = {'sender': power_name}
        agent_orders = list(self.player.get_orders(self.dipcc_game))
        msg['message'] = f"{power_name} Cicero suggests move: {', '.join(agent_orders)}"
        msg['recipient'] = 'GLOBAL'
        msg['type'] = 'suggested_move'
        self.send_message(msg, 'mila')

    def is_draw_token_message(self, msg ,power_name):
        if DRAW_VOTE_TOKEN in msg['message']:
            self.game.powers[power_name].vote = strings.YES
            return True
        if UNDRAW_VOTE_TOKEN in msg['message']:
            self.game.powers[power_name].vote = strings.NO
            return True
        if DATASET_DRAW_MESSAGE in msg['message']:
            self.game.powers[power_name].vote = strings.YES
            return True
        if DATASET_NODRAW_MESSAGE in msg['message']:
            self.game.powers[power_name].vote = strings.YES
            return True
        
        return False

    def eng_daide_eng_dipcc(self, msg: MessageDict):
        daide_msgs = self.to_daide_msg(msg)
        eng_daide_msgs = []
        for daide_m in daide_msgs:
            try:
                generated_English = gen_English(daide_m['message'], daide_m['recipient'], daide_m['sender'])
            except:
                print(f"Fail to translate the message into the English, from {daide_m['sender']}: {daide_m['message']}")
                self.send_log(f"Fail to translate the message into the English, from {daide_m['sender']}: {daide_m['message']}") 
            
            if generated_English.startswith("ERROR") or generated_English.startswith("Exception"):
                print(f"Fail to translate the message into the English, from {daide_m['sender']}: {daide_m['message']}")
                self.send_log(f"Fail to translate the message into the English, from {daide_m['sender']}: {daide_m['message']}") 
            else:
                eng_daide_msg = {'sender': daide_m['sender'] ,'recipient': daide_m['recipient'], 'message': generated_English}
                eng_daide_msgs.append(eng_daide_msg)
        
        return eng_daide_msgs

    def eng_daide_eng_mila(self, msg: Message):
        mila_dict_msg = {'sender': msg.sender ,'recipient': msg.recipient, 'message': msg.message, 'phase':msg.phase}
        return self.eng_daide_eng_dipcc(mila_dict_msg)

    def divide_sentences(self,sentence):
        if sentence.startswith('AND '):
            sentence = sentence[5:-1]
            list1 = sentence.split(') (')
            return list1
        elif sentence.startswith('PRP (AND '):
            sentence = sentence[10:-2]
            list1 = sentence.split(') (')
            for i in range(len(list1)):
                list1[i] = 'PRP ('+list1[i]+')'
            return list1
        elif sentence.startswith('THK (AND '):
            sentence = sentence[10:-2]
            list1 = sentence.split(') (')
            for i in range(len(list1)):
                list1[i] = 'THK ('+list1[i]+')'
            return list1

    def to_daide_msg(self, msg: MessageDict):
        print('-------------------------')
        print(f'Parsing {msg} to DAIDE')

        pseudo_orders = self.player.state.pseudo_orders_cache.maybe_get(
                self.dipcc_game, self.player.power, True, True, msg['recipient']
                ) 
        power_name = self.player.power.upper()

        list_msg = []

        if pseudo_orders is None or msg['sender'] not in pseudo_orders[self.dipcc_current_phase] or msg['recipient'] not in pseudo_orders[self.dipcc_current_phase]:
            return list_msg

        if self.daide_fallback:
            current_phase_code = pseudo_orders[msg["phase"]]
            FCT_DAIDE, PRP_DAIDE = self.psudo_code_gene(current_phase_code,msg,power_dict,af_dict)
            
            if FCT_DAIDE is not None:
                FCT_DAIDE = self.remove_ORR(FCT_DAIDE)
                fct_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': FCT_DAIDE,'daide_status':'fall_back'}
                if fct_msg['message'] not in self.sent_FCT[fct_msg['recipient']]:
                    list_msg.append(fct_msg)
                    self.sent_FCT[fct_msg['recipient']].add(fct_msg['message'])
            if PRP_DAIDE is not None:
                PRP_DAIDE = self.remove_ORR(PRP_DAIDE)
                prp_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': PRP_DAIDE,'daide_status':'fall_back'}
                if prp_msg['message'] not in self.sent_PRP[prp_msg['recipient']]:
                    list_msg.append(prp_msg)
                    self.sent_PRP[prp_msg['recipient']].add(prp_msg['message'])

            return list_msg

        try:
            daide_status,daide_s = self.eng_to_daide(msg, self.inference)
        except:
            daide_status,daide_s = 'NO-DAIDE',''
        # if isinstance(self.player.state, SearchBotAgentState):


        # I changed the rule of Full-DAIDE, it passes the daidepp checker now so we no longer to check fulldaide and remove additonal ORR here
        if daide_status == 'Full-DAIDE':
            print(daide_status)
            print(daide_s)
            # daide_s = self.check_fulldaide(daide_s)
            # daide_s = self.remove_ORR(daide_s)
            daide_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': daide_s,'daide_status':daide_status}
            list_msg.append(daide_msg)
        elif daide_status == 'Partial-DAIDE' or daide_status == 'Para-DAIDE':
            print(daide_status)
            print(daide_s)
            if 'NOT (YES' in daide_s or 'REJ' in daide_s:
                #reject
                daide_s = self.check_PRP(msg,power_name,False)
                if daide_s is not None:
                    daide_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': daide_s,'daide_status':daide_status}
                    list_msg.append(daide_msg)
            elif 'YES_LAST' in daide_s or 'YES' in daide_s:
                #agree
                daide_s = self.check_PRP(msg,power_name,True)
                if daide_s is not None:
                    daide_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': daide_s,'daide_status':daide_status}
                    list_msg.append(daide_msg)
            else:
                daide_list = self.divide_sentences(daide_s)
                if daide_list:
                    for i in daide_list:
                        daide_status = self.check_valid(i)
                        if daide_status == 'Full-DAIDE':
                            daide_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': i,'daide_status':'daide-split'}
                            list_msg.append(daide_msg)
                else:
                    current_phase_code = pseudo_orders[msg["phase"]]
                    FCT_DAIDE, PRP_DAIDE = self.psudo_code_gene(current_phase_code,msg,power_dict,af_dict)
                    
                    if FCT_DAIDE is not None:
                        FCT_DAIDE = self.remove_ORR(FCT_DAIDE)
                        fct_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': FCT_DAIDE,'daide_status':'fall_back'}
                        if fct_msg['message'] not in self.sent_FCT[fct_msg['recipient']]:
                            list_msg.append(fct_msg)
                            self.sent_FCT[fct_msg['recipient']].add(fct_msg['message'])
                    if PRP_DAIDE is not None:
                        PRP_DAIDE = self.remove_ORR(PRP_DAIDE)
                        prp_msg = {'sender': msg['sender'] ,'recipient': msg['recipient'], 'message': PRP_DAIDE,'daide_status':'fall_back'}
                        if prp_msg['message'] not in self.sent_PRP[prp_msg['recipient']]:
                            list_msg.append(prp_msg)
                            self.sent_PRP[prp_msg['recipient']].add(prp_msg['message'])
        else:
            print(daide_status)
            print(daide_s)
            self.add_openning_message(msg, list_msg)

        return list_msg

    def remove_ORR(self,daide_message):
        
        if 'ORR' in daide_message:
            print(f'removing ORR from {daide_message}')
            removed_message = ''
            daide_message = daide_message.replace('(ORR ','')
            daide_message = daide_message[0:-1]
            message_list = daide_message.split('XDO')
            if message_list:
                removed_message += message_list[0]+'XDO'+message_list[1]
            removed_message = removed_message.replace(') (',')')
            return removed_message
        else:
            return daide_message

    def check_fulldaide(self,daide_message):
        if daide_message.count('PRP') >1:
            daide_message = daide_message.replace('PRP (','',1)
        return daide_message[0:-1]

    def psudo_code_gene(self,current_phase_code,message,power_dict,af_dict):
        string1 = 'FCT (ORR'
        string2 = 'PRP (ORR'
        has_FCT_order = False
        has_PRP_order = False
        for country in current_phase_code.keys():
            if country == message["sender"]:
            #FCT for sender
                for i in current_phase_code[country]:
                    sen_length = len(i)
                    if sen_length == 11:
                        string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') MTO '+i[8:11]+'))'
                        has_FCT_order = True
                    elif sen_length == 7:
                        if i[6] == 'H':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') HLD))'
                            has_FCT_order = True
                        elif i[6] == 'B':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') BLD))'
                            has_FCT_order = True
                        elif i[6] == 'R':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') REM))'
                            has_FCT_order = True
                    elif sen_length == 19:
                        if i[6] =='S':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') SUP ('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') MTO '+i[16:19]+'))'
                            has_FCT_order = True
                        elif i[6] == 'C':
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') CVY ('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') CTO '+i[16:19]+'))'
                            string1 += ' (XDO (('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') CTO '+i[16:19]+' VIA ('+i[2:5]+')))'
                            has_FCT_order = True
            else:
            #PRP for recipient
                for i in current_phase_code[country]:
                    sen_length = len(i)
                    if sen_length == 11:
                        string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') MTO '+i[8:11]+'))'
                        has_PRP_order = True
                    elif sen_length == 7:
                        if i[6] == 'H':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') HLD))'
                            has_PRP_order = True
                        elif i[6] == 'B':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') BLD))'
                            has_PRP_order = True
                        elif i[6] == 'R':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') REM))'
                            has_PRP_order = True
                    elif sen_length == 19:
                        if i[6] =='S':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') SUP ('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') MTO '+i[16:19]+'))'
                            has_PRP_order = True
                        elif i[6] == 'C':
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[0]]+' '+i[2:5]+') CVY ('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') CTO '+i[16:19]+'))'
                            string2 += ' (XDO (('+power_dict[country]+' '+af_dict[i[8]]+' '+i[10:13]+') CTO '+i[16:19]+' VIA ('+i[2:5]+')))'
                            has_PRP_order = True
        string1 += ')'
        string2 += ')'
        if not has_FCT_order:
            string1 = None
        if not has_PRP_order:
            string2 = None
        return string1,string2


    def check_valid(self, daide_sentence):
        try:
            parse_tree = self.grammar.parse(daide_sentence)
            Full = True
        except:
            Full = False
        if regex.search(r'[A-Z]{3}', daide_sentence):
            if regex.search(r'[a-z]', daide_sentence):
                daide_status = 'Partial-DAIDE'
            elif Full == False:
                daide_status = 'Para-DAIDE'
            else:
                daide_status = 'Full-DAIDE'
        else:
            daide_status = 'No-DAIDE'
        return daide_status

    def eng_to_daide(self,message:MessageDict,inference):
        print('---------------------------')
        if self.model == 'best_model':
            gen_graphs = inference.parse_sents(['SEN'+' send to '+'REC'+' that '+message["message"]], disable_progress=False)
        elif self.model == 'baseline_model':
            gen_graphs = inference.parse_sents([message["message"]], disable_progress=False)
        elif self.model == 'dipdata_model':
            gen_graphs = inference.parse_sents([message["message"]], disable_progress=False)
        elif self.model == 'improvement1_model':
            gen_graphs = inference.parse_sents([message["sender"].capitalize()+' send to '+message["recipient"].capitalize()+' that '+message["message"]], disable_progress=False)
        for graph in gen_graphs:
            print(graph)
            if self.model == 'best_model':
                graph = graph.replace('SEN',message["sender"].capitalize()).replace('REC',message["recipient"].capitalize())
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
            # try:
            #     parse_tree = self.grammar.parse(daide_s)
            #     Full = True
            # except:
            #     Full = False
            # if regex.search(r'[A-Z]{3}', daide_s):
            #     if regex.search(r'[a-z]', daide_s):
            #         daide_status = 'Partial-DAIDE'
            #     elif Full == False:
            #         daide_status = 'Para-DAIDE'
            #     else:
            #         daide_status = 'Full-DAIDE'
            # else:
            #     daide_status = 'No-DAIDE'
            daide_status = self.check_valid(daide_s)
            return daide_status,daide_s

    def add_openning_message(self, message:MessageDict, list_msg):

        # if this is the first phase
        if self.game.get_current_phase() == "S1901M":
            possible_alliance_proposal = {message["sender"][0]+message["recipient"][0], message["recipient"][0]+message["sender"][0], message["sender"][0]+'/'+message["recipient"][0],message["recipient"][0]+'/'+message["sender"][0]}
            possible_alliance_name = {'juggernaut', 'wintergreen', 'lepanto'}
            other_powers = [power_dict[p] for p in power_dict if p != message["sender"] and p != message["recipient"]]
            other_powers = ' '.join(other_powers)

            if any(item in message['message'] for item in possible_alliance_proposal):
                prop_string = f'PRP (ALY ({power_dict[message["sender"]]} {power_dict[message["recipient"]]}) VSS ({other_powers}))'
                daide_msg = {'sender': message['sender'] ,'recipient': message['recipient'], 'message': prop_string,'daide_status':'open_message'}
                list_msg.append(daide_msg)
                return

            if 'juggernaut' in message['message'].lower() and (message["sender"] in ['RUSSIA', 'TURKEY'] or message["recipient"] in ['RUSSIA', 'TURKEY']):
                prop_string = f'PRP (ALY ({power_dict[message["sender"]]} {power_dict[message["recipient"]]}) VSS ({other_powers}))'
                daide_msg = {'sender': message['sender'] ,'recipient': message['recipient'], 'message': prop_string,'daide_status':'open_message'}
                list_msg.append(daide_msg)
                return

            if 'wintergreen' in message['message'].lower() and (message["sender"] in ['RUSSIA', 'ITALY'] or message["recipient"] in ['RUSSIA', 'ITALY']):
                prop_string = f'PRP (ALY ({power_dict[message["sender"]]} {power_dict[message["recipient"]]}) VSS ({other_powers}))'
                daide_msg = {'sender': message['sender'] ,'recipient': message['recipient'], 'message': prop_string,'daide_status':'open_message'}
                list_msg.append(daide_msg)
                return

            if 'lepanto' in message['message'].lower() and (message["sender"] in ['AUSTRIA', 'ITALY'] or message["recipient"] in ['AUSTRIA', 'ITALY']):
                prop_string = f'PRP (ALY ({power_dict[message["sender"]]} {power_dict[message["recipient"]]}) VSS (TUR))'
                daide_msg = {'sender': message['sender'] ,'recipient': message['recipient'], 'message': prop_string,'daide_status':'open_message'}
                list_msg.append(daide_msg)
                return
        return


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
        self.reset_comm_intent()

    def has_phase_changed(self)->bool:
        """ 
        check game phase 
        """
        return self.dipcc_game.get_current_phase() != self.game.get_current_phase()

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

    async def get_should_presubmit(self)->bool:
        schedule = await self.game.query_schedule()
        self.scheduler_event = schedule.schedule
        server_end = self.scheduler_event.time_added + self.scheduler_event.delay
        server_remaining = server_end - self.scheduler_event.current_time
        deadline_timer = server_remaining * self.scheduler_event.time_unit

        presubmit_second = 120

        if deadline_timer <= presubmit_second and self.game_type != 5:
            print(f'time to presubmit order')
            return True
        return False


    async def get_should_stop(self)->bool:
        """ 
        stop when:
        1. close to deadline! (make sure that we have enough time to submit order)
        2. reuse stale pseudoorders
        """
        if self.has_phase_changed():
            return True

        no_message_second = 30
        deadline = self.game.deadline
        
        close_to_deadline = deadline - no_message_second

        assert close_to_deadline >= 0, "Press period is less than zero"

        current_time = time.time()

        has_deadline = self.game.deadline > 0 
        if has_deadline and current_time - self.phase_start_time <= close_to_deadline:
            schedule = await self.game.query_schedule()
            self.scheduler_event = schedule.schedule
            server_end = self.scheduler_event.time_added + self.scheduler_event.delay
            server_remaining = server_end - self.scheduler_event.current_time
            deadline_timer = server_remaining * self.scheduler_event.time_unit
            print(f'remaining time to play: {deadline_timer}')
        else:
            deadline = DEFAULT_DEADLINE*60

        # PRESS allows in movement phase (ONLY)
        if not self.dipcc_game.get_current_phase().endswith("M"):
            return True
        if current_time - self.phase_start_time >= close_to_deadline:
            return True   
        if has_deadline and deadline_timer <= no_message_second:
            return True
        if self.last_received_message_time != 0 and current_time - self.last_received_message_time >=no_message_second:
            print(f'no incoming message for {current_time - self.last_received_message_time} seconds')
            return True
        
        # check if reuse state psedo orders for too long
        if self.reuse_stale_pseudo():
            return True

        return False

    async def update_press_dipcc_game(self, power_name: POWERS):
        """ 
        update new messages that present in Mila to dipcc
        """
        assert self.game.get_current_phase() == self.dipcc_current_phase, "Phase in two engines are not synchronized"

        mila_phase = self.game.get_current_phase()
        
        # new messages in current phase from Mila 
        phase_messages = self.get_messages(
            messages=self.game.messages, power=power_name
        )
        most_recent_mila = self.last_received_message_time
        most_recent_sent_mila = self.last_sent_message_time
        # print(f'most update message: {most_recent}')

        # update message in dipcc game
        for timesent, message in phase_messages.items():
            
            # skip those that were sent to GLOBAL
            if message.recipient not in POWERS:
                continue

            self.prev_received_msg_time_sent[message.sender] = message.time_sent
            
            if (int(str(timesent)[0:10]) > int(str(self.last_received_message_time)[0:10]) and message.recipient == power_name) or (int(str(timesent)[0:10]) > int(str(self.last_sent_message_time)[0:10]) and message.sender == power_name):
                dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)
                # dipcc_timesent =Timestamp.now()
                # print(f'time_sent in dipcc {dipcc_timesent}')
                

                if timesent > most_recent_mila and message.recipient == power_name:
                    most_recent_mila = timesent
                elif timesent > most_recent_sent_mila and message.sender == power_name:
                    most_recent_sent_mila = timesent

                # Excluding the parentheses, check if the message only contains three upper letters.
                # If so, go through daide++. If there is an error, then send 'ERROR parsing {message}' to global,
                # and don't add it to the dipcc game.
                # If it has at least one part that contains anything other than three upper letters,
                # then just keep message body as original

                if is_daide(message.message):
                    try:
                        generated_English = gen_English(message.message, message.recipient, message.sender)
                    except:
                        print(f"Fail to translate the message into the English, from {message.sender}: {message.message}")
                        self.send_log(f"Fail to translate the message into the English, from {message.sender}: {message.message}") 
                        return

                    # if the message is invalid daide, send an error to paquette global
                    if generated_English.startswith("ERROR") or generated_English.startswith("Exception"):
                        self.game.add_message(Message(
                            sender=message.sender,
                            recipient=message.recipient,
                            message=f'HUH ({message.message})',
                            phase=self.dipcc_current_phase,
                            time_sent=dipcc_timesent))

                        self.send_log(f"I got this message from {message.sender}: {message.message}") 
                        self.send_log(f"Fail to translate into the English") 
                        
                        # print(f'Error updating invalid daide from: {message.sender} to: {message.recipient} timesent: {timesent} and body: {message.message}, an error message is sent to global')

                    # if the message is valid daide, process and send it to dipcc recipient
                    else:
                        self.dipcc_game.add_message(
                            message.sender,
                            message.recipient,
                            generated_English,
                            time_sent=dipcc_timesent,
                            increment_on_collision=True)
                        
                        self.send_log(f"I got this message from {message.sender}: {message.message}") 
                        self.send_log(f"Translated into the English, that is: {generated_English}") 

                        # print(f'update a message from: {message.sender} to: {message.recipient} timesent: {timesent} and body: {message_to_send}')

                # if the message is english, just send it to dipcc recipient
                else:
                    print(f'message from mila to dipcc {message}')
                    if self.game_type==4:
                        list_eng_daide_message = self.eng_daide_eng_mila(message)
                        for msg_dict in list_eng_daide_message:
                            self.dipcc_game.add_message(
                                msg_dict['sender'],
                                msg_dict['recipient'],
                                msg_dict['message'],
                                # time_sent=dipcc_timesent,
                                time_sent=Timestamp.now(),
                                increment_on_collision=True,
                            )

                    else:
                        self.dipcc_game.add_message(
                            message.sender,
                            message.recipient,
                            message.message,
                            # time_sent=dipcc_timesent,
                            time_sent=Timestamp.now(),
                            increment_on_collision=True,
                        )

                    # print(f'update a message from: {message.sender} to: {message.recipient} timesent: {timesent} and body: {message.message}')

        # update last_received_message_time 
        self.last_received_message_time = most_recent_mila
        self.last_sent_message_time = most_recent_sent_mila

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

        # set human_intent
        human_intent = self.game.get_orders(power_name)
        if human_intent:
            self.agent.set_power_po(human_intent)
        
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

    def send_log(self, log: str):
        """ 
        send log to mila games 
        """ 
        if self.game_type == 5:
            print('skip log')
        else:
            log_data = self.game.new_log_data(body=log)
            self.game.send_log_data(log=log_data)

    def send_message(self, msg: MessageDict, engine: str):
        """ 
        send message in dipcc and mila games 
        """ 
        
        if engine =='dipcc':
            timesend = Timestamp.now()
            self.dipcc_game.add_message(
                        msg['sender'], 
                        msg['recipient'], 
                        msg['message'], 
                        time_sent=timesend,
                        increment_on_collision=True,
                    )

        if engine =='mila':
            mila_msg = Message(
                sender= "omniscient_type" if self.game_type ==5 else msg["sender"],
                recipient=msg["recipient"],
                message=msg["message"],
                phase=self.game.get_current_phase(),
                type = msg['type']
                )
            self.game.send_game_message(message=mila_msg)
            timesend = mila_msg.time_sent

        print(f'update a message in {engine}, {msg["sender"] }->{ msg["recipient"]}: {msg["message"]}')
        return timesend

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
            deadline = int(math.ceil(deadline/60))
        
        game = Game()

        game.set_scoring_system(Game.SCORING_SOS)
        game.set_metadata("phase_minutes", str(deadline))
        game = game_from_view_of(game, power_name)

        while game.get_state()['name'] != self.game.get_current_phase():
            self.update_past_phase(game,  game.get_state()['name'], power_name)

        return game
    
    def update_past_phase(self, dipcc_game: Game, phase: str, power_name: str):
        if phase not in self.game.message_history:
            dipcc_game.process()
            return

        phase_message = self.game.message_history[phase]
        for timesent, message in phase_message.items():

            if message.recipient != power_name or message.sender != power_name:
                continue

            dipcc_timesent = Timestamp.from_seconds(timesent * 1e-6)

            # Excluding the parentheses, check if the message only contains three upper letters.
            # If so, go through daide++. If there is an error, then send 'ERROR parsing {message}' to global,
            # and don't add it to the dipcc game.
            # If it has at least one part that contains anything other than three upper letters,
            # then just keep message body as original
            if message.recipient not in self.game.powers:
                continue
            # print(f'load message from mila to dipcc {message}')

            if is_daide(message.message):
                generated_English = gen_English(message.message, message.recipient, message.sender)

                # if the message is invalid daide, send an error to paquette global; do nothing
                if generated_English.startswith("ERROR") or generated_English.startswith("Exception"):
                # if the message is valid daide, process and send it to dipcc recipient
                    
                    dipcc_game.add_message(
                        message.sender,
                        message.recipient,
                        generated_English,
                        time_sent=dipcc_timesent,
                        increment_on_collision=True)

            # if the message is english, just send it to dipcc recipient
            else:
                if self.game_type==4:
                    list_eng_daide_message = self.eng_daide_eng_mila(message)
                    for msg_dict in list_eng_daide_message:
                        dipcc_game.add_message(
                            msg_dict['sender'],
                            msg_dict['recipient'],
                            msg_dict['message'],
                            # time_sent=dipcc_timesent,
                            time_sent=Timestamp.now(),
                            increment_on_collision=True,
                        )

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
    parser.add_argument(
        "--game_type",
        type=int, 
        default=0,
        help="0: AI-only game, 1: Human and AI game, 2: Human-only game, 3: silent, 4: human with eng-daide-eng Cicero, 5: chiron",
    )
    # parser.add_argument(
    #     "--agent",
    #     type=Path,
    #     # default ="/diplomacy_cicero/conf/common/agents/cicero.prototxt",
    #     default ="/diplomacy_cicero/conf/agents/bqre1p_parlai_20220819_cicero_2.prototxt",
    #     help="path to prototxt with agent's configurations (default: %(default)s)",
    # )
    parser.add_argument(
        "--deceptive",
        default= False, 
        action="store_true",
        help="Is Cicero being deceptive? -- removing PO correspondence filter from message module?",
    )
    parser.add_argument(
        "--daide_fallback", 
        action="store_true", 
        default=False, 
        help="Will you skip AMR->DAIDE parser and generate DAIDE with fallback only?",
    )
    parser.add_argument(
        "--outdir", default= "./fairdiplomacy_external/out", type=Path, help="output directory for game json to be stored"
    )
    
    args = parser.parse_args()
    host: str = args.host
    port: int = args.port
    game_id: str = args.game_id
    power: str = args.power
    deceptive: bool = args.deceptive
    daide_fallback : bool = args.daide_fallback
    outdir: Optional[Path] = args.outdir
    game_type : int = args.game_type

    print(f"settings:")
    print(f"host: {host}, port: {port}, game_id: {game_id}, power: {power}")

    if outdir is not None and not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    mila = milaWrapper(is_deceptive=deceptive)
    discord = Discord(url="https://discord.com/api/webhooks/1209977480652521522/auWUQRA8gz0HT5O7xGWIdKMkO5jE4Rby-QcvukZfx4luj_zwQeg67FEu6AXLpGTT41Qz")
    discord.post(content=f"Cicero as power {power} is joining {game_id}.")

    # while True:
    #     try:
    asyncio.run(
        mila.play_mila(args)
    )
        # except Exception:
        #     print(Exception)
        #     cicero_error = f"Cicero_{power} has an error occured but we are rerunning it"
        #     discord.post(content=cicero_error)

        

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
    
