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
from discordwebhook import Discord

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
from parlai_diplomacy.wrappers.classifiers import INF_SLEEP_TIME

logger = return_logger(__name__)

MESSAGE_DELAY_IF_SLEEP_INF = Timestamp.from_seconds(60)

DEFAULT_DEADLINE = 5

ADVICE_LEVELS_MESSAGE = (
    "Levels of advice: "
    f"{int(SuggestionType.NONE)}: none, "
    f"{int(SuggestionType.MESSAGE)}: message only, "
    f"{int(SuggestionType.MOVE)}: order only, "
    f"{int(SuggestionType.MESSAGE | SuggestionType.MOVE)}: both message and order, "
    f"{int(SuggestionType.OPPONENT_MOVE)}: opponent moves, "
    f"{int(SuggestionType.MESSAGE | SuggestionType.MOVE | SuggestionType.OPPONENT_MOVE)}: messages, moves, and opponent moves"
)

@dataclass
class CiceroBot(BaselineBot, ABC):
    async def gen_orders(self) -> List[str]:
        return []

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        return []


@dataclass
class CiceroAdvisor(CiceroBot):
    """Advisor form of `CiceroBot`."""

    bot_type = BotType.ADVISOR
    default_suggestion_type = (
        SuggestionType.MESSAGE
        | SuggestionType.MOVE
        | SuggestionType.OPPONENT_MOVE
    )


class milaWrapper:

    def __init__(self):
        self.game: NetworkGame = None
        self.chiron_agent: Optional[CiceroBot] = None
        self.dipcc_game: Game = None
        self.prev_state = 0                                         # number of number received messages in the current phase
        self.dialogue_state = {}                                    # {phase: number of all (= received + new) messages for agent}
        self.last_sent_message_time = 0                             # {phase: timestamp} 
        self.last_received_message_time = 0                         # {phase: time_sent (from Mila json)}                                      
        self.dipcc_current_phase = None                             
        self.last_successful_message_time = None                    # timestep for last message successfully sent in the current phase                       
        self.last_comm_intent={'RUSSIA':None,'TURKEY':None,'ITALY':None,'ENGLAND':None,'FRANCE':None,'GERMANY':None,'AUSTRIA':None,'final':None}
        self.prev_received_msg_time_sent = {'RUSSIA':None,'TURKEY':None,'ITALY':None,'ENGLAND':None,'FRANCE':None,'GERMANY':None,'AUSTRIA':None}
        self.new_message = {'RUSSIA':0,'TURKEY':0,'ITALY':0,'ENGLAND':0,'FRANCE':0,'GERMANY':0,'AUSTRIA':0}
        self.prev_suggest_moves = None
        self.prev_suggest_cond_moves = None
        self.power_to_advise = None
        self.advice_level = None
        self.weight_powers = dict()
        self.decrement_value = 0.2
        
        agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
        logger.info("Successfully loaded CICERO config")

        self.agent = PyBQRE1PAgent(agent_config.bqre1p)
        
    async def assign_advisor(self, file_dir, power_dist, advice_levels):
        # random N powers
        # random level 
        self.power_to_advise = sample_p_dict(power_dist)
        if len(power_dist) ==1 and len(advice_levels)>1 and int(SuggestionType.NONE) not in advice_levels:
            logger.info(f'We are left with only one power ({list(power_dist)[0]}), '
                        f'so let\'s add {int(SuggestionType.NONE)} as no advice to advice levels')
            advice_levels.append(int(SuggestionType.NONE))
        logger.info(f'Randomly choosing from advice levels: {advice_levels}')
        self.advice_level = random.choice(advice_levels)
        await self.send_log(f'Assigning Cicero to {self.power_to_advise} and advising at level {self.advice_level}')
        logger.info(f"Note: {ADVICE_LEVELS_MESSAGE}")
        # write to json
        with open(file_dir, 'w') as f:
            advisor_dict = {'assign_phase': self.game.get_current_phase(), 'power_to_advise':self.power_to_advise, 'advice_level':self.advice_level}
            json.dump(advisor_dict, f, indent=4)
            
        #adjust prob dist
        old_power_dist = copy.deepcopy(power_dist)
        power_dist[self.power_to_advise] = max(0, power_dist[self.power_to_advise] - self.decrement_value)
        power_dist = normalize_p_dict(power_dist)
        self.weight_powers = power_dist
        logger.info(f'Adjusting power distribution from {old_power_dist} to {power_dist}')
        
    async def reload_or_assign_advisor(self, file_dir, power_dist, advice_levels):
        if os.path.exists(file_dir):
            with open(file_dir, mode="r") as file:
                advisor_json = json.load(file)
                
            game_phase = self.game.get_current_phase()
            phase_file = advisor_json['assign_phase']
            # load if this is not the first phase in year which means year in game == year in phase file
            if game_phase != 'S1901M' and game_phase[1:5] == phase_file[1:5]:
                self.power_to_advise = advisor_json['power_to_advise']
                self.advice_level = advisor_json['advice_level']
                logger.info(f'Re-assigning Cicero to {self.power_to_advise} and advising {self.advice_level}')
                logger.info(f"Note: {ADVICE_LEVELS_MESSAGE}")
                return True

        await self.assign_advisor(file_dir, power_dist, advice_levels)
        return False

    
    def get_playable_human_powers(self, game, human_powers):
        playable_powers = []
        
        for power in human_powers:
            if not game.powers[power].is_eliminated():
                playable_powers.append(power)
            
        return playable_powers
    
    def get_curr_power_to_advise(self):
        return self.power_to_advise

    def get_curr_advice_level(self):
        return self.advice_level

    async def play_mila(
        self,
        hostname: str,
        port: int,
        use_ssl: bool,
        game_id: str,
        gamedir: Path ,
        human_powers: List[str],
        advice_levels_strs: List[str],
    ) -> None:
        advice_levels = [int(l) for l in advice_levels_strs]
        power_name = None

        connection = await connect(hostname, port, use_ssl)
        channel = await connection.authenticate(
            f"admin", "password"
        )
        self.game: NetworkGame = await channel.join_game(game_id=game_id)

        self.chiron_agent = CiceroAdvisor(power_name, self.game)

        # Wait while game is still being formed
        logger.info(f"Waiting for game to start")
        while self.game.is_game_forming:
            await asyncio.sleep(2)
            logger.info("Still waiting")

        # Playing game
        logger.info(f"Started playing")

        while not self.game.is_game_done:
            self.phase_start_time = time.time()
            self.presubmit = False
            if self.dipcc_game:
                self.dipcc_current_phase = self.dipcc_game.get_current_phase()

            # fix issue that there is a chance where retreat phase appears in dipcc but not mila 
            while self.dipcc_game and self.has_phase_changed():
                await self.send_log(f'process dipcc game {self.dipcc_current_phase} to catch up with a current phase in mila {self.game.get_current_phase()}') 
                agent_orders = self.player.get_orders(self.dipcc_game)
                if power_name is None:
                    power_name = self.get_curr_power_to_advise()
                self.dipcc_game.set_orders(power_name, agent_orders)
                self.dipcc_game.process()
                self.dipcc_current_phase = self.dipcc_game.get_current_phase()
            
            file_advisor = gamedir / f"{game_id}_{'_'.join(human_powers)}.json"
            #for every m turn -> reassign advisor
            # get whom to advice (not eliminated and human powers)
            playable_powers = self.get_playable_human_powers(self.game, human_powers)
            if len(playable_powers) != len(self.weight_powers):
                self.weight_powers = {power:  1/len(playable_powers) for power in playable_powers}
            # then assign power and level of advice
            is_reload_advisor = await self.reload_or_assign_advisor(file_advisor, self.weight_powers, advice_levels)

            # if newly assign or fist time loading cicero
            if not is_reload_advisor or power_name is None:
                power_name = self.get_curr_power_to_advise()
                self.chiron_type = self.get_curr_advice_level()
                # Only one instance of `ChironAdvisor` exists at a time,
                # so this is actually safe
                self.chiron_agent.suggestion_type = self.chiron_type
                self.chiron_agent.power_name = power_name
                self.dipcc_game = self.start_dipcc_game(power_name)
                self.player = Player(self.agent, power_name)
                self.dipcc_current_phase = self.dipcc_game.get_current_phase()

            await self.chiron_agent.declare_suggestion_type()

            # While agent is not eliminated
            if not self.game.powers[power_name].is_eliminated():
                logging.info(f"Press in {self.dipcc_current_phase}")
                self.sent_self_intent = False
                
                # set wait to True: to avoid being skipped in R/A phase
                self.game.set_wait(power_name, wait=True)

                # PRESS allowed in movement phase (ONLY)
                if self.dipcc_game.get_current_phase().endswith("M"):
                    await self.chiron_agent.wait_for_comm_stage()

                if self.chiron_type & SuggestionType.OPPONENT_MOVE:
                    await self.predict_opponent_moves(power_name)

                # PRESS
                should_stop = await self.get_should_stop()
                if self.chiron_type & SuggestionType.MOVE:
                    await self.suggest_move(power_name)
    
                while not should_stop:
                    # suggest move to human
                    if self.chiron_type & SuggestionType.MOVE:
                        await self.suggest_move(power_name)
                        
                    msg=None
                    # if there is new message incoming
                    if self.has_state_changed(power_name):
                        # update press in dipcc
                        await self.update_press_dipcc_game(power_name)

                    if self.chiron_type & SuggestionType.MESSAGE:
                        # reply/gen new message
                        msg = self.generate_message(power_name)
                        logger.info(f'Possible message suggestion: {msg!r}')
                    
                    if msg is not None:
                        draw_token_message = self.is_draw_token_message(msg,power_name)

                    # send message in dipcc and Mila
                    if msg is not None and not draw_token_message and msg['recipient'] in self.game.powers:
                        recipient_power = msg['recipient']
                        power_pseudo = self.player.state.pseudo_orders_cache.maybe_get(
                            self.dipcc_game, self.player.power, True, True, recipient_power) 

                        power_po = power_pseudo[self.dipcc_current_phase]

                        # keep track of intent that we talked to each recipient
                        self.set_comm_intent(recipient_power, power_po)

                        current_time = time.time()
                        if self.chiron_type & SuggestionType.MESSAGE and current_time - self.new_message[msg['recipient']]>=60:
                            K = 2
                            self.new_message[msg['recipient']] = current_time
                            msg_options = [msg] 
                            msg_str_options = {msg['message']}
                            
                            for i in range(K):
                                new_msg = self.generate_message(power_name)
                                if new_msg is not None and new_msg['message'] not in msg_str_options:
                                    msg_options.append(new_msg)
                                    msg_str_options.add(new_msg['message'])

                            for msg in msg_options:
                                if msg is None:
                                    continue
                                await self.chiron_agent.suggest_message(msg['recipient'], msg['message'])

                    should_stop = await self.get_should_stop()
                    randsleep = random.random()
                    await asyncio.sleep(1 + 10* randsleep)
        
                # ORDER

                if not self.has_phase_changed():
                    # keep track of our final order
                    agent_orders = self.game.get_orders(power_name)
                    self.set_comm_intent('final', agent_orders)
                    await self.send_log(f'A record of intents in {self.dipcc_current_phase}: {self.get_comm_intent()}') 

                # wait until the phase changed
                logger.info(f"Waiting until {self.dipcc_current_phase} is done")
                while not self.has_phase_changed():
                    logger.info("Still waiting")
                    await asyncio.sleep(0.5)
                
                # when the phase has changed, update submitted orders from Mila to dipcc
                if self.has_phase_changed():
                    self.phase_end_time = time.time()
                    self.update_and_process_dipcc_game()
                    self.init_phase()
                    logger.info(f"Proceeding to {self.game.get_current_phase()}")
                    await asyncio.sleep(5)
        
        if gamedir is not None and not gamedir.is_dir():
            gamedir.mkdir(parents=True, exist_ok=True)
            
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
        
            
    async def predict_opponent_moves(self, power_name: str) -> None:
        policies = self.player.get_plausible_orders_policy(self.dipcc_game)

        predicted_orders = {}
        for power, policy in policies.items():
            # Do not provide policy for the current power
            if power == power_name:
                continue

            best_orders = max(policy.items(), key=lambda x: (x[1], x))[0]
            predicted_orders[power] = best_orders

        await self.chiron_agent.suggest_opponent_orders(predicted_orders)

    async def suggest_move(self, power_name):
        agent_orders = list(self.player.get_orders(self.dipcc_game))
        if agent_orders != self.prev_suggest_moves:
            logger.info(f'Sending move advice at {round(time.time() * 1_000_000)}')
            await self.chiron_agent.suggest_orders(agent_orders)
            self.prev_suggest_moves = agent_orders
        
        # what if humans already set partial order and want to conditional on it
        cond_orders = self.game.get_orders(power_name)
        orderable_locs = self.game.get_orderable_locations(power_name=power_name)
        logger.info(f'Human has set orders: {cond_orders}; check if all orderable locations ({orderable_locs}) are set')
        if len(cond_orders) != 0 and len(cond_orders) != len(orderable_locs):
            # find if cond_orders are partially presented in bp_policy or rl_policy
            policy = self.player.get_plausible_orders_policy(self.dipcc_game)[power_name]
            logger.info(f'Searching for conditional orders using {cond_orders} in {power_name}\'s policy: {policy}')
            hit = False
            best_cond_action = None
            max_prob = 0.0
            for action, prob in policy.items():
                action = list(action)
                if all(c_order in action for c_order in cond_orders) and prob > max_prob:
                    hit = True
                    best_cond_action = action
            if hit and best_cond_action != self.prev_suggest_cond_moves:
                new_order = [action for action in best_cond_action if action not in cond_orders]
                logger.info(f'Sending move advice at {round(time.time() * 1_000_000)}')
                await self.chiron_agent.suggest_orders(new_order, partial_orders=cond_orders)
                self.prev_suggest_cond_moves = best_cond_action
            else:
                logger.info(f'Cannot find conditional orders in policy')

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

    def init_phase(self):
        """     
        update new phase to the following Dict:
        """
        self.dipcc_current_phase = self.game.get_current_phase()
        self.prev_state = 0
        self.num_stop = 0
        self.last_successful_message_time = None
        self.sent_self_intent = False
        self.reset_comm_intent()
        self.new_message = {'RUSSIA':0,'TURKEY':0,'ITALY':0,'ENGLAND':0,'FRANCE':0,'GERMANY':0,'AUSTRIA':0}
 
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

        return has_state_changed


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

        assert close_to_deadline >= 0 or deadline == 0, f"Press period is less than zero or there is no deadline: {close_to_deadline}"

        current_time = time.time()

        has_deadline = self.game.deadline > 0 
        if has_deadline and current_time - self.phase_start_time <= close_to_deadline:
            schedule = await self.game.query_schedule()
            self.scheduler_event = schedule.schedule
            server_end = self.scheduler_event.time_added + self.scheduler_event.delay
            server_remaining = server_end - self.scheduler_event.current_time
            deadline_timer = server_remaining * self.scheduler_event.time_unit
            logger.info(f'Remaining time to play: {deadline_timer} s')
        else:
            deadline = DEFAULT_DEADLINE*60

        # PRESS allows in movement phase (ONLY)
        if not self.dipcc_game.get_current_phase().endswith("M"):
            return True
        if has_deadline and current_time - self.phase_start_time >= close_to_deadline:
            return True   
        if has_deadline and deadline_timer <= no_message_second:
            return True

        return False

    async def update_press_dipcc_game(self, power_name: POWERS):
        """ 
        update new messages that present in Mila to dipcc
        """
        assert self.game.get_current_phase() == self.dipcc_current_phase, "Phase in two engines are not synchronized"

        # new messages in current phase from Mila 
        phase_messages = self.get_messages(
            messages=self.game.messages, power=power_name
        )
        most_recent_mila = self.last_received_message_time
        most_recent_sent_mila = self.last_sent_message_time

        # update message in dipcc game
        for timesent, message in phase_messages.items():
            
            # skip those that were sent to GLOBAL
            if message.recipient not in POWERS:
                continue

            self.prev_received_msg_time_sent[message.sender] = message.time_sent
            
            if (int(str(timesent)) > int(str(self.last_received_message_time)) and message.recipient == power_name) or (int(str(timesent)) > int(str(self.last_sent_message_time)) and message.sender == power_name):
                
                if message.recipient == power_name:
                    self.new_message[message.sender] = 0

                if timesent > most_recent_mila and message.recipient == power_name:
                    most_recent_mila = timesent
                elif timesent > most_recent_sent_mila and message.sender == power_name:
                    most_recent_sent_mila = timesent

                # Excluding the parentheses, check if the message only contains three upper letters.
                # If so, go through daide++. If there is an error, then send 'ERROR parsing {message}' to global,
                # and don't add it to the dipcc game.
                # If it has at least one part that contains anything other than three upper letters,
                # then just keep message body as original


                logger.info(f'Forwarding message from MILA to dipcc: {message}')

                self.dipcc_game.add_message(
                    message.sender,
                    message.recipient,
                    message.message,
                    time_sent=Timestamp.now(),
                    increment_on_collision=True,
                )

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

    def generate_message(self, power_name: POWERS)-> Optional[MessageDict]:
        """     
        call CICERO to generate message (reference from generate_message_for_approval function - webdip_api.py)
        """
        
        # get last message timesent
        if self.last_successful_message_time == None:
            self.last_successful_message_time = Timestamp.now()
        
        # timestamp condition
        last_timestamp_this_phase = self.get_last_timestamp_this_phase(default=Timestamp.now())
        sleep_time = self.player.get_sleep_time(self.dipcc_game, recipient=None)

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
        orderable_locs = self.game.get_orderable_locations(power_name=power_name)
        
        if human_intent:
            logger.info(f'Set player\'s intended moves to {tuple(human_intent)}')
            self.agent.set_power_po(human_intent)
        
        msg = self.player.generate_message(
            game=self.dipcc_game,
            timestamp=timestamp_for_conditioning,
            pseudo_orders=pseudo_orders,
        )
        
        # if human doesn't set a complete order then we won't gen message (avoid leaked move suggestion)
        if self.chiron_type == SuggestionType.MESSAGE and (human_intent is None or len(human_intent) != len(orderable_locs)):
            logger.info(f"we gen sth {msg} but still wait for human to complete order, which currently is {human_intent}")
            return None

        self.last_successful_message_time = Timestamp.now()
        return msg


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
        await self.chiron_agent.send_intent_log(log)

    def get_messages(
        self, 
        messages,
        power: POWERS,
        ):

        return {message.time_sent: message
                for message in messages.values()
                if message.recipient == power or message.sender == power}

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

            if message.recipient not in self.game.powers:
                continue

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
        default=chiron_utils.game_utils.DEFAULT_HOST,
        help="Host name of game server. (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=chiron_utils.game_utils.DEFAULT_PORT,
        help="Port of game server. (default: %(default)s)",
    )
    parser.add_argument(
        "--use-ssl",
        action="store_true",
        help="Whether to use SSL to connect to the game server. (default: %(default)s)",
    )
    parser.add_argument(
        "--game_id",
        type=str,
        required=True,
        help="ID of game to join.",
    )
    parser.add_argument(
        "--human_powers",
        nargs="+",
        choices=POWERS,
        help="Powers that we provide advice to.",
    )
    parser.add_argument(
        "--advice_levels",
        nargs="+",
        help=ADVICE_LEVELS_MESSAGE,
    )
    parser.add_argument(
        "--outdir",
        default="./fairdiplomacy_external/out",
        type=Path,
        help="Output directory for storing game JSON. (default: %(default)s)",
    )

    args = parser.parse_args()
    host: str = args.host
    port: int = args.port
    use_ssl: bool = args.use_ssl
    game_id: str = args.game_id
    outdir: Path = args.outdir
    human_powers: List[str] = args.human_powers
    advice_levels: List[str] = args.advice_levels

    logger.info(
        "Arguments:"
        f"\thost: {host}\n"
        f"\tport: {port}\n"
        f"\tuse_ssl: {use_ssl}\n"
        f"\tgame_id: {game_id}\n"
        f"\toutdir: {outdir}\n"
        f"\thuman_powers: {human_powers}\n"
        f"\tadvice_levels: {advice_levels}\n"
    )

    mila = milaWrapper()
    discord = Discord(url="https://discord.com/api/webhooks/1209977480652521522/auWUQRA8gz0HT5O7xGWIdKMkO5jE4Rby-QcvukZfx4luj_zwQeg67FEu6AXLpGTT41Qz")

    while True:
        try:
            asyncio.run(
                mila.play_mila(
                    hostname=host,
                    port=port,
                    use_ssl=use_ssl,
                    game_id=game_id,
                    gamedir=outdir,
                    human_powers=human_powers,
                    advice_levels_strs=advice_levels,
                )
            )
        except Exception as e:
            logger.exception(f"Error running {milaWrapper.play_mila.__name__}(): ")
            cicero_error = f"centaur cicero controlling {human_powers} has an error occurred: \n {e}"
            discord.post(content=cicero_error)


if __name__ == "__main__":
    main()
    
