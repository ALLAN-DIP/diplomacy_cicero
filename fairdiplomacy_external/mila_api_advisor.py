from abc import ABC
import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import time
from typing import List, Optional, Sequence
import sys
from diplomacy.utils.game_phase_data import GamePhaseData
from pprint import pprint

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
import chiron_utils.game_utils
from chiron_utils.utils import return_logger
from conf.agents_pb2 import *
from diplomacy import connect
from diplomacy.client.network_game import NetworkGame
from diplomacy.utils import strings
from diplomacy.utils.constants import SuggestionType
from diplomacy.utils.export import to_saved_game_format

from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from fairdiplomacy.agents.player import Player
from fairdiplomacy.data.build_dataset import (DATASET_DRAW_MESSAGE, DATASET_NODRAW_MESSAGE, DRAW_VOTE_TOKEN,
                                              UNDRAW_VOTE_TOKEN)
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import (
    Timestamp,
)
from fairdiplomacy.utils.game import game_from_view_of
import heyhi

logger = return_logger(__name__)

MESSAGE_DELAY_IF_SLEEP_INF = Timestamp.from_seconds(60)

DEFAULT_DEADLINE = 5



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
    suggestion_type = SuggestionType.MOVE | SuggestionType.COMMENTARY


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
        
        agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/diplodocus_high.prototxt')
        logger.info("Successfully loaded CICERO config")

        self.agent = BQRE1PAgent(agent_config.bqre1p)
    
    def get_curr_power_to_advise(self):
        return self.power_to_advise

    def get_curr_advice_level(self):
        return self.advice_level

    async def play_mila(self, args) -> None:
        hostname = args.host
        port = args.port
        use_ssl = args.use_ssl
        game_id = args.game_id
        power = args.power
        power_name = None

        logger.info("Settings:")
        logger.info(
            f"hostname: {hostname}, "
            f"port: {port}, "
            f"use_ssl: {use_ssl}, "
            f"game_id: {game_id}, "
            f"player to advise: {power}, "
        )

        connection = await connect(hostname, port, use_ssl)
        channel = await connection.authenticate(
            f"admin", "password"
        )
        self.game: NetworkGame = await channel.join_game(game_id=game_id)

        self.chiron_agent = CiceroAdvisor(power_name, self.game)

        # Wait while game is still being formed
        logger.info("Waiting for game to start")
        while self.game.is_game_forming:
            await asyncio.sleep(2)
            logger.info("Still waiting")

        # Playing game
        logger.info("Started playing")
        current_phase = None
        current_units = None
        current_homes = None
        current_centers = None

        while not self.game.is_game_done and not self.game.powers[power].is_eliminated():
            paquette_game = self.game.get_phase_data()
            game_state = GamePhaseData.to_dict(paquette_game)
            phase_name = game_state["name"]
            
            
            
            if phase_name == "COMPLETED" or self.game.status in ["completed", "paused"]:
                sys.exit(0)

            if current_phase is None or current_phase != phase_name:
                current_phase = phase_name
                logger.info(f"New phase: {current_phase}")
                prev_power_stance = None

                current_units = game_state["state"]["units"]
                if current_homes is None:
                    current_homes = game_state["state"]["homes"]
                current_centers = game_state["state"]["centers"]

                await self.chiron_agent.declare_suggestion_type()

                dipcc_game = Game()
                dipcc_json = json.loads(dipcc_game.to_json())
                

            await self.game.synchronize()
            msgs = self.game.messages

            if msgs:
                mms = msgs.sub()
                advice_requests = [x for x in mms if x.sender == power and x.recipient == "GLOBAL"]

            if len(advice_requests):
                last_request = max(advice_requests, key=lambda x: x.time_sent)
            
                try:
                    stance = json.loads(last_request.message)
                    ally_powers = [k for k, v in stance.items() if v > 0]
                    logger.info(f"latest stance: {stance}")

                    if prev_power_stance is None or set(stance) != set(prev_power_stance):
                        logger.info(f"Stance changed, sending {ally_powers}")
                        prev_power_stance = stance

                        # alter supply center and units
                        stance_unit = {}
                        stance_center = {}

                        for pp, units in current_units.items():
                            stance_unit[pp] = []
                            if pp in ally_powers or pp == power:
                                stance_unit[power].extend(units)
                            else:
                                stance_unit[pp].extend(units)

                        for pp, centers in current_centers.items():
                            stance_center[pp] = []
                            if pp in ally_powers or pp == power:
                                stance_center[power].extend(centers)
                            else:
                                stance_center[pp].extend(centers)


                        dipcc_json["phases"][0]["state"]["units"] = stance_unit
                        dipcc_json["phases"][0]["state"]["homes"] = current_homes
                        dipcc_json["phases"][0]["state"]["centers"] = stance_center
                        logger.debug(dipcc_json["phases"][0]["state"]["units"])
                        logger.debug(dipcc_json["phases"][0]["state"]["centers"])
                        dipcc_game = Game.from_json(json.dumps(dipcc_json))

                        cicero_player = Player(self.agent, power)
                        cicero_policy = cicero_player.agent.get_plausible_orders_policy(game=dipcc_game, agent_power=power, agent_state=cicero_player.state)
                        logger.info(f"Policy: {cicero_policy}")
                        sys.exit(0)
                        

                except json.JSONDecodeError:
                    logging.error("Invalid JSON")


            if current_phase[-1] != "M":
                await asyncio.sleep(1)
                continue

            await asyncio.sleep(1)


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
    
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        type=str,
        default=chiron_utils.game_utils.DEFAULT_HOST,
        help="host IP address (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=chiron_utils.game_utils.DEFAULT_PORT,
        help="port to connect to the game (default: %(default)s)",
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
        help="game id of game created in DATC diplomacy game",
    )
    parser.add_argument(
        "--outdir", default= "./fairdiplomacy_external/out", type=Path, help="output directory for game json to be stored"
    )
    parser.add_argument(
        "--power", default= "", type=str, help="power of human player to advise"
    )


    args = parser.parse_args()

    mila = milaWrapper()

    while True:
        try:
            asyncio.run(
                mila.play_mila(args)
            )
        except Exception as e:
            logger.exception(f"Error running {milaWrapper.play_mila.__name__}(): ")


if __name__ == "__main__":
    main()
    
