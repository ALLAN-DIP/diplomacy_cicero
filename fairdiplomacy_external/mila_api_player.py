from abc import ABC
import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import math
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
from fairdiplomacy.utils.typedefs import get_last_message
import heyhi
from parlai_diplomacy.wrappers.classifiers import INF_SLEEP_TIME

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
class CiceroPlayer(CiceroBot):
    """Player form of `CiceroBot`."""

    bot_type = BotType.PLAYER


class milaWrapper:

    def __init__(self):
        self.game: NetworkGame = None
        self.chiron_agent: Optional[CiceroBot] = None
        self.dipcc_game: Game = None
        self.prev_state = 0                                         # number of number received messages in the current phase
        self.dialogue_state = {}                                    # {phase: number of all (= received + new) messages for agent}
        self.last_received_message_time = 0                         # {phase: time_sent (from Mila json)}                                      
        self.dipcc_current_phase = None                             
        self.last_successful_message_time = None                    # timestep for last message successfully sent in the current phase                       
        self.reuse_stale_pseudo_after_n_seconds = 45                # seconds to reuse pseudo order to generate message
        self.last_comm_intent={'RUSSIA':None,'TURKEY':None,'ITALY':None,'ENGLAND':None,'FRANCE':None,'GERMANY':None,'AUSTRIA':None,'final':None}
        self.prev_received_msg_time_sent = {'RUSSIA':None,'TURKEY':None,'ITALY':None,'ENGLAND':None,'FRANCE':None,'GERMANY':None,'AUSTRIA':None}
        
        agent_config = heyhi.load_config('/diplomacy_cicero/conf/common/agents/cicero.prototxt')
        logger.info(f"successfully load cicero config")

        self.agent = PyBQRE1PAgent(agent_config.bqre1p)

    async def play_mila(
        self,
        hostname: str,
        port: int,
        use_ssl: bool,
        game_id: str,
        power_name: str,
        gamedir: Path,
        first_submit_time_deadline: int,
    ) -> None:
        logger.info(f"Cicero joining game: {game_id} as {power_name}")
        connection = await connect(hostname, port, use_ssl)
        channel = await connection.authenticate(
            f"cicero_{power_name}", "password"
        )
        self.game: NetworkGame = await channel.join_game(game_id=game_id, power_name=power_name, player_type=CiceroPlayer.player_type)
        
        self.chiron_agent = CiceroPlayer(power_name, self.game)
        
        self.first_submit_time_deadline = first_submit_time_deadline

        # Wait while game is still being formed
        logger.info(f"Waiting for game to start")
        while self.game.is_game_forming:
            await asyncio.sleep(2)
            logger.info("Still waiting")

        # Playing game
        logger.info(f"Started playing")

        self.dipcc_game = self.start_dipcc_game(power_name)
        logger.info(f"Started dipcc game")

        self.player = Player(self.agent, power_name)
        self.power_name = power_name
        
        logging.basicConfig(filename=f'/diplomacy_cicero/fairdiplomacy_external/{game_id}_{power_name}.log', format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

        while not self.game.is_game_done:
            self.phase_start_time = time.time()
            self.dipcc_current_phase = self.dipcc_game.get_current_phase()
            self.presubmit = False

            # fix issue that there is a chance where retreat phase appears in dipcc but not mila 
            while self.has_phase_changed():
                await self.send_log(f'process dipcc game {self.dipcc_current_phase} to catch up with a current phase in mila {self.game.get_current_phase()}') 
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

                # PRESS allows in movement phase (ONLY)
                if self.dipcc_game.get_current_phase().endswith("M"):
                    await self.chiron_agent.wait_for_comm_stage()

                # PRESS
                has_deadline = self.game.deadline > 0 
                should_stop = await self.get_should_stop()
                while not should_stop :
                    
                    if has_deadline:
                        # if times almost up but still can do some press, let's presubmit order
                        should_presubmit = await self.get_should_presubmit()
                        if should_presubmit and not self.presubmit:
                            self.presubmit = True
                            logger.info(f"Pre-submit orders in {self.dipcc_current_phase}")
                            agent_orders = self.player.get_orders(self.dipcc_game)
                            await self.chiron_agent.send_orders(agent_orders, wait=True)

                    # if there is new message incoming
                    if self.has_state_changed(power_name):
                        # update press in dipcc
                        await self.update_press_dipcc_game(power_name)

                    # reply/gen new message
                    msg = self.generate_message(power_name)
                    logger.info(f'msg from cicero to dipcc {msg}')
                    
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
                        
                        if not self.sent_self_intent:
                            self_pseudo_log = f'At the start of this phase, I intend to do: {self_po}'
                            await self.send_log(self_pseudo_log) 
                            self.sent_self_intent = True

                        # keep track of intent that we talked to each recipient
                        self.set_comm_intent(recipient_power, power_po)

                        await self.send_message(msg, 'dipcc')
                        await self.send_message(msg, 'mila')

                        self_pseudo_log = f'After I got the message (prev msg time_sent: {self.prev_received_msg_time_sent[msg["recipient"]]}) from {recipient_power}. \
                            My response is {msg["message"]}. I intend to do: {self_po}. I expect {recipient_power} to do: {recp_po}.'
                        await self.send_log(self_pseudo_log) 

                        if 'deceptive' in msg:
                            await self.send_log(msg['deceptive'])
                            logger.info(f'Cicero logs if message is deceptive: {msg["deceptive"]}')
                           
                    should_stop = await self.get_should_stop()
                    randsleep = random.random()
                    await asyncio.sleep(1 + 10* randsleep)
        
                # ORDER

                if not self.has_phase_changed():
                    logger.info(f"Submit orders in {self.dipcc_current_phase}")
                    agent_orders = self.player.get_orders(self.dipcc_game)

                    # keep track of our final order
                    self.set_comm_intent('final', agent_orders)
                    await self.send_log(f'A record of intents in {self.dipcc_current_phase}: {self.get_comm_intent()}') 

                    # set order in Mila
                    await self.chiron_agent.send_orders(agent_orders, wait=False)

                # wait until the phase changed
                logger.info(f"wait until {self.dipcc_current_phase} is done")
                while not self.has_phase_changed():
                    logger.info(".")
                    await asyncio.sleep(0.5)
                
                # when the phase has changed, update submitted orders from Mila to dipcc
                if self.has_phase_changed():
                    self.phase_end_time = time.time()
                    self.update_and_process_dipcc_game()
                    self.init_phase()
                    logger.info(f"Process to {self.game.get_current_phase()}")
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

    async def get_should_presubmit(self)->bool:
        schedule = await self.game.query_schedule()
        self.scheduler_event = schedule.schedule
        server_end = self.scheduler_event.time_added + self.scheduler_event.delay
        server_remaining = server_end - self.scheduler_event.current_time
        deadline_timer = server_remaining * self.scheduler_event.time_unit

        presubmit_second = self.first_submit_time_deadline

        if deadline_timer <= presubmit_second:
            logger.info(f'time to presubmit order')
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

        assert close_to_deadline >= 0 or deadline == 0, f"Press period is less than zero or there is no deadline: {close_to_deadline}"

        current_time = time.time()

        has_deadline = self.game.deadline > 0 
        if has_deadline and current_time - self.phase_start_time <= close_to_deadline:
            schedule = await self.game.query_schedule()
            self.scheduler_event = schedule.schedule
            server_end = self.scheduler_event.time_added + self.scheduler_event.delay
            server_remaining = server_end - self.scheduler_event.current_time
            deadline_timer = server_remaining * self.scheduler_event.time_unit
            logger.info(f'remaining time to play: {deadline_timer}')
        else:
            deadline = DEFAULT_DEADLINE*60

        # PRESS allows in movement phase (ONLY)
        if not self.dipcc_game.get_current_phase().endswith("M"):
            return True
        if has_deadline and current_time - self.phase_start_time >= close_to_deadline:
            return True   
        if has_deadline and deadline_timer <= no_message_second:
            return True
        if self.last_received_message_time != 0 and current_time - self.last_received_message_time >=no_message_second:
            logger.info(f'no incoming message for {current_time - self.last_received_message_time} seconds')
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
        
        # new messages in current phase from Mila 
        phase_messages = self.get_messages(
            messages=self.game.messages, power=power_name
        )
        most_recent_mila = self.last_received_message_time

        # update message in dipcc game
        for timesent, message in phase_messages.items():

            if message.recipient != power_name:
                continue
            
            self.prev_received_msg_time_sent[message.sender] = message.time_sent
            if int(str(timesent)[0:10]) > int(str(self.last_received_message_time)[0:10]):
                if timesent > most_recent_mila:
                    most_recent_mila = timesent

                # if the message is english, just send it to dipcc recipient
                logger.info(f'parising message from mila to dipcc {message}')
                self.dipcc_game.add_message(
                    message.sender,
                    message.recipient,
                    message.message,
                    time_sent=Timestamp.now(),
                    increment_on_collision=True,
                )

        # update last_received_message_time 
        self.last_received_message_time = most_recent_mila

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

        sleep_time_for_conditioning = (
            sleep_time if sleep_time < INF_SLEEP_TIME else MESSAGE_DELAY_IF_SLEEP_INF
        )

        if get_last_message(self.dipcc_game) is None:
            timestamp_for_conditioning = sleep_time_for_conditioning
        else:
            timestamp_for_conditioning = last_timestamp_this_phase + sleep_time_for_conditioning

        # generate message using pseudo orders
        pseudo_orders = None

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
        await self.chiron_agent.send_intent_log(log)

    async def send_message(self, msg: MessageDict, engine: str) -> None:
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
            await self.chiron_agent.send_message(
                recipient=msg["recipient"],
                message=msg["message"],
                sender=msg["sender"],
            )

        logger.info(f'update a message in {engine}, {msg["sender"] }->{ msg["recipient"]}: {msg["message"]}')

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

            if message.recipient not in self.game.powers:
                continue

            # if the message is english, just send it to dipcc recipient
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
        "--power",
        choices=POWERS,
        required=True,
        help="Name of power to play as.",
    )
    parser.add_argument(
        "--outdir",
        default="./fairdiplomacy_external/out",
        type=Path,
        help="Output directory for storing game JSON. (default: %(default)s)",
    )
    parser.add_argument(
        "--first_submit_time_deadline",
        type=int,
        default=7200,
        help="Deadline for the first submit time in seconds. (default: %(default)s)",
    )
    args = parser.parse_args()
    host: str = args.host
    port: int = args.port
    use_ssl: bool = args.use_ssl
    game_id: str = args.game_id
    power: str = args.power
    outdir: Path = args.outdir
    first_submit_time_deadline: int = args.first_submit_time_deadline

    logger.info(
        "Arguments:\n"
        f"\thost: {host}\n"
        f"\tport: {port}\n"
        f"\tuse_ssl: {use_ssl}\n"
        f"\tgame_id: {game_id}\n"
        f"\tpower: {power}\n"
        f"\toutdir: {outdir}\n"
        f"\tfirst_submit_time_deadline: {first_submit_time_deadline}\n"
    )

    if outdir is not None and not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    mila = milaWrapper()
    discord = Discord(url="https://discord.com/api/webhooks/1209977480652521522/auWUQRA8gz0HT5O7xGWIdKMkO5jE4Rby-QcvukZfx4luj_zwQeg67FEu6AXLpGTT41Qz")
    discord.post(content=f"Cicero as power {power} is joining {game_id}.")

    while True:
        try:
            asyncio.run(
                mila.play_mila(
                    hostname=host,
                    port=port,
                    use_ssl=use_ssl,
                    game_id=game_id,
                    power_name=power,
                    gamedir=outdir,
                    first_submit_time_deadline=first_submit_time_deadline,
                )
                    )
        except Exception as e:
            logger.exception(f"Error running {milaWrapper.play_mila.__name__}(): ")
            cicero_error = f"cicero controlling {power} has an error occured: \n {e}"
            discord.post(content=cicero_error)


if __name__ == "__main__":
    main()
    
