from stance_vector import ActionBasedStance, ScoreBasedStance
from copy import deepcopy
from enum import Enum, auto
from itertools import product
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union, overload

from diplomacy.utils import strings
from typing_extensions import Literal
from diplomacy import Game, GamePhaseData

class FlipReason(str, Enum):
    """Reasons for a betrayal."""

    END_GAME = auto()
    RANDOM = auto()

class CiceroStance(ActionBasedStance):
    """
    A turn-level action-based objective stance vector baseline
    "Whoever attacks me is my enemy, whoever supports me is my friend."
    Stance on nation k =  discount* Stance on nation k
        - alpha1 * count(k's hostile moves)
        - alpha2 * count(k's conflict moves)
        - beta1 * k's count(hostile supports/convoys)
        - beta2 * count(k's conflict supports/convoys)
        + gamma1 * count(k's friendly supports/convoys)
        + gamma2 * count(k's unrealized hostile moves)
    """

    alpha1: float
    alpha2: float
    discount: float
    beta1: float
    beta2: float
    gamma1: float
    gamma2: float
    end_game_flip: bool
    year_threshold: int
    random_betrayal: bool
    random: random.Random

    def __init__(
        self,
        my_identity: str,
        game: Game,
        invasion_coef: float = 1.0,
        conflict_coef: float =1.0,
        invasive_support_coef: float = 1.0,
        conflict_support_coef: float = 1.0,
        friendly_coef: float = 1.0,
        unrealized_coef: float = 0.5,
        discount_factor: float = 0.5,
        end_game_flip: bool = False,
        year_threshold: int = 1910,
        random_betrayal: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        
        super().__init__(my_identity, game)
        # hyperparameters weighting different actions
        self.alpha1 = invasion_coef
        self.alpha2 = conflict_coef
        self.discount = discount_factor
        self.beta1 = invasive_support_coef
        self.beta2 = conflict_support_coef
        self.gamma1 = friendly_coef
        self.gamma2 = unrealized_coef
        self.end_game_flip = end_game_flip
        self.year_threshold = year_threshold
        self.random_betrayal = random_betrayal
        self.random = random.Random(random_seed)
        self.mphase=None
        self.is_rollout = False
        self.rollout_dipcc_game = None
    
    def set_mphase(self, phase):
        self.mphase = phase
    
    def set_rollout(self, is_rollout):
        self.is_rollout = is_rollout
    
    def set_rollout_game(self, game):
        self.rollout_dipcc_game = game
        
    def get_prev_m_phase(self):
        # print('in cicero_stance get prev m phase')
        if self.mphase is not None and self.game is not None:
            # print(f'in this m phase {self.mphase}')
            phase_data = self.game.get_phase_from_history(self.mphase, game_role=self.game.role)
            # print(f'phase orders {self.mphase}: {phase_data.orders}')
            return phase_data
        
        game_to_get_prev_m_phase = None

        # if is_rollout then return dipccgame phase data
        if self.is_rollout and self.rollout_dipcc_game is not None:
            game_to_get_prev_m_phase = self.rollout_dipcc_game
            # print(f'get prev m from dippcc game')
        else:
            game_to_get_prev_m_phase = self.game
            
        phase_hist = game_to_get_prev_m_phase.get_phase_history()
        prev_m_phase = None
        for phase_data in reversed(phase_hist):
            if phase_data.name.endswith("M"):
                prev_m_phase = phase_data
                break
        if prev_m_phase is None:
            prev_m_phase = game_to_get_prev_m_phase.get_phase_data()
            # Remove private messages between other powers
            # prev_m_phase.messages = self.game.filter_messages(prev_m_phase.messages, self.game.role)
        # print(f'get prev_m_phase : {prev_m_phase.name}')
        return prev_m_phase
        
def test_stance_vector_dipcc():
    from fairdiplomacy.pydipcc import Game as CiceroGame
    # start mila_game
    game = Game()
    stance_vector = CiceroStance('AUSTRIA' ,game,conflict_coef=1.0, conflict_support_coef=1.0, unrealized_coef = 0.0
                , discount_factor=1.0, random_betrayal=False)
    print(f'stance vector before game starting : {stance_vector.stance}')
    
    stance_vector.is_rollout = True
    dipcc_game = CiceroGame()
    dipcc_game.set_orders('AUSTRIA', ['A VIE - GAL','F TRI - ALB', 'A BUD - SER'])
    dipcc_game.set_orders('ITALY', ['A ROM - APU','F NAP - ION', 'A VEN - TRI'])
    dipcc_game.set_orders('ENGLAND', [])
    dipcc_game.set_orders('FRANCE', [])
    dipcc_game.set_orders('GERMANY', [])
    dipcc_game.set_orders('RUSSIA', [])
    dipcc_game.set_orders('TURKEY', [])
    dipcc_game.process()
    stance_vector.rollout_dipcc_game = dipcc_game
    
    stance_vector.get_stance(game)
    print(f'stance vector at starting of {dipcc_game.get_current_phase()} expecting AUS->ITA to decrease: {stance_vector.stance}')
    