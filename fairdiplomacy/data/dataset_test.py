#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest

import nest
import numpy as np
import torch
import json

from conf import conf_cfgs
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.data.dataset import (
    maybe_augment_targets_inplace,
    shuffle_locations,
)
from fairdiplomacy.models.consts import POWERS, MAX_SEQ_LEN, LOCS, N_SCS
from fairdiplomacy.models.state_space import EOS_IDX
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.utils.order_idxs import action_strs_to_global_idxs, global_order_idxs_to_local, global_order_idxs_to_str
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder, DEFAULT_INPUT_VERSION
from fairdiplomacy.data.dataset import encode_phase
from fairdiplomacy.utils.tensorlist import TensorList



class LocationShuffleTest(unittest.TestCase):
    def testAllPowers(self):
        encoder = FeatureEncoder()
        features = encoder.encode_inputs([Game()] * 6, input_version=1)
        print(nest.map(lambda x: x.shape, features))
        shuffle_locations(features)

    def testSinglePower(self):
        encoder = FeatureEncoder()
        features = encoder.encode_inputs([Game()] * 6, input_version=1)
        # Removing power dimension and predenting everything is just a batch.
        for name in ["x_build_numbers", "x_loc_idxs", "x_possible_actions"]:
            features[name] = features[name][:, 0]
        print(nest.map(lambda x: x.shape, features))
        shuffle_locations(features)


class AugmentAllPowerTest(unittest.TestCase):
    def testAugmentCondition(self):
        encoder = FeatureEncoder()
        features = encoder.encode_inputs_all_powers([Game()], input_version=1)
        power_orders = [
            "A VIE - GAL, F TRI - ALB, A BUD - SER".split(", "),
            "F EDI - NWG, F LON - NTH, A LVP - EDI".split(", "),
            "F BRE - MAO, A PAR - BUR, A MAR S A PAR - BUR".split(", "),
            "F KIE - DEN, A MUN - RUH, A BER - KIE".split(", "),
            "F NAP - ION, A ROM - APU, A VEN H".split(", "),
            "F STP/SC - BOT, A MOS - UKR, A WAR - GAL, F SEV - BLA".split(", "),
            "F ANK - BLA, A SMY - CON, A CON - BUL".split(", "),
        ]
        joint_action = sum(power_orders, [])

        action_ids = action_strs_to_global_idxs(joint_action, sort_by_loc=True)
        global_actions = torch.full([1, 7, 34], EOS_IDX, dtype=torch.long)
        global_actions[0, 0, : len(action_ids)] = torch.LongTensor(action_ids)
        y_actions = global_order_idxs_to_local(global_actions, features["x_possible_actions"],)

        batch = DataFields(**features, y_actions=y_actions)
        print(nest.map(lambda x: x.shape, batch))

        power_conditioning_cfg = conf_cfgs.PowerConditioning(
            prob=0.5, min_num_power=1, max_num_power=2
        )

        for i in range(100):
            np.random.seed(i)
            maybe_augment_targets_inplace(
                batch,
                single_chances=None,
                double_chances=None,
                power_conditioning=power_conditioning_cfg,
            )

def testOneGameOnePhaseStance(power1,power2,power1_order,power2_order, p1p2stance,p2p1stance2,gamefile=None,phase_idx=0,phase='S1901M'):
    #create a game, do some actions
    if gamefile != None:
        with open(gamefile, 'r') as f:
            dipcc_json = f.read()
        dipcc_game = Game.from_json(dipcc_json)
        units = dipcc_game.get_phase_history()[phase_idx].state['units']
        units = units[power1] + units[power2]
        print(f'allunits: {units}')
    else:
        dipcc_game = Game()
        for power in POWERS:
            if power not in [power1, power2]:
                dipcc_game.set_orders(power, [])
            else:
                if power == power1:
                    dipcc_game.set_orders(power1, power1_order)
                else:
                    dipcc_game.set_orders(power2, power2_order)
        dipcc_game.process()
        # save to json and force edit stance vectors 
        dipcc_json = json.loads(dipcc_game.to_json())
        # print(dipcc_json)
        first_phase = dipcc_json['phases'][phase_idx]
        units = first_phase['state']['units'][power1] + first_phase['state']['units'][power2]
        first_phase['stance_vectors'] = {power1: {power2: 0.05 for power2 in POWERS} for power1 in POWERS}
        first_phase['stance_vectors'][power1][power2] = p1p2stance
        first_phase['stance_vectors'][power2][power1] = p2p1stance2
        dipcc_json = json.dumps(dipcc_json)
        dipcc_game = Game.from_json(dipcc_json)
        
    encoder = FeatureEncoder()
    #call encode phase
    phase_encoding = encode_phase(
        encoder,
        dipcc_game,
        dipcc_json,
        'test_game',
        phase_idx,
        phase,
        only_with_min_final_score=None,
        input_valid_power_idxs=[False]*7,
        exclude_n_holds=3,
        all_powers=False,
        input_version=DEFAULT_INPUT_VERSION,
        return_hold_for_invalid=False,
    )
    #get possible orders and expected stance (pre-calculated)
    #check output vs expected stance
    for i, global_idxs in enumerate(phase_encoding['x_possible_actions']):
        possible_action_list = global_order_idxs_to_str(global_idxs)
        if len(possible_action_list) != 0:
            if any(p in possible_action_list[0] for p in units):
                print(possible_action_list)
                print(phase_encoding['stance_weights'][i])
    # print(phase_encoding['stance_weights'])
    return phase_encoding
    

# test first phase with AUS and ITA / friend and foes
# testOneGameOnePhaseStance('AUSTRIA',
#                           'ITALY',
#                           ['A VIE - GAL','F TRI - ALB', 'A BUD - SER'],
#                           ['A ROM - APU','F NAP - ION', 'A VEN S F TRI'],
#                           0.5,
#                           -0.5,)

# test load with FRA and GER / both are foes expect 0.05/0.1
# testOneGameOnePhaseStance('FRANCE',
#                           'GERMANY',
#                           ["A BEL H",
#                             "F PIC S A BEL",
#                             "A PAR - BUR",
#                             "A SPA - POR"],
#                           ["A DEN S F SWE",
#                             "F HOL S A RUH - BEL",
#                             "A BUR - PIC",
#                             "A RUH - BEL"],
#                           0.5,
#                           -0.5,
#                           '/data/games_stance/game_111.json',4, 'F1902M')
# testTwoshuffle('/data/games_stance/game_111.json')