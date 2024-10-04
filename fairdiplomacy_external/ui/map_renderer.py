#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the APGLv3 license found in the
# LICENSE file in the fairdiplomacy_external directory of this source tree.
#
""" Renderer
    Calls out to external renderer.
"""
import json
from typing import Optional

from fairdiplomacy import pydipcc

######

# This import triggers a GPL license for this file.
# As a result, we release the fairdiplomacy_external subdirectory under a GPL license,
# while releasing the rest of the repository under an MIT license.
import diplomacy.engine.renderer
import diplomacy.utils.export
from diplomacy import Game 


######

def update_past_phase(mila_game, dipcc_game: Game, phase: str):
    phase_names = dipcc_game.get_all_phase_names()
    game_phases = dipcc_game.get_all_phases()
    
    for phase_hist in game_phases:
        phase_id = phase_names.index(phase_hist.name)
        phase_order = game_phases[phase_id].orders
        
        for power, orders in phase_order.items():
            mila_game.set_orders(power, orders)
        if phase_id == phase_names.index(phase):
            break
        
        mila_game.process()

def render(
    game: pydipcc.Game, phase: Optional[str] = None, incl_abbrev=True, hide_orders: bool = False
):
    if phase is not None:
        game = game.rolled_back_to_phase_end(phase)
    if hide_orders:
        game = pydipcc.Game(game)
        game.set_all_orders({})
    
    pygame = Game()
    update_past_phase(pygame, game, phase)
    
    # game_dict= json.loads(game.to_json())
    # pygame = diplomacy.utils.export.from_saved_game_format(game_dict)

    renderer = diplomacy.engine.renderer.Renderer(pygame)
    result: str = renderer.render(incl_abbrev=incl_abbrev)
    return result
