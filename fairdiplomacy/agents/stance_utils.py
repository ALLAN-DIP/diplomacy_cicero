from diplomacy import Game as milaGame
from diplomacy.utils import strings,  common
from fairdiplomacy.pydipcc import Game
import copy
import logging
from fairdiplomacy.models.consts import POWERS


default_is_bot ={
            'AUSTRIA': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'ENGLAND': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'FRANCE': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'GERMANY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'ITALY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'RUSSIA': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'TURKEY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False}
        }
default_stance ={
            'AUSTRIA': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'ENGLAND': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'FRANCE': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'GERMANY': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'ITALY': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'RUSSIA': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0},
            'TURKEY': {'AUSTRIA': 0, 'ENGLAND': 0, 'FRANCE': 0, 'GERMANY': 0, 'ITALY': 0, 'RUSSIA': 0, 'TURKEY': 0}
        }
default_order_log ={00000:'default'}  
default_deceiving = {
            'AUSTRIA': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'ENGLAND': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'FRANCE': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'GERMANY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'ITALY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'RUSSIA': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False},
            'TURKEY': {'AUSTRIA': False, 'ENGLAND': False, 'FRANCE': False, 'GERMANY': False, 'ITALY': False, 'RUSSIA': False, 'TURKEY': False}
        }

def is_neighbor(game, agent_power, opponent):
    phase_data = game.get_phase_data()
    opponents_territories = phase_data.state["centers"][opponent] + [unit[2:5] for unit in phase_data.state["units"][opponent]]
    neighbor = False
    
    possible_orders = game.get_all_possible_orders()
    
    for unit in phase_data.state["units"][agent_power]:
        try: 
            unit_possible_orders = possible_orders[unit[2:].strip()]
        except:
            unit_possible_orders = possible_orders[unit[2:5].strip()]
        for order in unit_possible_orders:
            for loc in opponents_territories:
                if loc in order:
                    neighbor = True
                    break
    return neighbor

def predict_stance_vector_from_to(game, from_power, to_power, curr_stance_vector, orders):
    # input with dipcc game!
    #save current stance
    sim_stance_vector = copy.deepcopy(curr_stance_vector)
    
    if isinstance(game, Game):
        #if dipcc game (in the rollout)
        dipcc_game = Game(game)
        dipcc_game.clear_old_all_possible_orders()
        dipcc_game.set_orders(to_power, list(orders))
        for power in POWERS:
            if power != to_power:
                dipcc_game.set_orders(power, [])
        dipcc_game.process()
        sim_stance_vector.set_rollout_game(game=dipcc_game)
        #create just to pass input but wont be used in get stance
        #dipcc_game will be used instead since we set it earlier
        sim_game= milaGame()
        
    else:
        #milagame
        #create a sim that to_power submit "order" 
        sim_game = __game_deepcopy__(game)
        print(f'simgame phase: {sim_game.get_current_phase()} and power {to_power} units: {sim_game.powers[to_power].units} with orders: {list(orders)}')
        logging.info(f'predict new stance from {from_power} ({sim_game.powers[from_power].units}) to {to_power} with {to_power} action {list(orders)}')
        
        #submit order
        sim_game.clear_orders()
        order_dict = dict()
        # sim_game.set_orders(power_name=to_power, orders=list(orders))
        for order in list(orders):
            tokens = order.split()
            unit = '{} {}'.format(tokens[0], tokens[1])
            unit_order = ' '.join(tokens[2:])
            order_dict[unit] = unit_order
        sim_game.get_power(to_power).orders = order_dict.copy()
        # print(f'{sim_game.orders}')

        sim_game.process()

        if sim_game.get_current_phase() not in sim_game.order_log_history:
            sim_game.order_log_history.put(sim_game._phase_wrapper_type(sim_game.get_current_phase()), default_order_log)
        if sim_game.get_current_phase() not in sim_game.is_bot_history:
            sim_game.is_bot_history.put(sim_game._phase_wrapper_type(sim_game.get_current_phase()), default_is_bot)
        if sim_game.get_current_phase() not in sim_game.deceiving_history:
            sim_game.deceiving_history.put(sim_game._phase_wrapper_type(sim_game.get_current_phase()), default_deceiving)
    #retreive new stance
    predict_stance_vector = copy.deepcopy(sim_stance_vector.get_stance(sim_game))
    # print(f'{sim_game.get_current_phase()} and prev order {sim_game.order_history}')
    
    return predict_stance_vector[from_power][to_power]

def __game_deepcopy__(game: milaGame) -> None:
    """Fast deep copy implementation, from Paquette's game engine https://github.com/diplomacy/diplomacy"""
    if game.__class__.__name__ != milaGame.__name__:
        # NetworkGame
        cls = list(game.__class__.__bases__)[0]
        result = cls.__new__(cls)
        print('----networkgame----')
    else:
        # local game
        cls = game.__class__
        result = cls.__new__(cls)
        print('----localgame----')
    # Deep copying
    for key in game._slots:
        if key in [
            "map",
            "renderer",
            "powers",
            "channel",
            "notification_callbacks",
            "data",
            "__weakref__",
        ]:
            continue
        setattr(result, key, copy.deepcopy(getattr(game, key)))
    setattr(result, "map", game.map)
    setattr(result, "powers", {})
    for power in game.powers.values():
        result.powers[power.name] = copy.deepcopy(power)
        setattr(result.powers[power.name], "game", result)
    result.role = strings.SERVER_TYPE
    return result