from daidepp import create_daide_grammar, AND, ORR, XDO, PRP, FCT, MTO, HLD, RTO, SUP, Location, DAIDEVisitor
import random
from types import List
from fairdiplomacy.typedefs import MessageDict

"""
Functions for randomizing orders using the daidepp library
"""

__author__ = "Konstantine Kahadze"
__email__ = "konstantinekahadze@gmail.com"

# The comments below signal the formatter not to expand these dicts to multiple lines
# fmt: off

# This dictionary represents every adjacent province and coast from any given province or coast
ADJACENCY = {'ADR': ['ALB', 'APU', 'ION', 'TRI', 'VEN'], 'AEG': ['BUL/SC', 'CON', 'EAS', 'GRE', 'ION', 'SMY'], 'ALB': ['ADR', 'GRE', 'ION', 'SER', 'TRI'], 'ANK': ['ARM', 'BLA', 'CON', 'SMY'], 'APU': ['ADR', 'ION', 'NAP', 'ROM', 'VEN'], 'ARM': ['ANK', 'BLA', 'SEV', 'SMY', 'SYR'], 'BAL': ['BER', 'BOT', 'DEN', 'LVN', 'KIE', 'PRU', 'SWE'], 'BAR': ['NWY', 'NWG', 'STP/NC'], 'BEL': ['BUR', 'ENG', 'HOL', 'NTH', 'PIC', 'RUH'], 'BER': ['BAL', 'KIE', 'MUN', 'PRU', 'SIL'], 'BLA': ['ANK', 'ARM', 'BUL/EC', 'CON', 'RUM', 'SEV'], 'BOH': ['GAL', 'MUN', 'SIL', 'TYR', 'VIE'], 'BOT': ['BAL', 'FIN', 'LVN', 'STP/SC', 'SWE'], 'BRE': ['ENG', 'GAS', 'MAO', 'PAR', 'PIC'], 'BUD': ['GAL', 'RUM', 'SER', 'TRI', 'VIE'], 'BUL/EC': ['BLA', 'CON', 'RUM'], 'BUL/SC': ['AEG', 'CON', 'GRE'], 'BUL': ['AEG', 'BLA', 'CON', 'GRE', 'RUM', 'SER'], 'BUR': ['BEL', 'GAS', 'RUH', 'MAR', 'MUN', 'PAR', 'PIC', 'SWI'], 'CLY': ['EDI', 'LVP', 'NAO', 'NWG'], 'CON': ['AEG', 'BUL/EC', 'BUL/SC', 'BLA', 'ANK', 'SMY'], 'DEN': ['BAL', 'HEL', 'KIE', 'NTH', 'SKA', 'SWE'], 'EAS': ['AEG', 'ION', 'SMY', 'SYR'], 'EDI': ['CLY', 'LVP', 'NTH', 'NWG', 'YOR'], 'ENG': ['BEL', 'BRE', 'IRI', 'LON', 'MAO', 'NTH', 'PIC', 'WAL'], 'FIN': ['BOT', 'NWY', 'STP/SC', 'SWE'], 'GAL': ['BOH', 'BUD', 'RUM', 'SIL', 'UKR', 'VIE', 'WAR'], 'GAS': ['BUR', 'BRE', 'MAO', 'MAR', 'PAR', 'SPA/NC'], 'GRE': ['AEG', 'ALB', 'BUL/SC', 'ION', 'SER'], 'HEL': ['DEN', 'HOL', 'KIE', 'NTH'], 'HOL': ['BEL', 'HEL', 'KIE', 'NTH', 'RUH'], 'ION': ['ADR', 'AEG', 'ALB', 'APU', 'EAS', 'GRE', 'NAP', 'TUN', 'TYS'], 'IRI': ['ENG', 'LVP', 'MAO', 'NAO', 'WAL'], 'KIE': ['BAL', 'BER', 'DEN', 'HEL', 'HOL', 'MUN', 'RUH'], 'LON': ['ENG', 'NTH', 'YOR', 'WAL'], 'LVN': ['BAL', 'BOT', 'MOS', 'PRU', 'STP/SC', 'WAR'], 'LVP': ['CLY', 'EDI', 'IRI', 'NAO', 'WAL', 'YOR'], 'LYO': ['MAR', 'PIE', 'SPA/SC', 'TUS', 'TYS', 'WES'], 'MAO': ['BRE', 'ENG', 'GAS', 'IRI', 'NAF', 'NAO', 'POR', 'SPA/NC', 'SPA/SC', 'WES'], 'MAR': ['BUR', 'GAS', 'LYO', 'PIE', 'SPA/SC', 'SWI'], 'MOS': ['LVN', 'SEV', 'STP', 'UKR', 'WAR'], 'MUN': ['BER', 'BOH', 'BUR', 'KIE', 'RUH', 'SIL', 'TYR', 'SWI'], 'NAF': ['MAO', 'TUN', 'WES'], 'NAO': ['CLY', 'IRI', 'LVP', 'MAO', 'NWG'], 'NAP': ['APU', 'ION', 'ROM', 'TYS'], 'NWY': ['BAR', 'FIN', 'NTH', 'NWG', 'SKA', 'STP/NC', 'SWE'], 'NTH': ['BEL', 'DEN', 'EDI', 'ENG', 'LON', 'HEL', 'HOL', 'NWY', 'NWG', 'SKA', 'YOR'], 'NWG': ['BAR', 'CLY', 'EDI', 'NAO', 'NWY', 'NTH'], 'PAR': ['BUR', 'BRE', 'GAS', 'PIC'], 'PIC': ['BEL', 'BRE', 'BUR', 'ENG', 'PAR'], 'PIE': ['LYO', 'MAR', 'TUS', 'TYR', 'VEN', 'SWI'], 'POR': ['MAO', 'SPA/NC', 'SPA/SC'], 'PRU': ['BAL', 'BER', 'LVN', 'SIL', 'WAR'], 'ROM': ['APU', 'NAP', 'TUS', 'TYS', 'VEN'], 'RUH': ['BEL', 'BUR', 'HOL', 'KIE', 'MUN'], 'RUM': ['BLA', 'BUD', 'BUL/EC', 'GAL', 'SER', 'SEV', 'UKR'], 'SER': ['ALB', 'BUD', 'BUL', 'GRE', 'RUM', 'TRI'], 'SEV': ['ARM', 'BLA', 'MOS', 'RUM', 'UKR'], 'SIL': ['BER', 'BOH', 'GAL', 'MUN', 'PRU', 'WAR'], 'SKA': ['DEN', 'NWY', 'NTH', 'SWE'], 'SMY': ['AEG', 'ANK', 'ARM', 'CON', 'EAS', 'SYR'], 'SPA/NC': ['GAS', 'MAO', 'POR'], 'SPA/SC': ['LYO', 'MAO', 'MAR', 'POR', 'WES'], 'SPA': ['GAS', 'LYO', 'MAO', 'MAR', 'POR', 'WES'], 'STP/NC': ['BAR', 'NWY'], 'STP/SC': ['BOT', 'FIN', 'LVN'], 'STP': ['BAR', 'BOT', 'FIN', 'LVN', 'MOS', 'NWY'], 'SWE': ['BAL', 'BOT', 'DEN', 'FIN', 'NWY', 'SKA'], 'SYR': ['ARM', 'EAS', 'SMY'], 'TRI': ['ADR', 'ALB', 'BUD', 'SER', 'TYR', 'VEN', 'VIE'], 'TUN': ['ION', 'NAF', 'TYS', 'WES'], 'TUS': ['LYO', 'PIE', 'ROM', 'TYS', 'VEN'], 'TYR': ['BOH', 'MUN', 'PIE', 'TRI', 'VEN', 'VIE', 'SWI'], 'TYS': ['ION', 'LYO', 'ROM', 'NAP', 'TUN', 'TUS', 'WES'], 'UKR': ['GAL', 'MOS', 'RUM', 'SEV', 'WAR'], 'VEN': ['ADR', 'APU', 'PIE', 'ROM', 'TRI', 'TUS', 'TYR'], 'VIE': ['BOH', 'BUD', 'GAL', 'TRI', 'TYR'], 'WAL': ['ENG', 'IRI', 'LON', 'LVP', 'YOR'], 'WAR': ['GAL', 'LVN', 'MOS', 'PRU', 'SIL', 'UKR'], 'WES': ['MAO', 'LYO', 'NAF', 'SPA/SC', 'TUN', 'TYS'], 'YOR': ['EDI', 'LON', 'LVP', 'NTH', 'WAL'], 'SWI': ['MAR', 'BUR', 'MUN', 'TYR', 'PIE']}

# This dict defines the type of every province. Every province is either "COAST", "WATER", "LAND" or "SHUT"
TYPES = {'ADR': 'WATER', 'AEG': 'WATER', 'ALB': 'COAST', 'ANK': 'COAST', 'APU': 'COAST', 'ARM': 'COAST', 'BAL': 'WATER', 'BAR': 'WATER', 'BEL': 'COAST', 'BER': 'COAST', 'BLA': 'WATER', 'BOH': 'LAND', 'BOT': 'WATER', 'BRE': 'COAST', 'BUD': 'LAND', 'BUL/EC': 'COAST', 'BUL/SC': 'COAST', 'bul': 'COAST', 'BUR': 'LAND', 'CLY': 'COAST', 'CON': 'COAST', 'DEN': 'COAST', 'EAS': 'WATER', 'EDI': 'COAST', 'ENG': 'WATER', 'FIN': 'COAST', 'GAL': 'LAND', 'GAS': 'COAST', 'GRE': 'COAST', 'HEL': 'WATER', 'HOL': 'COAST', 'ION': 'WATER', 'IRI': 'WATER', 'KIE': 'COAST', 'LON': 'COAST', 'LVN': 'COAST', 'LVP': 'COAST', 'LYO': 'WATER', 'MAO': 'WATER', 'MAR': 'COAST', 'MOS': 'LAND', 'MUN': 'LAND', 'NAF': 'COAST', 'NAO': 'WATER', 'NAP': 'COAST', 'NWY': 'COAST', 'NTH': 'WATER', 'NWG': 'WATER', 'PAR': 'LAND', 'PIC': 'COAST', 'PIE': 'COAST', 'POR': 'COAST', 'PRU': 'COAST', 'ROM': 'COAST', 'RUH': 'LAND', 'RUM': 'COAST', 'SER': 'LAND', 'SEV': 'COAST', 'SIL': 'LAND', 'SKA': 'WATER', 'SMY': 'COAST', 'SPA/NC': 'COAST', 'SPA/SC': 'COAST', 'spa': 'COAST', 'STP/NC': 'COAST', 'STP/SC': 'COAST', 'stp': 'COAST', 'SWE': 'COAST', 'SYR': 'COAST', 'TRI': 'COAST', 'TUN': 'COAST', 'TUS': 'COAST', 'TYR': 'LAND', 'TYS': 'WATER', 'UKR': 'LAND', 'VEN': 'COAST', 'VIE': 'LAND', 'WAL': 'COAST', 'WAR': 'LAND', 'WES': 'WATER', 'YOR': 'COAST', 'SWI': 'SHUT'}

# fmt: on

def randomize_move_order(order):
    assert isinstance(order, MTO)
    assert order.unit.unit_type in {'AMY', 'FLT'}
    
    rand = random.random()
    
    if rand < .33:
        return HLD(order.unit)
    elif rand < .66:
        return MTO(order.unit, Location(province=random.choice(ADJACENCY[order.unit.location.province])))
    else:
        return order
    
def randomize_hold_order(order):
    assert isinstance(order, HLD)
    assert order.unit.unit_type in {'AMY', 'FLT'}

    rand = random.random()
    
    if rand < .33:
        return order
    else:
        if order.unit.unit_type == "FLT":
            valid = [loc for loc in ADJACENCY[order.unit.location.province] if TYPES[loc] in {"WATER", "COAST"}]
        else:
            valid = [loc for loc in ADJACENCY[order.unit.location.province] if TYPES[loc] in {"LAND", "COAST"}]
        return MTO(order.unit, Location(province=random.choice(valid)))
    
def randomize_support_order(order):
    assert isinstance(order, SUP)
    assert order.supporting_unit.unit_type in {'AMY', 'FLT'}
    assert order.province_no_coast.province in ADJACENCY

    rand = random.random()

    supporter_location = order.supporting_unit.location
    supportee_location = order.supported_unit.location
    common = {loc for loc in ADJACENCY[supportee_location.province] if loc in ADJACENCY[supporter_location.province]}

    if len(common) <= 1:
        if rand <= .33:
            return order
        elif rand <= .66:
            return HLD(order.supporting_unit)
        else:
            if order.supporting_unit.unit_type == "FLT":
                valid = [loc for loc in ADJACENCY[supporter_location.province] if TYPES[loc] in {"WATER", "COAST"}]
            else:
                valid = [loc for loc in ADJACENCY[supporter_location.province] if TYPES[loc] in {"LAND", "COAST"}]
            return MTO(order.supporting_unit, Location(province=random.choice(valid)))
    else: # if there are other choices for for locations to support into
        if rand <= .25:
            return order
        elif rand <= .5:
            return SUP(order.supporting_unit, order.supported_unit, Location(province=random.choice(common)))
        elif rand <= .75:
            if order.unit.unit_type == "FLT":
                valid = [loc for loc in ADJACENCY[supporter_location.province] if TYPES[loc] in {"WATER", "COAST"}]
            else:
                valid = [loc for loc in ADJACENCY[supporter_location.province] if TYPES[loc] in {"LAND", "COAST"}]
        else:
            return HLD(order.supporting_unit)\
            
def randomize_retreat_order(order):
    assert isinstance(order, RTO)
    assert order.unit.unit_type in {'AMY', 'FLT'}
    
    rand = random.random()
    
    if rand < .33:
        return order
    else:
        if order.unit.unit_type == "FLT":
            valid = [loc for loc in ADJACENCY[order.unit.location.province] if TYPES[loc] in {"WATER", "COAST"}]
        else:
            valid = [loc for loc in ADJACENCY[order.unit.location.province] if TYPES[loc] in {"LAND", "COAST"}]
        return RTO(order.unit, Location(province=random.choice(valid)))

def randomize_order(arrangement):
    assert isinstance(arrangement, XDO)
    
    if isinstance(arrangement.order, MTO):
        return randomize_move_order(arrangement.order)
    elif isinstance(arrangement.order, HLD):
        return randomize_hold_order(arrangement.order)
    elif isinstance(arrangement.order, SUP):
        return randomize_support_order(arrangement.order)
    elif isinstance(arrangement.order, ORR) or isinstance(arrangement.order, AND):
        for i in range(len(arrangement.order.arrangements)):
            all_arrangements_list = list(arrangement.order.arrangements)
            all_arrangements_list[i] = randomize_order(all_arrangements_list[i])
            arrangement.order.arrangements = tuple(all_arrangements_list)
        return arrangement
    else:
        return arrangement

def randomize_visited_tree(visited):
    if isinstance(visited, PRP) and (isinstance(visited.arrangement, ORR) or isinstance(visited.arrangement, AND)):
        for i in range(len(visited.arrangement.arrangements)):
            all_arrangements_list = list(visited.arrangement.arrangements)
            all_arrangements_list[i] = randomize_order(all_arrangements_list[i])
            visited.arrangement.arrangements = tuple(all_arrangements_list)
    return visited

def randomize_daide_string(daide_text):
    visited = string_to_visited_tree(daide_text)
    randomized = randomize_visited_tree(visited)
    return randomized

def string_to_visited_tree(text : str):
    grammar = create_daide_grammar(level=130)
    parse_tree = grammar.parse(text)
    daide_visitor = DAIDEVisitor(None, None)
    visited = daide_visitor.visit(parse_tree)
    return visited

def randomize_message_dict_list(message_dicts: List):
    randomized = []

    for message_dict in message_dicts:
        message_tree = string_to_visited_tree(message_dict)
        if isinstance(message_tree, FCT): # can be randomized
            drop_chance = 0.2 # The chance that an order in an FCT message will get dropped completely
            if random.random() > drop_chance: # message not getting dropped
                randomized_message = randomize_daide_string(message_dict.message)
                randomized.append(randomized_message)
        else:
            randomized.append(message_dict)



    

