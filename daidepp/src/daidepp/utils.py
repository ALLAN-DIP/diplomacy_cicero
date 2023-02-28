from daidepp import create_daide_grammar
from daidepp.daide_visitor import DAIDEVisitor
from daidepp.keywords.keyword_utils import power_dict, power_list
from typing import List
import parsimonious, re

def pre_process(daide: str) -> str:
    '''
        change the dipnet syntax to daidepp syntax
    '''
    # substitutions
    # case CTO province VIA (sea_province sea_province ...)

    # case RTO province

    # case DMZ (power power ...) (province province ...)

    # case HOW (province)

    # since 'ENG' is used both as a power and a location, we need to substitute
    # the location with 'ECH'.
    # and replace coast with SCS, NCS, ECS, WCS
    return daide.replace('BOT', 'GOB') \
                .replace('FLT ENG', 'FLT ECH') \
                .replace('AMY ENG', 'AMY ECH') \
                .replace('CTO LON', 'CTO ECH') \
                .replace('/SC', ' SCS') \
                .replace('/NC', ' NCS') \
                .replace('/EC', ' ECS') \
                .replace('/WC', ' WCS')

    

def gen_English(daide: str, self_power=None, send_power=None) -> str:
    '''
    Generate English from DAIDE
    :param daide: DAIDE string, e.g. '(ENG FLT LON) BLD'
    :param self_power: power sending the message
    :param send_power: power to which the message is sent
    '''

    try:
        # create daide grammar
        grammar = create_daide_grammar(
            level=130, 
            allow_just_arrangement=True, 
            string_type='all'
        )
        
        parse_tree = grammar.parse(daide)
        daide_visitor = DAIDEVisitor(self_power, send_power)
        output = str(daide_visitor.visit(parse_tree))

        return output
        
    except parsimonious.exceptions.ParseError:
        return 'ERROR parsing ' + daide
    
def post_process(sentence: str, self_power=None, send_power=None) -> str:
    '''
    Make the sentence more grammatical and readable
    :param sentence: DAIDE string, e.g. '(ENG FLT LON) BLD'
    '''

    if sentence.startswith('ERROR'):
        return sentence

    # remove extra spaces
    output = " ".join(sentence.split())
    # add period if needed
    if not output.endswith('.') or not output.endswith('?'):
        output += '.'


    # first & second person possessive/substitution
    if (send_power in power_list):
        pattern = send_power + "'s"
        output = output.replace(pattern, 'your')
        output = output.replace(send_power, 'you')

    if (self_power in power_list):
        pattern = self_power + "'s"
        output = output.replace(pattern, 'my')
        output = output.replace(self_power, 'I')

    # TODO: Third singular s for verbs

    # TODO: disambiguate 'ENG' as a power and a location

    return output

# remove punctuations
def tokenize(sentence: str) -> List[str]:
    def trim_all(token: str) -> str:
        if len(token) > 0:
            token = token.strip()
        while len(token) > 0 and token[0] == '"':
            token = token[1:]
        while len(token) > 0 and not token[-1].isalnum():
            token = token[:-1]
        return token

    tokens = list(map(lambda x: trim_all(x), 
                      sentence.replace('(', ' ')
                              .replace(')', ' ')
                              .split(' ')
                 ))
    
    return list(filter(None, tokens))

def is_daide(sentence: str) -> bool:
    '''
    Check if the tokens are three uppercase letters
    '''
    PRESS_TOKENS = ['PRP', 'YES', 'REJ', 'BWX', 'HUH', 'CCL', 'FCT', 'TRY',
                    'INS', 'QRY', 'THK', 'IDK', 'WHT', 'HOW', 'EXP', 'SRY', 
                    'IFF', 'FRM', 'WHY', 'POB', 'UHY', 'HPY', 'ANG']
    
    if sentence[:3] in PRESS_TOKENS:
        return True

    tokens = tokenize(pre_process(sentence))

    for token in tokens:
        if not token.isupper() or len(token) != 3:
            return False
        
    return True