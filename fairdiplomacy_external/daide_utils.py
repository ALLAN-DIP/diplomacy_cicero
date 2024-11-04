import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import regex
import sys
from typing import Optional, Tuple, Union
from fairdiplomacy.AMR.AMR_graph import AMR, AMRnode

import copy

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
short_POWERS = {'AUS':'AUSTRIA', 'ENG':'ENGLAND', 'FRA':'FRANCE', 'GER':'GERMANY', 'ITA':'ITALY', 'RUS':'RUSSIA', 'TUR':'TURKEY'}
all_info= {'hold-03' : {'country':'', 'unit': '', 'from':'', 'action':'H'},
        'move-01' : {'country':'','unit': '', 'from':'', 'to':'', 'action':'-'},
        'support-01' : {'support_country':'','support_unit': '', 'support_from':'','support_action':'S', 'country':'', 'unit':'', 'from':'','to':'','action':''},
        'transport-01' : {'transport_country':'','transport_unit': '', 'transport_from':'','transport_action':'C', 'country':'', 'unit':'', 'from':'','to':'','action':'-'},
        'attack-01' : {'country':'','unit': '', 'from':'','to':'', 'action':'-'},
        'bounce-03' : {'country':'','unit': '', 'from':'', 'to':'', 'action':'-'},
        'bounce-03-2' : {'country':'','unit': '', 'from':'', 'to':'', 'action':'-'},
        'retreat-01' : {'country':'','unit': '', 'from':'', 'to':'', 'action':'-'},
        'demilitarize-01': {'country':'','unit': '', 'from':'', 'to':'', 'action':'-','polarity':True},
        'demilitarize-01-2': {'country':'','unit': '', 'from':'', 'to':'', 'action':'-','polarity':True},
       }
sea_keywords = ['sea','gulf','ocean', 'eastern mediterranean','western mediterranean', 'english channel', 'heligoland bight', 'skagerrak', 'st. petersburg']


# attack-01 :ARG0 who ARG1 location or whom (country)
# bounce-03
# :ARG1 who :ARG2 who :location
# :ARG1 and op1 country name op2 country name :ARG3 :location
# demilitarize-01 :ARG1 and op1 country name op2 country name :ARG2 location
# retreat-01 :ARG1 who (unit) :destination province
#check :polarity if its in sub = question
#check possible, porpose in direcet ancestors = question

def get_amr_moves(amr_s, msg):
    amr = AMR()
    amr_tuple = amr.string_to_amr(amr_s)
    extracted_moves = dfs_missing_info_recursive(amr_tuple[0])
    fix_dmz_case(extracted_moves, msg)
    clean_extracted_moves = clean_move_dict(extracted_moves)
    clean_extracted_moves = remove_duplicate(clean_extracted_moves)
    return clean_extracted_moves

def is_concept_in_ancestors(amr_node, concept_list):
    to_search = amr_node.parents
    bool_level = False
    # print(f'{amr_node.concept} {amr_node.variable}')
    for parent in amr_node.parents:
      # print(f'parents {parent.concept} {parent.variable}')
      if parent.concept in concept_list and not any('polarity'==role for role,sub in parent.subs):
        # print('return true')
        return True
      return bool_level or is_concept_in_ancestors(parent, concept_list)
    return bool_level

def identify_missing_info(amr_node: AMRnode):
    info = copy.deepcopy(all_info)
    new_info = {}
    polarity = False
    question = False
    emotion = False
    time = False

    if is_concept_in_ancestors(amr_node, ['propose-01', 'possible-01']):
      # print(amr_node.concept)
      question = True
    if is_concept_in_ancestors(amr_node, ['fear-01','threaten-01','expect-01','warn-01']):
      # print(amr_node.concept)
      emotion = True
    if is_concept_in_ancestors(amr_node, ['polarity']):
      # print(amr_node.concept)
      polarity = True

    for parent in amr_node.parents:
      for role, sub in parent.subs:
        if role=='polarity':
            question = True

    for role, sub in amr_node.subs:
        try:
          for role1,sub1 in sub.subs:
            if role1=='year':
              time=True
              year=sub1
        except:
          continue
    # print(amr_node.concept)
    # print(amr_node.subs)
    list_info = []
    double_move = dict()
    i=0
    for role, sub in amr_node.subs:
          # if polarity is next to amr_node concept then yes it is
          if role=='polarity' and i == 0:
            polarity = True
          # if polarity is last in children level of amr_node concept
          elif role=='polarity':
            question = True
          i+=1
          # Check for hold-03 relation
          # print(role, sub)
          if amr_node.concept == 'hold-03':
              if role == 'ARG0':
                  info['hold-03']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  info['hold-03']['country']  = dfs_finding_var_recursive(sub,'country')
              elif role == 'ARG1' and (info[amr_node.concept]['from']==''):
                  info['hold-03']['from'] =  dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')

          # Check for move-01 relation
          if amr_node.concept == 'move-01':
              if role == 'ARG1' and (info[amr_node.concept]['from']==''):
                  info['move-01']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  info['move-01']['country']  = dfs_finding_var_recursive(sub,'country')
                  info['move-01']['from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
              elif role == 'ARG2' and (info[amr_node.concept]['to']==''):
                  info['move-01']['to']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
                  if info['move-01']['to']!= '' and  info['move-01']['from'] ==info['move-01']['to']:
                    info['move-01']['to'] = ''

          # Check for transport-01 relation
          if amr_node.concept == 'transport-01':
              if role == 'ARG0' and (info[amr_node.concept]['transport_from']==''):
                  info['transport-01']['transport_unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  info['transport-01']['transport_country']  = dfs_finding_var_recursive(sub,'country')
                  info['transport-01']['transport_from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
              elif role == 'ARG1'  and (info[amr_node.concept]['from']==''):
                  info['transport-01']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  info['transport-01']['country']  = dfs_finding_var_recursive(sub,'country')
                  info['transport-01']['from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
              elif role == 'ARG3' and (info[amr_node.concept]['to']==''):
                  info['transport-01']['to']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')

          # Check for support-01 relation
          # please add case instrument!
          if amr_node.concept == 'support-01':
              if role == 'ARG0' and (info[amr_node.concept]['support_from']==''):
                  info['support-01']['support_unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  info['support-01']['support_country']  = dfs_finding_var_recursive(sub,'country')
                  info['support-01']['support_from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
              elif role == 'ARG1':
                  if not isinstance(sub,AMRnode) or sub.concept not in info:
                    continue
                  # print(f'------checking nested {sub.concept} -------')
                  move_info = identify_missing_info(sub)  # Recursively check for nested relation
                  if not len(move_info):
                    continue
                  move_info = move_info[0]
                  # print(f'------checking nested {sub.concept} and its output: {move_info}')
                  for key, value in move_info.items():
                    info['support-01'][key] = value
              elif role == 'instrument':
                  info['support-01']['support_unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  info['support-01']['support_country']  = dfs_finding_var_recursive(sub,'country')
                  info['support-01']['support_from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')


          # attack-01 :ARG0 who ARG1 location or whom (country)
          if amr_node.concept == 'attack-01':
            if role== 'ARG0' and (info[amr_node.concept]['from']==''):
                  info['attack-01']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  info['attack-01']['country']  = dfs_finding_var_recursive(sub,'country')
                  info['attack-01']['from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
            if role== 'ARG1' and (info[amr_node.concept]['to']==''):
                  info['attack-01']['to']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
                  if info['attack-01']['to']!= '' and  info['attack-01']['from'] ==info['attack-01']['to']:
                    info['attack-01']['to'] = ''

          # bounce-03
          # :ARG1 who :ARG2 who :location
          # :ARG1 and op1 country name op2 country name :ARG3 :location
          if amr_node.concept == 'bounce-03':
            # print(amr_node.concept)
            # print(amr_node.subs)
            if role== 'ARG1' and (info[amr_node.concept]['from']==''):
                  info['bounce-03']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  # if we get more than one country, split into two moves
                  countries = dfs_finding_var_recursive(sub,'country').split()
                  # print(f'countries getting from ARG1 country: {countries}')
                  if not countries:
                    continue
                  info['bounce-03']['country']  = countries[0]
                  if len(countries) > 1:
                    info[f'{amr_node.concept}'+'-2']['country'] = countries[1]
                  info['bounce-03']['from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
            if role== 'ARG2' and (info[f'{amr_node.concept}'+'-2']['from']==''):
                  # print('in ARG2 ------------')
                  info[f'{amr_node.concept}'+'-2']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  info[f'{amr_node.concept}'+'-2']['country']  = dfs_finding_var_recursive(sub,'country')
                  # print(info[f'{amr_node.concept}'+'-2']['country'])
                  info[f'{amr_node.concept}'+'-2']['from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
            if role== 'ARG3' and (info[amr_node.concept]['to']==''):
                  # print('in ARG3 ------------')
                  info['bounce-03']['to']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
                  info[f'{amr_node.concept}'+'-2']['to'] = info['bounce-03']['to']
            # print(f'bounce_move {bounce_move}')

          # retreat-01 :ARG1 who (unit) :destination province
          if amr_node.concept == 'retreat-01':
              if role == 'ARG1' and (info[amr_node.concept]['from']==''):
                  info['retreat-01']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
                  info['retreat-01']['country']  = dfs_finding_var_recursive(sub,'country')
                  info['retreat-01']['from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
              elif role == 'destination' and (info[amr_node.concept]['to']==''):
                  info['retreat-01']['to']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')

          # demilitarize-01 :ARG1 and op1 country name op2 country name :ARG2 location
          if amr_node.concept == 'demilitarize-01':
              polarity = True
              if role== 'ARG1' and (info[amr_node.concept]['from']==''):
                    # if we get more than one country, split into two moves
                    countries = dfs_finding_var_recursive(sub,'country').split()
                    info['demilitarize-01']['country']  = countries[0]
                    if len(countries) > 1:
                      info[f'{amr_node.concept}'+'-2']['country'] = countries[1]
              if role== 'ARG2' and (info[amr_node.concept]['to']==''):
                    info['demilitarize-01']['to']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
                    info[f'{amr_node.concept}'+'-2']['to'] = info['demilitarize-01']['to']

    #add to the list, all has list with len =1 except bounce, dmz that has len =2
    if 'action' in double_move:
        list_info.append(double_move)
    if amr_node.concept in ['bounce-03','demilitarize-01'] :
        list_info.append(info[f'{amr_node.concept}'+'-2'])

    if amr_node.concept in info:
      list_info.append(info[amr_node.concept])

    for new_info in list_info:
      if 'action' in new_info:
        new_info['concept'] = amr_node.concept
        new_info['variable'] = amr_node.variable
        new_info['polarity'] = copy.deepcopy(polarity)
        new_info['question'] = copy.deepcopy(question)
        new_info['emotion'] = copy.deepcopy(emotion)
        if time:
          new_info['year']= year
    # print(list_info)
    return list_info

def dfs_missing_info_recursive(node):
    list_of_moves = []
    if isinstance(node, AMRnode):
        # print(f'{node.concept} {node.variable} {node.subs}')

        if node.concept not in all_info:
          for role, sub in node.subs:
              # if 'op' in role:
                  #print(sub)
              list_of_moves += dfs_missing_info_recursive(sub)
        else:
          return identify_missing_info(node)
    return list_of_moves

def dfs_finding_var_recursive(node, finding_key):
      # print(f'{node}')
    if isinstance(node, AMRnode):
        # print(f'{node.concept} {node.variable} {node.subs}')
        # found var
        if node.concept == finding_key and node.concept == 'name':
          loc_name = []
          for role, sub in node.subs:
            if 'op' in role:
              loc_name.append(sub)
          return ' '.join(loc_name)
        # found key -> find var
        if node.concept == finding_key:
          # print(f'{node.concept} and {finding_key}')
          if finding_key in ['army', 'fleet']:
            return finding_key
          else:
            return dfs_finding_var_recursive(node, 'name')

        if len(node.subs)==0:
          return ''
        answer = ''
        for role, sub in node.subs:
            # if 'op' in role:
                # print(sub)
            possible_find = dfs_finding_var_recursive(sub, finding_key)
            if answer != '':
              answer+= ' ' +possible_find
            else:
              answer+= possible_find
        return answer
    else:
      return ''



# deception
def copy_move(move):
  # if country is double
  # if unit is double
  # if province is double
  return copy.deepcopy(move)

def clean_move_dict(extracted_moves):
  new_moves = []
  # if country is double
  # if unit is double
  # if province is double
  for move in extracted_moves:
      for key in ['support_country', 'transport_country','country','support_from','transport_from','from','to']:

          info_key = move.get(key, '')
          info_key = info_key.split()
          if len(info_key)>1:
            if 'to' in key or 'from' in key:
              if any([sea_key in move[key].lower() for sea_key in sea_keywords]):
                continue

            move[key] = info_key[0]
            new_move = copy_move(move)
            new_move[key] = info_key[1]
            new_moves.append(new_move)
  return extracted_moves + new_moves

def remove_duplicate(extracted_moves):
  seen = set()
  unique_moves = []
  for move in extracted_moves:
      # Make a hashable version of the dictionary
      move_tuple = tuple(sorted(move.items()))
      if move_tuple not in seen:
          seen.add(move_tuple)
          unique_moves.append(move)
  return unique_moves

def remove_emotion(extracted_moves):
  new_extracted_moves = []
  for move in extracted_moves:
    if move['emotion'] != True:
      new_extracted_moves.append(move)
  return new_extracted_moves

def filter_none_agreement(extracted_moves, future_message):
  amr = AMR()
  new_extracted_moves = []
  # if move is question
  # if there is agree in future_message
  for move in extracted_moves:
    agree = False
    move_country =  move.get('support_country', move.get('transport_country', move.get('country', '')))
    if move['question']:
      for message in future_message:
        amr_tuple = amr.string_to_amr(message['parsed-amr'])
        root_node = amr_tuple[0]
        if root_node.concept =='agree-01' and move_country!=message['sender']:
          # print(message)
          agree = True
          break
      if not agree:
        continue
    new_extracted_moves.append(move)

  return new_extracted_moves

def fix_dmz_case(extracted_moves,m):
  # if len(extracted_moves) ==1:
  for move in extracted_moves:
    if move['concept'] == 'move-01' and 'dmz' in m['message'].lower() and not move['polarity']:
      move['concept'] = 'demilitarize-01'
      move['polarity'] = True

def is_power_unit(units, move_prov):
  for unit in units:
    unit_loc = unit.split()[1]
    is_prov,_ = check_province(move_prov, unit_loc)
    if is_prov:
      return True
  return False

def get_power_unit(units, move_prov):
  for unit in units:
    unit_loc = unit.split()[1]
    is_prov,_ = check_province(move_prov, unit_loc)
    if is_prov:
      return unit
  return ''

def is_prov_in_units(units, move_prov):
  if move_prov =='':
    return False
  for unit in units:
    unit_loc = unit.split()[1]
    is_prov,_ = check_province(move_prov, unit_loc)
    if is_prov:
      return True
  return False

def is_move_in_order_set(order_set, move, order_country=''):
    exist = False
    # if to_do_move is not to any final order
    for order in order_set:
        if order == '':
          continue
        order_tokens = order.split()
        is_exist = check_moves(move, order_tokens, order_country)
        if is_exist=='not enough info':
          return is_exist
        # if is_exist:
        #   print(order_tokens)

        exist = exist or is_exist
    return exist

def get_move_in_order_set(order_set, move):
    # if to_do_move is not to any final order
    for order in order_set:
        order_tokens = order.split()
        is_exist = check_moves(move, order_tokens)
        if is_exist=='not enough info':
          return ''
        elif is_exist:
          return order_tokens
    return ''

def remove_bounce_order_in_prev_m(order_set, bounce):
  new_set = []
  for order in order_set:
    order_toks = order.split()
    unit = order_toks[0] + ' ' + order_toks[1]
    if not any(r=='bounce' for  r in bounce[unit]):
      new_set.append(order)
  return new_set

def check_country(daide_sentence, move):
  country1 =  move.get('support_country', move.get('transport_country', ''))
  if country1 != '':
    country1 = country1[:3].upper()

  country2 =  move.get('country', '')
  if country2 != '':
    country2 = country2[:3].upper()
  return country1 in daide_sentence and country2 in daide_sentence

def check_action(daide_sentence, move):
  if move['concept'] in daide_sentence:
    return True
  if move['concept'] in ['move-01', 'attack-01', 'retreat-01','bounce-03'] and 'MTO' in daide_sentence:
    return True
  if move['concept'] in ['support-01'] and 'SUP' in daide_sentence:
    return True
  if move['concept'] in ['transport-01'] and 'CVY' in daide_sentence:
    return True
  if move['concept'] in ['retreat-01'] and 'RTO' in daide_sentence:
    return True
  if move['concept'] in ['hold-03'] and 'HLD' in daide_sentence:
    return True
  if move['concept'] in ['demilitarize-01'] and 'DMZ' in daide_sentence:
    return True
  return False

def check_province(daide_sentence, move):
  with open('/diplomacy_cicero/fairdiplomacy_external/mapping_abbre.json', 'r') as file:
      mapping_province = json.load(file)
  p1 =  move.get('support_from', move.get('transport_from', ''))
  if p1 != '':
      p1 = mapping_province[p1.strip()][0].upper() if p1 in mapping_province else ''

  p2 =  move.get('from', '')
  if p2 != '':
      p2 = mapping_province[p2.strip()][0].upper() if p2 in mapping_province else ''

  p3 =  move.get('to', '')
  if p3 != '':
      p3 = mapping_province[p3.strip()][0].upper() if p3 in mapping_province else ''
  return p1 in daide_sentence and p2 in daide_sentence and p3 in daide_sentence

def get_action_sentence(move, propose):
    with open('/diplomacy_cicero/fairdiplomacy_external/mapping_abbre.json', 'r') as file:
        mapping_province = json.load(file)
    action_s = ''
    propose_s = 'I propose an order '
    fct_s = 'Please expect an order '

    if move['concept'] in ['move-01', 'attack-01'] and (move['country'] != '' or (move['from'] != '' and move['from'] in mapping_province )) and (move['to']!='' and move['to'] in mapping_province) :
        u_country = f"{move['country'][:3].upper()}\'s " if move['country'] != '' else ''
        u_from = f"unit in {mapping_province[move['from'].strip()][0].upper()} " if move['from'] != '' and move['from'] in mapping_province else 'unit '
        u_to = f"to {mapping_province[move['to'].strip()][0].upper()}"
        action_s = f'moving {u_country}{u_from}{u_to}'

    if move['concept'] in ['retreat-01'] and (move['country'] != '' or (move['from'] != '' and move['from'] in mapping_province )) and (move['to']!='' and move['to'] in mapping_province):
        u_country = f"{move['country'][:3].upper()}\'s " if move['country'] != '' else ''
        u_from = f"unit in {mapping_province[move['from'].strip()][0].upper()} " if move['from'] != '' and move['from'] in mapping_province else 'unit '
        u_to = f"to {mapping_province[move['to'].strip()][0].upper()}"
        action_s = f'retreating {u_country}{u_from}{u_to}'

    if move['concept'] in ['bounce-03'] and move['country'] != '' and (move['to']!='' and move['to'] in mapping_province):
        u_country = f"{move['country'][:3].upper()}\'s "
        u_from = "unit "
        u_to = f"in {mapping_province[move['to'].strip()][0].upper()}"
        action_s = f'bouncing {u_country}{u_from}{u_to}'

    if move['concept'] in ['support-01'] and (move['to']!='' and move['to'] in mapping_province) and ((move['support_from'] != '' and move['support_from'] in mapping_province ) or move['support_country']!=''):
        u_support_country = f"{move['support_country'][:3].upper()}\'s " if move['support_country'] != '' else ''
        u_support_from = f"unit in {mapping_province[move['support_from'].strip()][0].upper()} " if move['support_from'] != '' and move['support_from'] in mapping_province else 'unit '
        u_country = f"{move['country'][:3].upper()}\'s " if move['country'] != '' else ''
        u_from = f"unit in {mapping_province[move['from'].strip()][0].upper()} " if move['from'] != '' and move['from'] in mapping_province else 'unit '
        if move['action'] == '-':
          u_to = f"moving into {mapping_province[move['to'].strip()][0].upper()}"
        else:
          u_to = ''
        action_s = f'using {u_support_country}{u_support_from}to support {u_country}{u_from}{u_to}'

    if move['concept'] in ['transport-01'] and move['to']!='' and move['to'] in mapping_province and ((move['transport_from'] != '' and move['transport_from'] in mapping_province ) or move['transport_country']!=''):
        u_support_country = f"{move['transport_country'][:3].upper()}\'s " if move['transport_country'] != '' else ''
        u_support_from = f"unit in {mapping_province[move['transport_from'].strip()][0].upper()} " if move['transport_from'] != ''and move['transport_from'] in mapping_province  else 'unit '
        u_country = f"{move['country'][:3].upper()}\'s " if move['country'] != '' else ''
        u_from = f"unit in {mapping_province[move['from'].strip()][0].upper()} " if move['from'] != '' and move['from'] in mapping_province else 'unit '
        u_to = f"into {mapping_province[move['to'].strip()][0].upper()}"
        action_s = f'using {u_support_country}{u_support_from}to convoy {u_country}{u_from}{u_to}'

    if move['concept'] in ['hold-03'] and move['from'] != '' and move['from'] in mapping_province :
        u_country = f"{move['country'][:3].upper()}\'s " if move['country'] != '' else ''
        u_from = f"unit in {mapping_province[move['from'].strip()][0].upper()} "
        action_s = f'holding {u_country}{u_from}'

    if move['concept'] in ['demilitarize-01'] and move['to']!='' and move['to'] in mapping_province :
        u_to = f"{mapping_province[move['to'].strip()][0].upper()}"
        action_s = f'to DMZ in {u_to}'

    if action_s == '':
      return None
    if propose:
      return propose_s + action_s
    else:
      if move['polarity']:
        return fct_s + 'not '+ action_s
      else:
        return fct_s + action_s