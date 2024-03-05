import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import regex
import sys
from typing import Optional, Tuple, Union
from AMR_graph import AMR, AMRnode
import copy
POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
all_info= {'hold-03' : {'country':'', 'unit': '', 'from':'', 'action':'H'},
       'move-01' : {'country':'','unit': '', 'from':'', 'to':'', 'action':'-'},
        'retreat-01' : {'country':'','unit': '', 'from':'', 'to':'', 'action':'R'},
       'support-01' : {'support_country':'','support_unit': '', 'support_from':'','support_action':'S', 'country':'', 'unit':'', 'from':'','to':'','action':''},
       'transport-01' : {'transport_country':'','transport_unit': '', 'transport_from':'','transport_action':'C', 'country':'', 'unit':'', 'from':'','to':'','action':'-'}
       }

# def identify_missing_info(amr_node: AMRnode):
#     info = copy.deepcopy(all_info)
#     # print(amr_node.subs)
#     new_info = {}
#     for role, sub in amr_node.subs:
#           # Check for hold-03 relation
#           if amr_node.concept == 'hold-03':
#               if role == 'ARG0':
#                   info['hold-03']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
#                   info['hold-03']['country']  = dfs_finding_var_recursive(sub,'country')
#               elif role == 'ARG1':
#                   info['hold-03']['from'] = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')

#           # Check for move-01 relation
#           if amr_node.concept == 'move-01':
#               if role == 'ARG1':
#                   info['move-01']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
#                   info['move-01']['country']  = dfs_finding_var_recursive(sub,'country')
#                   info['move-01']['from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
#               elif role == 'ARG2':
#                   info['move-01']['to']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')

#           # Check for transport-01 relation
#           if amr_node.concept == 'transport-01':
#               if role == 'ARG0':
#                   info['transport-01']['transport_unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
#                   info['transport-01']['transport_country']  = dfs_finding_var_recursive(sub,'country')
#                   info['transport-01']['transport_from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
#               elif role == 'ARG1':
#                   info['transport-01']['unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
#                   info['transport-01']['country']  = dfs_finding_var_recursive(sub,'country')
#                   info['transport-01']['from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
#               elif role == 'ARG3':
#                   info['transport-01']['to']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')

#           # Check for support-01 relation
#           # please add case instrument!
#           if amr_node.concept == 'support-01':
#               if role == 'ARG0':
#                   info['support-01']['support_unit'] = dfs_finding_var_recursive(sub,'army') + dfs_finding_var_recursive(sub,'fleet')
#                   info['support-01']['support_country']  = dfs_finding_var_recursive(sub,'country')
#                   info['support-01']['support_from']  = dfs_finding_var_recursive(sub,'province') + dfs_finding_var_recursive(sub,'sea')
#               elif role == 'ARG1':
#                   move_info = identify_missing_info(sub)  # Recursively check for nested relation
#                   # print(f'------checking nested {sub.concept} and its output: {move_info}')
#                   for key, value in move_info.items():
#                     info['support-01'][key] = value
#               # print(info[amr_node.concept])
#           new_info = info[amr_node.concept] if amr_node.concept in info else {}
#           if 'action' in new_info:
#             new_info['concept'] = amr_node.concept
#             new_info['variable'] = amr_node.variable

#     return new_info

def refill(amr_node: AMRnode,filled_moves):
    info = copy.deepcopy(all_info)
    # print(amr_node.subs)
    new_info = {}
    print(filled_moves)
    for role, sub in amr_node.subs:
          # Check for hold-03 relation
          if amr_node.concept == 'hold-03':
              if role == 'ARG0':
                  info['hold-03']['unit'] = dfs_finding_refill(sub,'army',filled_moves,'unit') + dfs_finding_refill(sub,'fleet',filled_moves,'unit')
                  info['hold-03']['country']  = dfs_finding_refill(sub,'country',filled_moves,'country')
              elif role == 'ARG1':
                  info['hold-03']['from'] = dfs_finding_refill(sub,'province',filled_moves,'from') + dfs_finding_refill(sub,'sea',filled_moves,'from')

          # Check for move-01 relation
          if amr_node.concept == 'move-01':
              if role == 'ARG1':
                  info['move-01']['unit'] = dfs_finding_refill(sub,'army',filled_moves,'unit') + dfs_finding_refill(sub,'fleet',filled_moves,'unit')
                  info['move-01']['country']  = dfs_finding_refill(sub,'country',filled_moves,'country')
                  info['move-01']['from']  = dfs_finding_refill(sub,'province',filled_moves,'from') + dfs_finding_refill(sub,'sea',filled_moves,'from')
              elif role == 'ARG2':
                  info['move-01']['to']  = dfs_finding_refill(sub,'province',filled_moves,'to') + dfs_finding_refill(sub,'sea',filled_moves,'to')

          # Check for transport-01 relation
          if amr_node.concept == 'transport-01':
              if role == 'ARG0':
                  info['transport-01']['transport_unit'] = dfs_finding_refill(sub,'army',filled_moves,'transport_unit') + dfs_finding_refill(sub,'fleet',filled_moves,'transport_unit')
                  info['transport-01']['transport_country']  = dfs_finding_refill(sub,'country',filled_moves,'transport_country')
                  info['transport-01']['transport_from']  = dfs_finding_refill(sub,'province',filled_moves,'transport_from') + dfs_finding_refill(sub,'sea',filled_moves,'transport_from')
              elif role == 'ARG1':
                  info['transport-01']['unit'] = dfs_finding_refill(sub,'army',filled_moves,'unit') + dfs_finding_refill(sub,'fleet',filled_moves,'unit')
                  info['transport-01']['country']  = dfs_finding_refill(sub,'country',filled_moves,'country')
                  info['transport-01']['from']  = dfs_finding_refill(sub,'province',filled_moves,'from') + dfs_finding_refill(sub,'sea',filled_moves,'from')
              elif role == 'ARG3':
                  info['transport-01']['to']  = dfs_finding_refill(sub,'province',filled_moves,'to') + dfs_finding_refill(sub,'sea',filled_moves,'to')

          # Check for support-01 relation
          # please add case instrument!
          if amr_node.concept == 'support-01':
              if role == 'ARG0':
                  info['support-01']['support_unit'] = dfs_finding_refill(sub,'army',filled_moves,'support_unit') + dfs_finding_refill(sub,'fleet',filled_moves,'support_unit')
                  info['support-01']['support_country']  = dfs_finding_refill(sub,'country',filled_moves,'support_country')
                  info['support-01']['support_from']  = dfs_finding_refill(sub,'province',filled_moves,'support_from') + dfs_finding_refill(sub,'sea',filled_moves,'support_from')
              elif role == 'ARG1':
                  move_info = refill(sub,filled_moves)  # Recursively check for nested relation
                  # print(f'------checking nested {sub.concept} and its output: {move_info}')
                  for key, value in move_info.items():
                    info['support-01'][key] = value
              # print(info[amr_node.concept])
          new_info = info[amr_node.concept] if amr_node.concept in info else {}
          if 'action' in new_info:
            new_info['concept'] = amr_node.concept
            new_info['variable'] = amr_node.variable

    return new_info

# def dfs_missing_info_recursive(node):
#     list_of_moves = []
#     if isinstance(node, AMRnode):
#         # print(f'{node.concept} {node.variable} {node.subs}')

#         if node.concept not in ['hold-03','move-01','transport-01','support-01']:
#           for role, sub in node.subs:
#               list_of_moves += dfs_missing_info_recursive(sub)
#         else:
#           return [identify_missing_info(node)]
#     return list_of_moves

def dfs_missing_info_refill(node,filled_moves):
    list_of_moves = []
    if isinstance(node, AMRnode):
        # print(f'{node.concept} {node.variable} {node.subs}')

        if node.concept not in ['hold-03','move-01','transport-01','support-01']:
          for role, sub in node.subs:
              list_of_moves += dfs_missing_info_refill(sub,filled_moves)
        else:
          return [refill(node,filled_moves)]
    return list_of_moves

# def dfs_finding_var_recursive(node, finding_key):
#     if isinstance(node, AMRnode):
#         # print(f'{node.concept} {node.variable} {node.subs}')

#         # found var
#         if node.concept == finding_key and node.concept == 'name':
#           loc_name = []
#           for role, sub in node.subs:
#             if 'op' in role:
#               loc_name.append(sub)
#           return ' '.join(loc_name)
#         # found key -> find var
#         if node.concept == finding_key:
#           if finding_key in ['army', 'fleet']:
#             return finding_key
#           else:
#             return dfs_finding_var_recursive(node, 'name')

#         if len(node.subs)==0:
#           return ''

#         for role, sub in node.subs:
#             # if 'op' in role:
#                 # print(sub)
#             return '' + dfs_finding_var_recursive(sub, finding_key)
#     else:
#       return ''

def dfs_finding_refill(node, finding_key,filled_moves,map_dic):
    if isinstance(node, AMRnode):
        # print(f'{node.concept} {node.variable} {node.subs}')
        # print(finding_key)

        # found var
        if node.concept == finding_key and node.concept == 'name':
          print(f'{node.concept} {node.variable} {node.subs}')
          print(map_dic)
          loc_name = []
          for role, sub in node.subs:
            if 'op' in role:
              loc_name.append(sub)
          return ' '.join(loc_name)
        # found key -> find var
        if node.concept == finding_key:
          if finding_key in ['army', 'fleet']:
            return finding_key
          else:
            return dfs_finding_refill(node, 'name',filled_moves,map_dic)

        if len(node.subs)==0:
          return ''

        for role, sub in node.subs:
            # if 'op' in role:
                # print(sub)
            return '' + dfs_finding_refill(sub, finding_key,filled_moves,map_dic)
    else:
      return ''

file_path = './test_gold_example.json'

with open(file_path, 'r') as file:
  test_gold_amr = json.load(file)
amr = AMR()

for msg_tuple in test_gold_amr:
  gold_AMR = msg_tuple['gold-AMR']
  incomplete_AMR = msg_tuple['parsed-AMR']
  filled_moves = msg_tuple['extracted_moves']
  if incomplete_AMR:
    # print(msg_tuple['msg'])
    print(incomplete_AMR)
    print(gold_AMR)
    amr_tuple = amr.string_to_amr(incomplete_AMR)
    root_node = amr_tuple[0]
    # extracted_moves = dfs_missing_info_recursive(root_node)
    # print(extracted_moves)
    extracted_moves = dfs_missing_info_refill(root_node,filled_moves)
    #print(extracted_moves)
    amr_s2 = amr.amr_to_string()
    #print(amr_s2)
