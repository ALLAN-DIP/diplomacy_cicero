#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Ulf Hermjakob, USC/ISI
"""

import os
from pathlib import Path
import re
from typing import Optional, Tuple, Union

data_dir = Path(__file__).parent.parent / 'data'
data_dir_path = str(data_dir.resolve())


def slot_value_in_double_colon_del_list(line: str, slot: str, default: Optional = None) -> str:
    """For a given slot, e.g. 'cost', get its value from a line such as '::s1 of course ::s2 ::cost 0.3' -> 0.3
    The value can be an empty string, as for ::s2 in the example above."""
    m = re.match(fr'(?:.*\s)?::{slot}(|\s+\S.*?)(?:\s+::\S.*|\s*)$', line)
    return m.group(1).strip() if m else default


class Daide:
    def __init__(self, filename: str):
        self.power_name = {}
        self.province_name = {}
        self.sea_name = {}
        self.unit_type_name = {}
        self.coast_name = {}
        self.pertainym = {}
        self.name_uses_def_article = {}
        self.to_name = {}
        self.name_to_id = {}
        self.sem_annotation = {}
        self.synt_annotation = {}
        self.load_resources(filename)

    def load_resources(self, filename: str) -> None:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('::power-id '):
                    power_id = slot_value_in_double_colon_del_list(line, 'power-id')
                    power_name = slot_value_in_double_colon_del_list(line, 'power-name')
                    if power_id and power_name:
                        self.power_name[power_id] = power_name
                        self.to_name[power_id] = power_name
                        self.name_to_id[power_name] = power_id
                elif line.startswith('::province-id '):
                    province_id = slot_value_in_double_colon_del_list(line, 'province-id')
                    province_name = slot_value_in_double_colon_del_list(line, 'province-name')
                    if province_id and province_name:
                        self.province_name[province_id] = province_name
                        self.to_name[province_id] = province_name
                        self.name_to_id[province_name] = province_id
                elif line.startswith('::sea-id '):
                    sea_id = slot_value_in_double_colon_del_list(line, 'sea-id')
                    sea_name = slot_value_in_double_colon_del_list(line, 'sea-name')
                    sea_alt_names = slot_value_in_double_colon_del_list(line, 'sea-alt-names')
                    if sea_id and sea_name:
                        self.sea_name[sea_id] = sea_name
                        self.to_name[sea_id] = sea_name
                        self.name_uses_def_article[sea_name] = True
                        self.name_to_id[sea_name] = sea_id
                        if sea_alt_names:
                            for sea_alt_name in re.split(r'[,;]\s*', sea_alt_names):
                                self.name_to_id[sea_alt_name] = sea_id
                elif line.startswith('::unit-type-id '):
                    unit_type_id = slot_value_in_double_colon_del_list(line, 'unit-type-id')
                    unit_type_name = slot_value_in_double_colon_del_list(line, 'unit-type-name')
                    if unit_type_id and unit_type_name:
                        self.unit_type_name[unit_type_id] = unit_type_name
                        self.to_name[unit_type_id] = unit_type_name
                        self.name_to_id[unit_type_name] = unit_type_id
                elif line.startswith('::coast-id '):
                    coast_id = slot_value_in_double_colon_del_list(line, 'coast-id')
                    coast_name = slot_value_in_double_colon_del_list(line, 'coast-name')
                    coast_alt_names = slot_value_in_double_colon_del_list(line, 'coast-alt-names')
                    if coast_id and coast_name:
                        self.coast_name[coast_id] = coast_name
                        self.to_name[coast_id] = coast_name
                        if coast_alt_names:
                            for coast_alt_name in re.split(r'[,;]\s*', coast_alt_names):
                                self.name_to_id[coast_alt_name] = coast_id
                elif line.startswith('::name '):
                    name = slot_value_in_double_colon_del_list(line, 'name')
                    pertainym = slot_value_in_double_colon_del_list(line, 'pertainym')
                    if name and pertainym:
                        self.pertainym[name] = pertainym
                        self.to_name[pertainym] = name

    def parse_daide_tree(self, s: str,
                         rec_level: int = 0,
                         index: int = 0, max_index: Optional[int] = None) \
            -> Tuple[list, list, int]:
        """returns tuple(list-tree, error-list, new_index)"""
        if max_index is None:
            max_index = len(s) - 1
        tree = []
        errors = []
        while True:
            if index <= max_index:
                char = s[index]
                current_index = index
                index += 1  # next index
                if char == ' ':
                    pass
                elif char == '(':
                    sub_tree, sub_errors, sub_index = self.parse_daide_tree(s, rec_level+1, index, max_index)
                    tree.append(sub_tree)
                    index = sub_index
                    errors.extend(sub_errors)
                elif char == ')':
                    if rec_level:
                        return tree, errors, index
                    else:
                        errors.append(f'Ignoring spurious close parenthesis at position {current_index}')
                elif char.isalpha():
                    element = char
                    while index <= max_index and s[index].isalpha():
                        element += s[index]
                        index += 1
                    tree.append(element)
                else:
                    errors.append(f'Ignoring spurious character {char} at position {current_index}')
            elif rec_level == 0:
                return tree, errors, index
            else:
                errors.append(f'Missing close parenthesis')
                return tree, errors, index

    def print_daide_tree(self, tree: Union[list, str], rec_level: int = 0) -> str:
        result = ''
        if isinstance(tree, list):
            if rec_level:
                result += '('
            result += ' '.join([self.print_daide_tree(tree[x], rec_level+1) for x in list(range(len(tree)))])
            if rec_level:
                result += ')'
        else:
            result += tree
        return result

    def ann(self, s: str, synt: str = None, sem: str = None) -> str:
        """Annotate (result) string"""
        if synt:
            self.synt_annotation[s] = synt
        if sem:
            self.sem_annotation[s] = sem
        return s

    def daide_to_english(self, tree: Union[list, str], form: Optional[str] = None, rec_level: int = 0) -> str:
        if isinstance(tree, list):
            if form == 'N LIST':
                result = ''
                n = len(tree)
                for i in range(n):
                    result += self.daide_to_english(tree[i], 'N', rec_level+1)
                    if i < n-2:
                        result += ', '
                    elif i == n-2:
                        result += ' and '
                return result
            # move order
            elif len(tree) == 3 \
                    and isinstance(tree[1], str) \
                    and tree[1] == 'MTO':
                unit_name = self.daide_to_english(tree[0], '', rec_level+1)
                destination_name = self.daide_to_english(tree[2], '', rec_level+1)
                if form and re.match(r'.*\border\b', form):
                    return self.ann(f"{unit_name} shall move to {destination_name}", synt='snt')
                else:
                    return self.ann(f"{unit_name} moved to {destination_name}", synt='snt')
            # hold order
            elif len(tree) == 2 \
                    and isinstance(tree[1], str) \
                    and tree[1] == 'HLD':
                unit_name = self.daide_to_english(tree[0], '', rec_level+1)
                if re.match(r'.*\border\b', form):
                    return self.ann(f"{unit_name} shall remain in place", synt='snt')
                else:
                    return self.ann(f"{unit_name} remained in place", synt='snt')
            # support order
            elif len(tree) in (3, 5) \
                    and isinstance(tree[1], str) \
                    and tree[1] == 'SUP':
                unit1_name = self.daide_to_english(tree[0], '', rec_level+1)
                unit2_name = self.daide_to_english(tree[2], '', rec_level+1)
                support_object = unit2_name
                if len(tree) == 5 and isinstance(tree[3], str):
                    if tree[3] == 'MTO':
                        destination_name = self.daide_to_english(tree[4], '', rec_level+1)
                        support_object = f"{unit2_name} moving to {destination_name}"
                    else:
                        support_object = f"{unit2_name} " \
                                         f"{self.daide_to_english(tree[3], '', rec_level+1)} " \
                                         f"{self.daide_to_english(tree[4], '', rec_level+1)}"
                if re.match(r'.*\border\b', form):
                    return self.ann(f"{unit1_name} shall support {support_object}", synt='snt')
                else:
                    return self.ann(f"{unit1_name} supported {support_object}", synt='snt')
            elif isinstance(tree[0], str):
                name = (self.power_name.get(tree[0], None)
                                   or self.province_name.get(tree[0], None)
                                   or self.sea_name.get(tree[0], None))

                if len(tree) == 1 \
                        and name:
                    return name
                # the English fleet in Liverpool
                else:
                    power_name = self.power_name.get(tree[0], None)
                    unit_type_name = self.unit_type_name.get(tree[1], None)
                    location_name = self.daide_to_english(tree[2], '', rec_level+1)
                    if len(tree) == 3 \
                        and isinstance(tree[0], str) \
                        and isinstance(tree[1], str) \
                        and power_name \
                        and unit_type_name \
                        and location_name:
                        power_pertainym = self.pertainym.get(power_name, power_name)
                        location_prep = 'on' if self.sem_annotation.get(location_name,'') == 'coast' else 'in'
                        return f"the {power_pertainym} {unit_type_name} {location_prep} {location_name}"
                    else:
                        location_name = self.province_name.get(tree[0], None)
                        coast_name = self.coast_name.get(tree[1], None)
                        if len(tree) == 2 \
                            and isinstance(tree[0], str) \
                            and isinstance(tree[1], str) \
                            and location_name \
                            and coast_name:
                            location_def_article_clause = "the " if self.name_uses_def_article.get(location_name, None) else ""
                            return self.ann(f"the {coast_name} of {location_def_article_clause}{location_name}", sem="coast")
                # elif len(tree) == 3 \
                #         and isinstance(tree[0], str) \
                #         and isinstance(tree[1], str) \
                #         and power_name \
                #         and unit_type_name \
                #         and location_name:
                #     power_pertainym = self.pertainym.get(power_name, power_name)
                #     location_prep = 'on' if self.sem_annotation.get(location_name,'') == 'coast' else 'in'
                #     return f"the {power_pertainym} {unit_type_name} {location_prep} {location_name}"
                # the south coast of Spain
                # elif len(tree) == 2 \
                #         and isinstance(tree[0], str) \
                #         and isinstance(tree[1], str) \
                #         and (location_name := self.province_name.get(tree[0], None)) \
                #         and (coast_name := self.coast_name.get(tree[1], None)):
                #     location_def_article_clause = "the " if self.name_uses_def_article.get(location_name, None) else ""
                #     return self.ann(f"the {coast_name} of {location_def_article_clause}{location_name}", sem="coast")
                # submit order
                        else:
                            if tree[0] == 'SUB' and len(tree) >= 2:
                                if len(tree) == 2:
                                    return self.ann("we submit the following order: "
                                                    + self.daide_to_english(tree[1], 'order', rec_level+1), synt='snt')
                                else:
                                    result = "we submit the following orders:"
                                    n = len(tree)
                                    for i in range(1, len(tree)):
                                        result += f" ({i}) {self.daide_to_english(tree[i], 'order', rec_level+1)}"
                                        if i < n-2:
                                            result += ";"
                                        elif i < n-1:
                                            result += "; and"
                                    return self.ann(result, synt='snt')
                        # proposal
                            else:
                                if tree[0] == 'PRP' and len(tree) >= 2:
                                    return self.ann('we propose ' + self.daide_to_english(tree[1], 'COMPL', rec_level+1), synt='snt')
                            # alliance
                                else:
                                    if tree[0] == 'ALY':
                                        allies = self.daide_to_english(tree[1], 'N LIST', rec_level+1) if len(tree) >= 2 else None
                                        enemies = self.daide_to_english(tree[2], 'N LIST', rec_level+1) if len(tree) >= 3 else None
                                        enemy_clause = f' against {enemies}' if enemies else ''
                                        if allies:
                                            if form in ('COMPL', 'N'):
                                                return f'an alliance between {allies}{enemy_clause}'
                                            else:
                                                return self.ann(f'{allies} are allies{enemy_clause}', synt='snt')
                                        else:
                                            return "an alliance"
            result = ''
            if rec_level:
                result += '('
            n = len(tree)
            for i in range(n):
                result += self.daide_to_english(tree[i], '', rec_level+1)
                if i < n-1:
                    result += ' '
            if rec_level:
                result += ')'
            return result
        elif isinstance(tree, str):
            location_name = self.province_name.get(tree, None) or self.sea_name.get(tree, None)
            if location_name :
                return f"the {location_name}" if self.name_uses_def_article.get(location_name) else location_name
            name = self.to_name.get(tree, '')
            if name :
                return name
            else:
                return tree
        return '?E'


if __name__ == "__main__":
    resource_filename = os.path.join(data_dir_path, 'diplomacy-resources.txt')
    daide = Daide(resource_filename)
    print(f'resource_filename: {resource_filename}')
    for daide_expr in ('AUS',
                       'ALY (GER AUS ITA) (FRA RUS)',
                       'PRP (ALY (GER AUS ITA) (FRA RUS))',
    #                  'SCO (AUS BUD TRI VIE)( ENG LPL EDI LON)',
                       'ITA FLT VEN',
                       '(FRA FLT (SPA NCS)) MTO MAO',
                       'SUB ((ENG AMY LVP) HLD) ((ENG FLT LON) MTO NTH) ((ENG FLT EDI) SUP (ENG FLT LON) MTO NTH)'
    #                  'WHY (THK (arrangement))'
                       ):
        tree, errors, index = daide.parse_daide_tree(daide_expr)
        print('DAIDE:', daide_expr)
        for error in errors:
            print('***', error)
    #   print('Mirror:', daide.print_daide_tree(tree))
        english = daide.daide_to_english(tree)
        if daide.synt_annotation.get(english, '') == 'snt':
            english = english[0].upper() + english[1:] + '.'
        print('Engl.:', english)
        print()
