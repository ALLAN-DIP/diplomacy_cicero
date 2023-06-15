#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Ulf Hermjakob, USC/ISI
"""

import argparse
from collections import defaultdict
from daide import Daide
import json
import os
from pathlib import Path
import re
import regex
import sys
from typing import Optional, Tuple, Union
from daide2eng.utils import pre_process,is_daide,create_daide_grammar

data_dir = Path(__file__).parent.parent / 'data'
data_dir_path = str(data_dir.resolve())


def slot_value_in_double_colon_del_list(line: str, slot: str, default: Optional = None) -> str:
    """For a given slot, e.g. 'cost', get its value from a line such as '::s1 of course ::s2 ::cost 0.3' -> 0.3
    The value can be an empty string, as for ::s2 in the example above."""
    m = re.match(fr'(?:.*\s)?::{slot}(|\s+\S.*?)(?:\s+::\S.*|\s*)$', line)
    return m.group(1).strip() if m else default


def has_matching_outer_parentheses(s: str) -> bool:
    if not s.startswith('('):
        return False
    elif not s.endswith(')'):
        return False
    n_open_parentheses = 0
    for i, c in enumerate(s):
        if c == '(':
            n_open_parentheses += 1
        elif c == ')':
            n_open_parentheses -= 1
        if n_open_parentheses <= 0 and i < len(s)-1:
            return False
    return n_open_parentheses == 0


class AMRnode:
    def __init__(self, concept: str, parent=None, variable: Optional = None):
        self.concept = concept
        self.variable = variable
        self.parent = parent
        self.subs: list[Tuple[str, Union[AMRnode, str, None]]] = []  # role sub
        self.parents: list[AMRnode] = []


class AMR:
    def __init__(self):
        self.root: Optional[AMRnode] = None
        self.variable_to_amr_node = {}
        self.orphan_variables = defaultdict(list)
        self.previously_printed_variables = []
        self.countries = ['AUS','TUR','RUS','GER','ITA','ENG','FRA']

    def match_map(self, amr_node, d: dict, s: str):
        #while m3 := re.match(r'(.*?)\$([a-z][a-z0-9]*)(?![a-z0-9])(.*)$', s):
        while True:
            m3 = re.match(r'(.*?)\$([a-z][a-z0-9]*)(?![a-z0-9])(.*)$', s)
            if not m3:
                break
            pre, var, post = m3.group(1, 2, 3)
            value = d.get(var)
            if value is None:
                value = '$' + var
            elif ' ' in value \
                    and (not (has_matching_outer_parentheses(value))) \
                    and (not (pre.endswith('(') and post.startswith(')'))) \
                    and amr_node.concept != 'have-03' \
                    and amr_node.concept != 'attack-01' \
                    and amr_node.concept != 'support-01' \
                    and amr_node.concept != 'move-01' :
                value = '(' + value + ')'
            s = pre + value + post
        matches = re.findall(r'\b[A-Z]{3}\b', s)
        #s = 'PCE (ENG AUS ENG ITA AUS)'
        tokens = re.findall(r"(?=("+'|'.join(self.countries)+r"))", s)
        new_string1 = f"({' '.join(tokens)})"
        current_order = sorted(re.findall(r"(?=("+'|'.join(self.countries)+r"))", s))
        new_string2 = f"({' '.join(set(current_order))})"
        s = s.replace(new_string1,new_string2)

        if s.split()[0] == 'ALY_NOVSS':
            excluded_countries = [country for country in self.countries if country not in matches]
            combined_string = ' '.join(excluded_countries)
            s = s.replace('ALY_NOVSS','ALY')+' VSS ('+combined_string+')'
        if amr_node.concept == 'attack-01' and self.sub_amr_node_by_role(amr_node, ['polarity']) != '-' and [tpl[1] for tpl in amr_node.subs if tpl[0] == 'ARG1'][0].concept != 'province':
            s = f'NOT ({s})'
        if self.sub_amr_node_by_role(amr_node, ['polarity']) == '-' and amr_node.concept == 'expect-01':
            #s = f'NOT ({s})'
            h = re.search(r'\((.*?)\)', s).group(0)
            s = f'{s.split()[0]} (NOT {h})'
        elif self.sub_amr_node_by_role(amr_node, ['polarity']) == '-' and amr_node.concept != 'attack-01':
            s = f'NOT ({s})'
        if self.amr_has_unknown_sub(amr_node):
            d['warnings'] = self.extend_new_warnings(d.get('warnings', []), ['includes question'])
        #while m3 := re.match(r'(.*)\((\([^()]*\))\)(.*)$', s):
        while True:
            m3 = re.match(r'(.*)\((\([^()]*\))\)(.*)$', s)
            if not m3:
                break
            s = m3.group(1) + m3.group(2) + m3.group(3)
        # simplify AUS SUP ((FRA AMY TYR) MTO VEN) -> AUS SUP (FRA AMY TYR) MTO VEN
        s = re.sub(r'( SUP )\((\([^()]+\))( MTO [A-Z]{3})\)', r'\1\2\3', s)
        return s, d.get('warnings', [])

    def string_to_amr(self, s: str, parent: Optional[AMRnode] = None, rec_level: int = 0):
        #  -> \
        #     Tuple[Optional[AMRnode], str, list[str], Optional[str], Optional[str], Optional[str]]:
        # AMR_Node built, rest, error message, sentence_id, sentence
        errors = []
        snt = None
        snt_id = None
        amr_s = None
        #while m2 := re.match(r'\s*(#[^\n]*)\n(.*)', s, re.DOTALL):
        while True:
            m2 = re.match(r'\s*(#[^\n]*)\n(.*)', s, re.DOTALL)
            if not m2:
                break
            comment_line = m2.group(1).strip()
            s = m2.group(2)
            snt_cand = slot_value_in_double_colon_del_list(comment_line, 'snt')
            #if snt_cand := slot_value_in_double_colon_del_list(comment_line, 'snt'):
            if snt_cand:
                snt = snt_cand
            #elif snt_id_cand := slot_value_in_double_colon_del_list(comment_line, 'id'):
            else:
                snt_id_cand = slot_value_in_double_colon_del_list(comment_line, 'id')
                if snt_id_cand:
                    snt_id = snt_id_cand
        # indent = ' '*4*rec_level
        # if snt_id and (m1 := re.match(r'(\(.*?\n(?:[ \t]+.*\S\s*?\n)*)', s)):
        #     amr_s = m1.group(1)
        m1 = re.match(r'(\(.*?\n(?:[ \t]+.*\S\s*?\n)*)', s)
        if snt_id and m1:
            amr_s = m1.group(1)

        #if m3 := re.match(r'\s*\(([a-z]\d*)\s*/\s*([a-z][a-z0-9]*(?:-[a-z0-9]+)*)(.*)', s, re.DOTALL):
        m3 = re.match(r'\s*\(([a-z]\d*)\s*/\s*([a-z][a-z0-9]*(?:-[a-z0-9]+)*)(.*)', s, re.DOTALL)
        if m3:
            variable, concept, s = m3.group(1, 2, 3)
            amr_node = AMRnode(concept, parent=parent, variable=variable)
            self.variable_to_amr_node[variable] = amr_node
            if self.root is None:
                self.root = amr_node
            while True:
                m_role = re.match(r'\s*:([a-z][a-z0-9]*(?:-[a-z0-9]+)*)(.*)', s, re.DOTALL | re.IGNORECASE)
                if not m_role:
                    break
            #while m_role := re.match(r'\s*:([a-z][a-z0-9]*(?:-[a-z0-9]+)*)(.*)', s, re.DOTALL | re.IGNORECASE):
                role, s = m_role.group(1, 2)
                # sub is AMR
                if re.match(r'\s*\(', s, re.DOTALL):
                    sub_amr, s, sub_errors, _snt_id, _snt, _amr_s = self.string_to_amr(s, rec_level=rec_level+1)
                    errors.extend(sub_errors)
                    if sub_amr:
                        amr_node.subs.append((role, sub_amr))
                        sub_amr.parents.append(amr_node)
                    else:
                        errors.append(f'Unexpected non-AMR: {s}')
                        return amr_node, s, errors, snt_id, snt, amr_s
                # sub is quoated string
                else:
                    m_string = re.match(r'\s*"((?:\\"|[^"]+)*)"(.*)', s, re.DOTALL)
                    if m_string:
                        quoted_string_value, s = m_string.group(1, 2)
                        amr_node.subs.append((role, quoted_string_value))
                    else:
                        m_string = re.match(r'\s*"((?:\\"|[^"]+)*)"(.*)', s, re.DOTALL)
                        if m_string:
                            quoted_string_value, s = m_string.group(1, 2)
                            amr_node.subs.append((role, quoted_string_value))
                # elif m_string := re.match(r'\s*"((?:\\"|[^"]+)*)"(.*)', s, re.DOTALL):
                #     quoted_string_value, s = m_string.group(1, 2)
                #     amr_node.subs.append((role, quoted_string_value))
                # sub is reentrancy variable
                        else:
                            m_string = re.match(r'\s*([a-z]\d*)(?![a-z])(.*)', s, re.DOTALL)
                            if m_string:
                                ref_variable, s = m_string.group(1, 2)
                                ref_amr = self.variable_to_amr_node.get(ref_variable)
                                if ref_amr:
                                    amr_node.subs.append((role, ref_amr))
                                else:
                                    self.orphan_variables[ref_variable].append((amr_node, len(amr_node.subs)))
                                    amr_node.subs.append((role, None))
                            else:
                                m_string = re.match(r'\s*([^\s()]+)(.*)', s, re.DOTALL)
                                if m_string:
                                    unquoted_string_value, s = m_string.group(1, 2)
                                    amr_node.subs.append((role, unquoted_string_value))
                                else:
                                    errors.append(f'Unexpected :{role} arg: {s}')
                                    return amr_node, s, errors, snt_id, snt, amr_s
                # elif m_string := re.match(r'\s*([a-z]\d*)(?![a-z])(.*)', s, re.DOTALL):
                    # ref_variable, s = m_string.group(1, 2)
                    # if ref_amr := self.variable_to_amr_node.get(ref_variable):
                    #     amr_node.subs.append((role, ref_amr))
                    # else:
                    #     self.orphan_variables[ref_variable].append((amr_node, len(amr_node.subs)))
                    #     amr_node.subs.append((role, None))
                # sub is non-quoted string
                # elif m_string := re.match(r'\s*([^\s()]+)(.*)', s, re.DOTALL):
                #     unquoted_string_value, s = m_string.group(1, 2)
                #     amr_node.subs.append((role, unquoted_string_value))
                # else:
                #     errors.append(f'Unexpected :{role} arg: {s}')
                #     return amr_node, s, errors, snt_id, snt, amr_s
            m_rest = re.match(r'\s*\)(.*)', s, re.DOTALL)
            # if m_rest := re.match(r'\s*\)(.*)', s, re.DOTALL):
            if m_rest:
                s = m_rest.group(1)
            else:
                errors.append(f'Inserting missing ) at: {snt_id or s}')
            if rec_level == 0:
                for ref_variable in self.orphan_variables.keys():
                    ref_amr_node = self.variable_to_amr_node.get(ref_variable)
                    # if ref_amr_node := self.variable_to_amr_node.get(ref_variable):
                    if ref_amr_node:
                        for orphan_location in self.orphan_variables[ref_variable]:
                            parent_amr_node, child_index = orphan_location
                            role = parent_amr_node.subs[child_index][0]
                            parent_amr_node.subs[child_index] = (role, ref_amr_node)
                    else:
                        errors.append(f"Error: For {snt_id}, can't resolve orphan reference {ref_variable}")
            return amr_node, s, errors, snt_id, snt, amr_s
        else:
            return None, "", errors, snt_id, snt, amr_s

    def amr_to_string(self, amr_node: AMRnode = None, rec_level: int = 0, no_indent: bool = False) -> str:
        if amr_node is None:
            amr_node = self.root
            self.previously_printed_variables = []
        indent = ' '*6*(rec_level+1)
        concept = amr_node.concept
        variable = amr_node.variable
        result = f"({variable} / {concept}"
        for role, sub in amr_node.subs:
            if role == "name" and isinstance(sub, AMRnode) and sub.concept == "name":
                no_indent = True
            if no_indent:
                result += f" :{role} "
            else:
                result += f"\n{indent}:{role} "
            if isinstance(sub, AMRnode):
                if sub.variable in self.previously_printed_variables:
                    result += sub.variable
                else:
                    result += self.amr_to_string(amr_node=sub, rec_level=rec_level+1, no_indent=no_indent)
                    self.previously_printed_variables.append(sub.variable)
            elif isinstance(sub, str):
                result += f'"{sub}"'
            # elif isinstance(sub, int)
        result += ')'
        return result

    @staticmethod
    def sub_amr_node_by_role(amr_node, roles) -> Optional[Union[AMRnode, str]]:
        for role, sub in amr_node.subs:
            if role in roles:
                return sub
        return None

    @staticmethod
    def amr_has_unknown_sub(amr_node) -> bool:
        for role, sub in amr_node.subs:
            try:
                concept = sub.concept
            except AttributeError:
                pass
            else:
                if concept == 'amr-unknown':
                    return True
        return False

    @staticmethod
    def sub_amr_concept_by_role(amr_node, roles) -> Optional[str]:
        for role, sub in amr_node.subs:
            if role in roles:
                return sub.concept
        return None

    @staticmethod
    def parents(amr_node):
    #-> list[AMRnode]:
        return amr_node.parents

    #def parent_is_in_concepts(self, amr_node, concepts: list[str]) -> bool:
    def parent_is_in_concepts(self, amr_node, concepts) -> bool:
        for parent in self.parents(amr_node):
            if parent.concept in concepts:
                return True
        return False

    # def ancestor_is_in_concepts(self, amr_node, concepts: list[str],
    #                             visited_amr_nodes: Optional[list[AMRnode]] = None) -> bool:
    def ancestor_is_in_concepts(self, amr_node, concepts,
                                visited_amr_nodes: Optional = None) -> bool:
        # Avoid loops
        if visited_amr_nodes and amr_node in visited_amr_nodes:
            return False
        if visited_amr_nodes is None:
            visited_amr_nodes = []
        for parent in self.parents(amr_node):
            if parent.concept in concepts:
                return True
            visited_amr_nodes.append(amr_node)
            if self.ancestor_is_in_concepts(parent, concepts, visited_amr_nodes):
                return True
        return False

    def ne_amr_to_name(self, amr_node) -> str:
        if isinstance(amr_node, AMRnode) and amr_node.concept in ['country', 'province', 'sea']:
            name_amr_node = self.sub_amr_node_by_role(amr_node, ['name'])
            if name_amr_node and name_amr_node.concept == 'name':
                name_elements = []
                i = 1
                while True:
                    op = self.sub_amr_node_by_role(name_amr_node, [f"op{i}"])
                    if not op:
                        break
                    name_elements.append(op)
                    i += 1
                return ' '.join(name_elements)

        return ''

    def match_for_daide(self, amr_node: AMRnode, target_s: str, in_dict: Optional[dict] = None) -> Optional[dict]:
        warnings = []
        m2 = re.match(r'\((\S+)\s+(.*)\)$', target_s)
        #if m2 := re.match(r'\((\S+)\s+(.*)\)$', target_s):
        if m2:
            result = in_dict or {'match': True}
            concept = amr_node.concept
            instance_s = m2.group(1)
            if instance_s == concept:
                pass
            else:
                m2b = re.match(r'\$([a-z][a-z0-9]*)\((.*)\)$', instance_s)
                if (m2b and (concept in m2b.group(2).split('|'))):
                    result[m2b.group(1)] = daide.name_to_id.get(concept) or concept
                else:
                    m1 = re.match(r'\$([a-z][a-z0-9]*)$', instance_s)
                    if m1:
                        result[m1.group(1)] = daide.name_to_id.get(concept) or concept
                    else:
                        return None
            for arg_value in regex.findall(r':([a-z][-a-z0-9]*)\s+([^\s()]+(?:\([^\s()]+\))?'
                                           r'|(\((?:[^()]++|(?3))*\))(?:\([^\s()]+\))?)',
                                           m2.group(2), re.IGNORECASE):
                arg, value = arg_value[0], arg_value[1]
                sub_amr_node = self.sub_amr_node_by_role(amr_node, [arg])
                #if sub_amr_node := self.sub_amr_node_by_role(amr_node, [arg]):
                if sub_amr_node:
                    # if isinstance(sub_amr_node,AMRnode):
                    sub_concept = sub_amr_node.concept
                    if value == sub_concept:
                        pass
                    else:
                        m2c = (re.match(r'\$([a-z][a-z0-9]*)\((.*)\)$', value)
                                   or re.match(r'\$([a-z][a-z0-9]*)$', value))
                        if (m2c and ((m2c.lastindex == 1) or (sub_concept in m2c.group(2).split('|')))):
                            var = m2c.group(1)
                            sub_name = self.ne_amr_to_name(sub_amr_node)
                            # if sub_name := self.ne_amr_to_name(sub_amr_node):
                            if sub_name:
                                result[var] = daide.name_to_id.get(sub_name) or sub_name
                            elif sub_amr_node.subs:
                                result[var], sub_warnings = self.amr_to_daide(sub_amr_node, top=False) or sub_concept
                                warnings = self.extend_new_warnings(warnings, sub_warnings)
                            else:
                                result[var] = daide.name_to_id.get(sub_concept) or sub_concept
                        else:
                            if has_matching_outer_parentheses(value):
                                if self.match_for_daide(sub_amr_node, value, result) is None:
                                    return None
                            else:
                                return None
                else:
                    return None
            result['warnings'] = warnings
            return result
        return None

    @staticmethod
    # def extend_new_warnings(warnings: list[str], new_warnings: list[str]):
    def extend_new_warnings(warnings, new_warnings):
        for new_warning in new_warnings:
            if new_warning not in warnings:
                warnings.append(new_warning)
        return warnings

    def add_warning_to_match_dict(self, match_dict, warning):
        match_dict['warnings'] = self.extend_new_warnings(match_dict.get('warnings', []), [warning])



    def optional_AND(self,arrangements):
        """Wraps a list of arrangements in an `AND`.
        If the list has a single element, return that element instead.
        :param arrangements: List of arrangements.
        :return: Arrangement object.
        """
        arrangements = sorted(set(arrangements), key=str)
        if len(arrangements) > 1:
            return f"AND {' '.join(arrangements)}"
        else:
            return arrangements[0][1:-1]

    #def amr_to_daide(self, amr_node: AMRnode = None, top: bool = True) -> Tuple[str, list[str]]:
    def amr_to_daide(self, amr_node: AMRnode = None, top: bool = True):
        # returns pair of (daide_element, warnings)
        warnings = []
        if amr_node is None:
            amr_node = self.root
        entity_name = self.ne_amr_to_name(amr_node)
        daide_id = daide.name_to_id.get(entity_name)
        # if (entity_name := self.ne_amr_to_name(amr_node)) \
        #         and (daide_id := daide.name_to_id.get(entity_name)):
        if entity_name and daide_id:
            return daide_id, warnings
        concept = amr_node.concept
        if concept == 'and' or concept == 'or':
            daide_elements = []
            i = 1
            while True:
                op_amr_node = self.sub_amr_node_by_role(amr_node, [f"op{i}"])
                if not op_amr_node:
                    break
            #while op_amr_node := self.sub_amr_node_by_role(amr_node, [f"op{i}"]):
                daide_element, sub_warnings = self.amr_to_daide(op_amr_node, top=False)
                warnings = self.extend_new_warnings(warnings, sub_warnings)
                if daide_element:
                    if ' ' in daide_element and not has_matching_outer_parentheses(daide_element):
                        daide_element = '(' + daide_element + ')'
                    daide_elements.append(daide_element)
                    i += 1
            #if self.parent_is_in_concepts(amr_node, ['ally-01', 'demilitarize-01','attack-01','have-03']):
            if self.parent_is_in_concepts(amr_node, ['ally-01', 'demilitarize-01','have-03','attack-01','move-01']):
                return ' '.join(sorted(daide_elements)), warnings
            else:
                result = self.optional_AND(sorted(daide_elements))
                return result, warnings

        
        d = self.match_for_daide(amr_node,
                                     '($utype(army|fleet) :mod $power(country) :location $location(sea|province))')
        # if d := self.match_for_daide(amr_node,
        #                              '($utype(army|fleet) :mod $power(country) :location $location(sea|province))'):
        if d:
            return self.match_map(amr_node, d, '($power $utype $location)')
        # elif self.match_for_daide(amr_node,
        #                              '($utype(army|fleet) :mod $power(country)'):
        #     m =self.match_for_daide(amr_node,
        #                               '($utype(army|fleet) :mod $power(country)')
        #     return self.match_map(amr_node, m, '$power $utype')

        d = self.match_for_daide(amr_node, '(move-01 :ARG1 $unit :ARG2 $destination)')
        if d:
            unit = d.get('unit', '')
            if top:
                self.add_warning_to_match_dict(d, 'MTO at top level')
            if re.match(r'^\([A-Z]{3} (?:AMY|FLT) ', unit):
                return self.match_map(amr_node, d, 'XDO ($unit MTO $destination)')
            else:
                match = re.findall(r'\b[A-Z]{3}\b', unit)
                if match:
                    d['unit'] = match[0]
                    if self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                        return self.match_map(amr_node, d, 'SCD ($unit $destination)')
                    else:
                        s = self.match_map(amr_node, d, 'SCD ($unit $destination)')
                        s_list = list(s)
                        s_list[0] = 'PRP (' + s_list[0] + ')'
                        s_modified = tuple(s_list)
                        return s_modified
                        #return self.match_map(amr_node, d, 'PRP (SCD ($unit $destination))')





        d = self.match_for_daide(amr_node, '(coast :location ($compass(north|east|south|west)'
                                               ' :part-of $province(province)))')    
        if d:
            return self.match_map(amr_node, d, '($province $compass)')
        d = self.match_for_daide(amr_node, '(hold-03 :ARG1 $unit)')
        if d :
            unit = d.get('unit', '')
            if top:
                self.add_warning_to_match_dict(d, 'HLD at top level')
            if not re.match(r'^\([A-Z]{3} (?:AMY|FLT) ', unit):
                self.add_warning_to_match_dict(d, f"HLD unit must be a specific unit, not {unit}")
            return self.match_map(amr_node, d, '$unit HLD')
        d = self.match_for_daide(amr_node, '(support-01 :ARG0 $supporter :ARG1 $supportee)')
        if d :
            supporter = d.get('supporter', '')
            supportee = d.get('supportee', '')
            if not re.match(r'^\([A-Z]{3} (?:AMY|FLT) ', supporter):
                self.add_warning_to_match_dict(d, f"SUP supporter must be a specific unit, not {supporter}")
            if not re.match(r'^\([A-Z]{3} (?:AMY|FLT) ', supportee):
                self.add_warning_to_match_dict(d, f"SUP supportee must be a specific unit, not {supportee}")
            if top:
                self.add_warning_to_match_dict(d, 'SUP at top level')
            if supportee.startswith('SCD'):
                return self.match_map(amr_node, d, '$supportee')
            else:
                return self.match_map(amr_node, d, 'XDO ($supporter SUP $supportee)')



        d = self.match_for_daide(amr_node, '(ally-01 :ARG1 $allies :ARG2 $countries :ARG3 $ennemies)')
        if d :
            ennemies = d.get('ennemies', '')
            print(ennemies)
            ennemies_list = ennemies.split()
            if top:
                self.add_warning_to_match_dict(d, 'ALY at top level')
            if all(country in self.countries for country in ennemies_list):
            # if ennemies in self.countries:
                if self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                    return self.match_map(amr_node, d, 'ALY ($allies $countries) VSS ($ennemies)')
                else:
                    s = self.match_map(amr_node, d, 'ALY ($allies $countries) VSS ($ennemies)')
                    s_list = list(s)
                    s_list[0] = 'PRP (' + s_list[0] + ')'
                    s_modified = tuple(s_list)
                    return s_modified
                # return self.match_map(amr_node, d, 'PRP (ALY ($allies) VSS ($ennemies))')
            else:
                if self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                    return self.match_map(amr_node, d, 'ALY_NOVSS ($allies $countries)')
                else:
                    s = self.match_map(amr_node, d, 'ALY_NOVSS ($allies $countries)')
                    s_list = list(s)
                    s_list[0] = 'PRP (' + s_list[0] + ')'
                    s_modified = tuple(s_list)
                    return s_modified

        d = self.match_for_daide(amr_node, '(ally-01 :ARG1 $allies :ARG2 $countries)')
        if d :
            self.add_warning_to_match_dict(d, 'ALY without VSS')
            if top:
                self.add_warning_to_match_dict(d, 'ALY at top level')
            if self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                return self.match_map(amr_node, d, 'ALY_NOVSS ($allies $countries)')
            else:
                s = self.match_map(amr_node, d, 'ALY_NOVSS ($allies $countries)')
                s_list = list(s)
                s_list[0] = 'PRP (' + s_list[0] + ')'
                s_modified = tuple(s_list)
                return s_modified

        d = self.match_for_daide(amr_node, '(ally-01 :ARG1 $allies :ARG3 $ennemies)')
        if d :
            print(d)
            ennemies = d.get('ennemies', '')
            print(ennemies)
            ennemies_list = ennemies.split()
            if top:
                self.add_warning_to_match_dict(d, 'ALY at top level')
            if all(country in self.countries for country in ennemies_list):
                if self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                    return self.match_map(amr_node, d, 'ALY ($allies) VSS ($ennemies)')
                else:
                    s = self.match_map(amr_node, d, 'ALY ($allies) VSS ($ennemies)')
                    s_list = list(s)
                    s_list[0] = 'PRP (' + s_list[0] + ')'
                    s_modified = tuple(s_list)
                    return s_modified
                # return self.match_map(amr_node, d, 'PRP (ALY ($allies) VSS ($ennemies))')
            else:
                if self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                    return self.match_map(amr_node, d, 'ALY_NOVSS ($allies)')
                else:
                    s = self.match_map(amr_node, d, 'ALY_NOVSS ($allies)')
                    s_list = list(s)
                    s_list[0] = 'PRP (' + s_list[0] + ')'
                    s_modified = tuple(s_list)
                    return s_modified


        d = self.match_for_daide(amr_node, '(ally-01 :ARG1 $allies)')
        if d :
            self.add_warning_to_match_dict(d, 'ALY without VSS')
            if top:
                self.add_warning_to_match_dict(d, 'ALY at top level')
            if self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                return self.match_map(amr_node, d, 'ALY_NOVSS ($allies)')
            else:
                s = self.match_map(amr_node, d, 'ALY_NOVSS ($allies)')
                s_list = list(s)
                s_list[0] = 'PRP (' + s_list[0] + ')'
                s_modified = tuple(s_list)
                return s_modified



        d = self.match_for_daide(amr_node, '(submit-01 :ARG1 $submission)')
        if d :
            return self.match_map(amr_node, d, 'SUB $submission')
        d = self.match_for_daide(amr_node, '(propose-01 :ARG1 $proposal(build-01|hold-03|remove-01'
                                               '|retreat-01|transport-01))')
        if d :
            return self.match_map(amr_node, d, 'PRP (XDO ($proposal))')
        d = self.match_for_daide(amr_node, '(propose-01 :ARG1 $proposal)')
        if d :
            return self.match_map(amr_node, d, 'PRP ($proposal)')
        d = self.match_for_daide(amr_node, '(build-01 :ARG0 $power(country) :ARG1 $utype(army|fleet) '
                                               ':location $location(province))')
        if d :
            if top:
                self.add_warning_to_match_dict(d, 'BLD at top level')
            return self.match_map(amr_node, d, '($power $utype $location) BLD')
        d = self.match_for_daide(amr_node, '(agree-01 :ARG1 $proposal(build-01|hold-03|remove-01'
                                               '|retreat-01|support-01|transport-01))')
        if d :
            return self.match_map(amr_node, d, 'YES (PRP (XDO ($proposal)))')
        d = self.match_for_daide(amr_node, '(agree-01 :ARG1 $proposal(ally-01|demilitarize-01|have-03|peace|move-01))')
        if d :
            return self.match_map(amr_node, d, 'YES (PRP ($proposal))')
        d = self.match_for_daide(amr_node, '(agree-01 :ARG1 $proposal)')
        if d :
            return self.match_map(amr_node, d, 'YES ($proposal)')
        d = self.match_for_daide(amr_node, '(agree-01 :ARG0 $country1 :ARG2 $country2)')
        if d :
            return self.match_map(amr_node, d, 'YES_LAST')
        d = self.match_for_daide(amr_node, '(reject-01 :ARG1 $proposal)')
        if d :
            return self.match_map(amr_node, d, 'REJ ($proposal)')
        d = self.match_for_daide(amr_node, '(demilitarize-01 :ARG1 $powers :ARG2 $locations)')
        if d :
            for location_id in re.findall(r'[A-Z]+', d.get('locations', '')):
                if not daide.province_name.get(location_id):
                    self.add_warning_to_match_dict(d, f"DMZ locations must be provinces, not {location_id}")
            if top:
                self.add_warning_to_match_dict(d, 'DMZ at top level')
            return self.match_map(amr_node, d, 'DMZ ($powers) ($locations)')
        d = self.match_for_daide(amr_node, '(remove-01 :ARG1 $unit(army|fleet))')
        if d :
            if top:
                self.add_warning_to_match_dict(d, 'REM at top level')
            return self.match_map(amr_node, d, '$unit REM')
        d = self.match_for_daide(amr_node, '(transport-01 :ARG1 $army(army) :ARG3 $destination(province) '
                                               ':ARG4 $path(sea))')
        if d :
            if top:
                self.add_warning_to_match_dict(d, 'CTO at top level')
            return self.match_map(amr_node, d, '$army CTO $destination VIA $path')
        d = self.match_for_daide(amr_node, '(transport-01 :ARG0 $fleet(fleet) :ARG1 $army(army) '
                                               ':ARG3 $destination(province))')
        if d :
            if top:
                self.add_warning_to_match_dict(d, 'CVY at top level')
            return self.match_map(amr_node, d, '$fleet CVY $army CTO $destination')
        d = self.match_for_daide(amr_node, '(retreat-01 :ARG1 $unit(army|fleet) '
                                               ':destination $destination(province|sea))')
        if d :
            if top:
                self.add_warning_to_match_dict(d, 'RTO at top level')
            return self.match_map(amr_node, d, '$unit RTO $destination')

        d = self.match_for_daide(amr_node, '(have-03 :ARG0 $owner(country) :ARG1 $province)')
        if d:
            if self.ancestor_is_in_concepts(amr_node, ['propose-01', 'agree-01','possible-01','expect-01']):
                return self.match_map(amr_node, d, 'SCD ($owner $province)')
            else:
                return self.match_map(amr_node, d, 'PRP (SCD ($owner $province))')

        # d = self.match_for_daide(amr_node, '(have-03 :ARG0 $owner(country) :ARG1 $province)')
        # if d :
        #     return self.match_map(amr_node, d, 'SCD ($owner $province)')



        d = self.match_for_daide(amr_node, '(betray-01 :ARG0 $owner1(country) :ARG1 $owner2(country))')
        if d :
            if self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                return self.match_map(amr_node, d, 'NOT (PCE ($owner1 $owner2))')
            else:
                return self.match_map(amr_node, d, 'FCT (NOT (PCE ($owner1 $owner2)))')
        # d = self.match_for_daide(amr_node, '(attack-01 :ARG0 $owner1(country) :ARG1 $owner2(country))')
        # if d :
        #     return self.match_map(amr_node, d, 'PCE ($owner1 $owner2)')


        d = self.match_for_daide(amr_node, '(trust-01 :ARG0 $owner1(country) :ARG2 $owner2(country))')
        if d :
            if self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                return self.match_map(amr_node, d, 'ALY_NOVSS ($owner1 $owner2)')
            else:
                s = self.match_map(amr_node, d, 'ALY_NOVSS ($owner1 $owner2)')
                s = ('FCT ('+s[0]+')',s[1])
                return s
        
        d = self.match_for_daide(amr_node, '(expect-01 :ARG1 $proposal(build-01|hold-03|remove-01'
                                               '|retreat-01|transport-01))')
        if d :
            return self.match_map(amr_node, d, 'THK (XDO ($proposal))')

        d = self.match_for_daide(amr_node, '(expect-01 :ARG1 $thoughts)')
        if d :
            return self.match_map(amr_node, d, 'THK $thoughts')

        d = self.match_for_daide(amr_node, '(possible-01 :ARG1 $proposal(build-01|hold-03|remove-01'
                                               '|retreat-01|transport-01))')
        if d :
            return self.match_map(amr_node, d, 'THK (XDO ($proposal))')

        d = self.match_for_daide(amr_node, '(possible-01 :ARG1 $thoughts)')
        if d :
            return self.match_map(amr_node, d, 'THK $thoughts')
        
        # d = self.match_for_daide(amr_node, '(attack-01 :ARG0 $unit :ARG1 $province)')
        # if d:
        #     print('yes1')
        #     if top:
        #         self.add_warning_to_match_dict(d, 'MTO at top level')
        #     return self.match_map(amr_node, d, '$unit MTO $province')
        
        d = self.match_for_daide(amr_node, '(attack-01 :ARG0 $allies :ARG1 $ennemies)')
        if d:
            value = [tpl[1] for tpl in amr_node.subs if tpl[0] == 'ARG1']
            if value[0].concept == 'province':
                return self.match_map(amr_node, d, 'XDO ($allies MTO $ennemies)')
                #return self.match_map(amr_node, d, 'SCD ($allies $ennemies)')
            elif self.ancestor_is_in_concepts(amr_node, ['possible-01','propose-01','agree-01','expect-01']):
                return self.match_map(amr_node, d, 'PCE ($allies $ennemies)')
            else:
                s = self.match_map(amr_node, d, 'PCE ($allies $ennemies)')
                s_list = list(s)
                s_list[0] = 'PRP (' + s_list[0] + ')'
                s_modified = tuple(s_list)
                return s_modified
                #return self.match_map(amr_node, d, 'PRP (PCE ($allies $ennemies))')


        d = self.match_for_daide(amr_node, '(peace :op1 $c1(country) :op2 $c2(country) :op3 $c3(country))')
        if d :
            return self.match_map(amr_node, d, 'PCE ($c1 $c2 $c3)')
        d = self.match_for_daide(amr_node, '(peace :op1 $c1(country) :op2 $c2(country))')
        if d :
            if top:
                self.add_warning_to_match_dict(d, 'PCE at top level')
            return self.match_map(amr_node, d, 'PCE ($c1 $c2)')
        result = f'({concept}'
        for role, sub in amr_node.subs:
            result += f" :{role} "
            if isinstance(sub, AMRnode):
                daide_element, sub_warnings = self.amr_to_daide(amr_node=sub, top=False)
                warnings = self.extend_new_warnings(warnings, sub_warnings)
                result += daide_element
            elif isinstance(sub, str):
                result += f'"{sub}"'
        result += ')'
        return result, warnings

    @staticmethod
    def file_to_amrs(filename: str, max_n: Optional[int]) -> Tuple[list, list, list, list, list]:
        s = ""
        with open(filename) as f:
            s += f.read()
        n_amrs = 0
        amr_roots = []
        snt_ids = []
        snts = []
        errors = []
        amr_strings = []
        while re.match(r'\s*\S', s):
            orig_s = s
            amr = AMR()
            amr_node, s, error_list, snt_id, snt, amr_s = amr.string_to_amr(s)
            if amr_node:
                amr.root = amr_node
                amr_roots.append(amr)
                snt_ids.append(snt_id)
                snts.append(snt)
                errors.append(error_list)
                amr_strings.append(amr_s)
                n_amrs += 1
            else:
                print(f'Break at\n{orig_s[0:500]}')
                print(f'{0/0}')
                break
            if max_n is not None and n_amrs >= max_n:
                break
        return amr_roots, snt_ids, snts, errors, amr_strings


daide = Daide(os.path.join(data_dir_path, 'diplomacy-resources.txt'))
extended_amr_concepts1 = ['attack-01', 'betray-01', 'defend-01', 'dislodge-01', 'expect-01', 'fear-01',
                          'gain-02', 'lie-08', 'lose-02', 'possible-01', 'prevent-01', 'threaten-01',
                          'trust-01', 'warn-01']


def main_test():
    amr = AMR()
    amr_s = '(a / army :mod (c / country :name (n / name :op1 "Italy")) ' \
            ':location (p / province :name (n2 / name :op1 "Burgundy")))'
    print(amr_s)
    target_s = "($unit(army|fleet) :mod $power(country) :location $loc(province|sea))"
    print(target_s)
    amr_tuple = amr.string_to_amr(amr_s)
    amr_node = amr_tuple[0]
    result = amr.match_for_daide(amr_node, target_s)
    print(result)


def main():
    parser = argparse.ArgumentParser(description='Maps AMR to DAIDE in classical or jsonl format')
    parser.add_argument('-i', '--input', type=Path, metavar='AMR-INPUT-FILENAME')
    parser.add_argument('-o', '--output', type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                        default=sys.stdout, metavar='OUTPUT-FILENAME', help='(default: STDOUT)')
    parser.add_argument('-j', '--json', type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                        default=None, metavar='JSONL-OUTPUT-FILENAME', help='(default: None)')
    parser.add_argument('-m', '--max', type=int, default=None, help='(maximum number of AMRs in ouput)')
    parser.add_argument('-d', '--developer_mode', action='count', default=0)
    parser.add_argument('-v', '--verbose', action='count', default=0, help='write change log etc. to STDERR')
    args = parser.parse_args()
    n_amrs = 0
    n_amr_empty = 0
    n_amr_with_extended_concept1 = 0
    n_amr_with_extended_concept2 = 0
    n_underspecified_unit = 0
    n_daide_without_problem = 0
    extended_concept_counter1 = defaultdict(int)
    extended_concept_counter2 = defaultdict(int)
    snt_ids_with_recursion_error = []
    amrs, snt_ids, snts, errors, amr_strings \
        = AMR.file_to_amrs(args.input, args.max)
    out = args.output
    grammar = create_daide_grammar(level=130)
    for amr, snt_id, snt, error_list, amr_s in zip(amrs, snt_ids, snts, errors, amr_strings):
        n_amrs += 1
        daide_problematic = False
        show_daide_in_dev_mode = True
        try:
            amr_s2 = amr.amr_to_string()
            print(f'No error for {snt_id}')
        except:
            snt_ids_with_recursion_error.append(snt_id)
            print(f'RecursionError for {snt_id}')
            continue
        if amr_s2 == '(a / amr-empty)':
            n_amr_empty += 1
            show_daide_in_dev_mode = False
            daide_s, warnings = '', []
        else:
            daide_s, warnings = amr.amr_to_daide()
            extended_amr_concepts = re.findall(r'([a-z]\S*-\d\d\b)', daide_s)
            if extended_amr_concepts :
                if set(extended_amr_concepts1) & set(extended_amr_concepts):
                    n_amr_with_extended_concept1 += 1
                    for extended_amr_concept in extended_amr_concepts:
                        if extended_amr_concept in extended_amr_concepts1:
                            extended_concept_counter1[extended_amr_concept] += 1
                    show_daide_in_dev_mode = False
                else:
                    n_amr_with_extended_concept2 += 1
                    for extended_amr_concept in extended_amr_concepts:
                        extended_concept_counter2[extended_amr_concept] += 1
                daide_problematic = True
            if '(unit ' in daide_s \
                    or re.search(r'\((?:army|fleet) :(?:mod|location) [A-Z]{3}\)', daide_s):
                n_underspecified_unit += 1
                daide_problematic = True
                show_daide_in_dev_mode = False
            if re.search('[a-z]', daide_s):
                daide_problematic = True
        try:
            parse_tree = grammar.parse(daide_s)
            Full = True
            #print('True')
        except:
            Full = False
            #print('False')
        if regex.search(r'[A-Z]{3}', daide_s):
            if regex.search(r'[a-z]', daide_s):
                daide_status = 'Partial-DAIDE'
            elif Full == False:
                daide_status = 'Para-DAIDE'
            else:
                daide_status = 'Full-DAIDE'
        else:
            daide_status = 'No-DAIDE'
        if ((not args.developer_mode) or show_daide_in_dev_mode) and not (args.json and args.output == sys.stdout):
            out.write(f'# ::id {snt_id}\n')
            out.write(f'# ::snt {snt}\n')
            for error in error_list:
                out.write(f'# ::error {error}\n')
            out.write(f'AMR:\n{amr_s.strip()}\n')
            # out.write(f'AMR string (r): {amr_s2.strip()}\n')
            if daide_status == 'Full-DAIDE':
                out.write(f'FULL-DAIDE: {daide_s}\n')
            elif daide_status == 'Partial-DAIDE':
                out.write(f'PARTIAL-DAIDE: {daide_s}\n')
            elif daide_status == 'Para-DAIDE':
                out.write(f'PARA-DAIDE: {daide_s}\n')
            else:
                out.write('NO-DAIDE\n')
            out.write('\n')
        if args.json:
            d = {'id': snt_id, 'snt': snt, 'amr': amr_s.strip(), 'daide-status': daide_status}
            if daide_s:
                d['daide'] = daide_s
                if warnings:
                    d['warnings'] = warnings
            args.json.write(json.dumps(d) + "\n")
        if not daide_problematic:
            n_daide_without_problem += 1
    if args.developer_mode:
        out.write(f'Summary: {n_amrs} AMRs; {n_amr_empty} empty AMRs; {n_daide_without_problem} unproblematic; '
                  f'{n_underspecified_unit} underspecified units; '
                  f'{n_amr_with_extended_concept1}/{n_amr_with_extended_concept2} AMRs with extended concept\n')
        out.write(f'Last snt-id: {snt_ids[-1]}')
        if snt_ids_with_recursion_error:
            sys.stderr.write(f'Recursion errors for {snt_ids_with_recursion_error}\n')
        for extended_concept in sorted(extended_concept_counter1.keys()):
            sys.stderr.write(f'  Extended concept {extended_concept} ({extended_concept_counter1[extended_concept]})\n')
        sys.stderr.write('\n')
        for extended_concept in sorted(extended_concept_counter2.keys()):
            sys.stderr.write(f'  Extended concept {extended_concept} ({extended_concept_counter2[extended_concept]})\n')


if __name__ == "__main__":
    main()
