import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import regex
import sys
from typing import Optional, Tuple, Union
from daide2eng.utils import pre_process,is_daide,create_daide_grammar


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

    def string_to_amr(self, s: str, parent: Optional[AMRnode] = None, rec_level: int = 0):

        errors = []
        snt = None
        snt_id = None
        amr_s = None

        while True:
            m2 = re.match(r'\s*(#[^\n]*)\n(.*)', s, re.DOTALL)
            if not m2:
                break
            comment_line = m2.group(1).strip()
            s = m2.group(2)
            snt_cand = slot_value_in_double_colon_del_list(comment_line, 'snt')
            if snt_cand:
                snt = snt_cand
            else:
                snt_id_cand = slot_value_in_double_colon_del_list(comment_line, 'id')
                if snt_id_cand:
                    snt_id = snt_id_cand

        m1 = re.match(r'(\(.*?\n(?:[ \t]+.*\S\s*?\n)*)', s)
        if snt_id and m1:
            amr_s = m1.group(1)

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
                role, s = m_role.group(1, 2)
                if re.match(r'\s*\(', s, re.DOTALL):
                    sub_amr, s, sub_errors, _snt_id, _snt, _amr_s = self.string_to_amr(s, rec_level=rec_level+1)
                    errors.extend(sub_errors)
                    if sub_amr:
                        amr_node.subs.append((role, sub_amr))
                        sub_amr.parents.append(amr_node)
                    else:
                        errors.append(f'Unexpected non-AMR: {s}')
                        return amr_node, s, errors, snt_id, snt, amr_s
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

            m_rest = re.match(r'\s*\)(.*)', s, re.DOTALL)

            if m_rest:
                s = m_rest.group(1)
            else:
                errors.append(f'Inserting missing ) at: {snt_id or s}')
            if rec_level == 0:
                for ref_variable in self.orphan_variables.keys():
                    ref_amr_node = self.variable_to_amr_node.get(ref_variable)
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
    def parents(amr_node):
    #-> list[AMRnode]:
        return amr_node.parents

    def parent_is_in_concepts(self, amr_node, concepts) -> bool:
        for parent in self.parents(amr_node):
            if parent.concept in concepts:
                return True
        return False

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



def main():
    amr = AMR()
    amr_s = '(f / fear-01\n      :ARG0 (c / country :name (n / name :op1 \"Germany\"))\n      :ARG1 (m / move-01\n            :ARG1 (u / unit\n                  :mod (c2 / country :name (n2 / name :op1 \"Italy\")))\n            :ARG2 (p / province :name (n3 / name :op1 \"Tyrolia\"))))'
    amr_tuple = amr.string_to_amr(amr_s)
    amr_tuple[0].concept = 'fear-02'
    amr_s2 = amr.amr_to_string()
    print(amr_s2)
    print(amr_tuple[0])
    dfs_recursive(amr_tuple[0])

def dfs_recursive(node):
    if isinstance(node, AMRnode):
        print(node.subs)
        for role, sub in node.subs:
            # if 'op' in role:
                #print(sub)
            dfs_recursive(sub)


if __name__ == "__main__":
    main()