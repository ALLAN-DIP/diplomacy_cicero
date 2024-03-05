#!/usr/bin/env python3

"""
Written by Ulf Hermjakob, USC/ISI, in May 2022
This script checks the validity of an AMR workset (.txt) and any companion (.info).
It outputs errors, warnings, number of AMR worksets checked.
Its arguments are AMR workset filesnames or directories.
Companion files (.info) are automatically checked.
"""

import logging as log
import os
from pathlib import Path
import re
import sys
from typing import Optional

log.basicConfig(level=log.INFO)


def slot_value_in_double_colon_del_list(line: str, slot: str, default: Optional = None) -> str:
    m = re.match(fr'(?:.*\s)?::{slot}(|\s+\S.*?)(?:\s+::\S.*|\s*)$', line)
    return m.group(1).strip() if m else default


def usage():
    return f'Usage: {sys.argv[0]} [-h|--help] [filename(s)] [directory(s)]'


def main():
    dirs = []
    workset_files = []
    info_files = []
    help_printed = False
    if len(sys.argv) <= 1:
        print(usage())
        help_printed = True
    else:
        for arg in sys.argv[1:]:
            path = Path(arg)
            if path.is_dir():
                dirs.append(path)
            elif path.is_file() and arg.endswith('.txt'):
                if not path in workset_files:
                    workset_files.append(path)
                info_file = re.sub('\.txt$', '.info', arg)
                info_path = Path(info_file)
                if info_path.is_file():
                    if not info_path in info_files:
                        info_files.append(info_path)
            elif path.is_file() and arg.endswith('.info'):
                if not path in info_files:
                    info_files.append(path)
            elif arg in ('-h', '--help'):
                print(usage())
                help_printed = True
            else:
                log.info(f'Ignoring unrecognized arg {arg}')
    for directory in dirs:
        for filename in os.listdir(directory):
            path = Path(os.path.join(directory, filename))
            if path.is_file():
                if filename.endswith('.txt'):
                    if not path in workset_files:
                        workset_files.append(path)
                elif filename.endswith('.info'):
                    if not path in info_files:
                        info_files.append(path)
    # print('Worksets', workset_files)
    # print('Info', info_files)
    ht = {}
    n_of_worksets_checked = 0
    workset_ids = set()
    for workset_file in sorted(workset_files):
        basename = os.path.basename(workset_file)
        ext_workset_id = basename.removesuffix('.txt')
        with open(workset_file) as f:
            line_number = 0
            sentence_ids = set()
            n_sentences = 0
            snt_id_core = None
            snt_id_sub_index = None
            for line in f:
                line_number += 1
                if not line.endswith('\n'):
                    log.error(f"{workset_file} line {line_number} does not end with newline character ('{line}')")
                if line_number == 1:
                    file_type = slot_value_in_double_colon_del_list(line, 'type')
                    if file_type:
                        if file_type != 'workset':
                            log.error(f"{workset_file} line {line_number} has bad type '{file_type}'")
                    else:
                        log.error(f'{workset_file} line {line_number} lacks ::type')
                    workset_id = slot_value_in_double_colon_del_list(line, 'id')
                    if workset_id:
                        workset_ids.add(workset_id)
                        if workset_id != ext_workset_id:
                            log.error(f'{workset_file} line {line_number} ::id {workset_id} '
                                      f'does not match filename {basename}')
                        if not re.match(r'dip[-_a-zA-Z0-9]+[a-zA-Z0-9]', workset_id, re.IGNORECASE):
                            log.error(f'{workset_file} line {line_number} has invalid ::id {workset_id}')
                    else:
                        log.error(f'{workset_file} line {line_number} lacks ::id')
                    username = slot_value_in_double_colon_del_list(line, 'username')
                    if username:
                        if not re.match(r'[a-z]{3,}$', username):
                            log.error(f"{workset_file} line {line_number} has invalid ::username '{username}'")
                    else:
                        log.error(f'{workset_file} line {line_number} lacks ::username')
                    date = slot_value_in_double_colon_del_list(line, 'date')
                    if date:
                        if not re.match(r'(?:(Mon|Tue|Wed|Thu|Fri|Sat|Sun) )?'
                                        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, 20\d\d$', date):
                            log.error(f"{workset_file} line {line_number} has invalid ::date '{date}'")
                    else:
                        log.error(f'{workset_file} line {line_number} lacks ::date')
                    if not line.startswith('# '):
                        log.error(f"{workset_file} line {line_number} should start with '# '")
                elif line_number == 2:
                    description = slot_value_in_double_colon_del_list(line, 'description')
                    if description:
                        if len(description) < 20:
                            log.warning(f"{workset_file} line {line_number} has short ::description '{description}'")
                    else:
                        log.error(f'{workset_file} line {line_number} lacks ::description')
                    if not line.startswith('# '):
                        log.error(f"{workset_file} line {line_number} should start with '# '")
                else:
                    if m4 := re.match(r'(([a-z][-_a-z0-9]*[a-z0-9]_\d\d\d\d)\.(\d+))\s*(.*?)\s*$', line, re.IGNORECASE):
                        n_sentences = +1
                        snt_id = m4.group(1)
                        if snt_id in sentence_ids:
                            log.error(f'{workset_file} line {line_number} has duplicate sentence ID {snt_id}')
                        else:
                            sentence_ids.add(snt_id)
                        if m4.group(3) == '0':
                            log.error(f'{workset_file} line {line_number} sentence ID sub-index is 0 ({snt_id})')
                        elif m4.group(3).startswith('0'):
                            log.error(f'{workset_file} line {line_number} sentence ID sub-index '
                                      f'starts with 0 ({snt_id})')
                        if snt_id_core and not snt_id_core == m4.group(2):
                            log.warning(f'{workset_file} line {line_number} change of code sentence ID '
                                        f'from {snt_id_core} to {m4.group(2)}')
                        elif snt_id_sub_index and (int(snt_id_sub_index) +1 != int(m4.group(3))):
                            log.warning(f'{workset_file} line {line_number} non-sequitive sentence ID sub-index '
                                        f'{snt_id_sub_index} to {m4.group(3)}')
                        snt_id_core = m4.group(2)
                        snt_id_sub_index = m4.group(3)
                        if m4.group(4) == '':
                            log.error(f'{workset_file} line {line_number} sentence ID {snt_id} has empty sentence')
                    else:
                        if m := re.match(r'\S+', line):
                            first_token = m.group(0)
                            log.error(f"{workset_file} line {line_number} does not start "
                                      f"with a valid sentence ID: {first_token}")
                        else:
                            log.error(f"{workset_file} line {line_number} does not start with a valid sentence ID; "
                                      f"rather, it starts with a space")
            if n_sentences == 0:
                log.warning(f"{workset_file} contains no sentences")
            n_of_worksets_checked += 1
            ht[('workset snt-IDs', ext_workset_id)] = sentence_ids

    n_of_info_files_checked = 0
    info_ids = set()
    for info_file in sorted(info_files):
        basename = os.path.basename(info_file)
        ext_workset_id = basename.removesuffix('.info')
        with open(info_file) as f:
            line_number = 0
            sentence_ids = set()
            n_sentences = 0
            info_ids.add(ext_workset_id)
            for line in f:
                line_number += 1
                if not line.endswith('\n'):
                    log.error(f"{info_file} line {line_number} does not end with newline character ('{line}')")
                if m4 := re.match(r'(([a-z][-_a-z0-9]*[a-z0-9]_\d\d\d\d)\.(\d+))\s*(.*?)\s*$', line, re.IGNORECASE):
                    n_sentences += 1
                    snt_id = m4.group(1)
                    if snt_id in sentence_ids:
                        log.error(f'{info_file} line {line_number} has duplicate sentence ID {snt_id}')
                    else:
                        sentence_ids.add(snt_id)
                    content = m4.group(4)
                    if not re.search(r'\bsender: (Austria|England|France|Germany|Italy|Russia|Turkey)\b',
                                     content, re.IGNORECASE):
                        log.error(f'{info_file} line {line_number} lacks valid sender')
                    if not re.search(r'\brecipient: (Austria|England|France|Germany|Italy|Russia|Turkey)\b',
                                     content, re.IGNORECASE):
                        log.error(f'{info_file} line {line_number} lacks valid recipient')
                    if not re.search(r'\btime: (Spring|Summer|Fall|Winter) 19\d\d\b',
                                     content, re.IGNORECASE):
                        log.error(f'{info_file} line {line_number} lacks valid time')

            n_of_info_files_checked += 1
            ht[('info snt-IDs', ext_workset_id)] = sentence_ids
    if workset_ids != info_ids:
        missing_info_ids = workset_ids - info_ids
        if missing_info_ids:
            log.warning(f'Missing info files for worksets: {missing_info_ids}')
        missing_workset_ids = info_ids - workset_ids
        if missing_workset_ids:
            log.warning(f'Missing worksets for: {missing_workset_ids}')
    joint_workset_ids = workset_ids.intersection(info_ids)
    for workset_id in joint_workset_ids:
        workset_snt_ids = ht.get(('workset snt-IDs', workset_id))
        info_snt_ids = ht.get(('info snt-IDs', workset_id))
        if workset_snt_ids != info_snt_ids:
            missing_snt_ids_in_info = workset_snt_ids - info_snt_ids
            if missing_snt_ids_in_info:
                log.error(f'{workset_id}.info does not cover snt IDs (compared to {workset_id}.txt): {sorted(list(missing_snt_ids_in_info))}')
            missing_snt_ids_in_workset = info_snt_ids - workset_snt_ids
            if missing_snt_ids_in_workset:
                log.error(f'{workset_id}.info contains spurious snt IDs (compared to {workset_id}.txt): {sorted(list(missing_snt_ids_in_workset))}')
    if n_of_worksets_checked or not help_printed:
        log.info(f"Number of worksets checked: {n_of_worksets_checked}")
    if n_of_info_files_checked or not help_printed:
        log.info(f"Number of info files checked: {n_of_info_files_checked}")


if __name__ == "__main__":
    main()
