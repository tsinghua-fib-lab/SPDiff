# -*- coding: utf-8 -*-
"""
***

"""

import yaml
from functools import reduce


def generate_combination(l1: list, l2: list):
    res = []
    for u in l1:
        for v in l2:
            if type(u) is not list:
                u = [u]
            if type(v) is not list:
                v = [v]
            res.append(u+v)
    return res


def generate_grid_search_params(search_params: dict):
    if len(search_params.keys()) == 1:
        return [[u] for u in list(search_params.values())[0]]
    else:
        return reduce(generate_combination, search_params.values())


def yaml_to_grid_params(input_path, script_name):
    with open(input_path, 'r') as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)

    for k, v in data.items():
        if type(v) is list:
            data[k] = ['--' + str(k) + ' ' + str(u) for u in v]
        else:
            data[k] = '--' + str(k) + ' ' + str(v)

    candidates = {u: v for u, v in data.items() if type(v) is list}
    non_candidates = [u for u, v in data.items() if type(v) is not list]
    grid_search_params = generate_grid_search_params(candidates)

    cmds = []
    for params_list in grid_search_params:
        cmd = ''
        for u in non_candidates:
            cmd += data[u] + ' '
        for u in params_list:
            cmd += u + ' '
        cmd = 'python3 ' + script_name + ' ' + cmd.strip()
        cmds.append(cmd + '\n')

    return cmds
