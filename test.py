from collections import namedtuple

Car = namedtuple("Car", 'color make model mileage')

import itertools as it


it.permutations

friends = ['Jack', "jill", 'Joe']

ox = it.permutations(friends, r=1)
list(ox)


from functools import reduce

# -*- coding: utf-8 -*-
import yaml
import io

# Define data
data = {
    'a list': [
        1,
        42,
        3.141,
        1337,
        'help',
        u'€'
    ],
    'a string': 'bla',
    'another dict': {
        'foo': 'bar',
        'key': 'value',
        'the answer': 42
    }
}

# Write YAML file
with io.open('data.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

# Read YAML file
with open("data.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)

print(data == data_loaded)

