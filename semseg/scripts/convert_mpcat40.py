'''
Convert mpcat40.tsv into usable python object
'''
import csv

import numpy as np


mpcat40 = []

with open('semseg/mpcat40.tsv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        entry = [int(row['mpcat40index']), row['mpcat40']]
        color_hex = row['hex']
        color_rgb = np.array([int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)])
        entry.append(color_rgb)
        mpcat40.append(entry)

print(mpcat40)
