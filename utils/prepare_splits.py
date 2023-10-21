import os, random
from glob import glob

import math

all_files = glob("./../data/full_data/*.json")

split_size = .10

dev_files = random.sample(all_files, math.ceil(split_size*len(all_files)))

train_90_files = [f for f in all_files if f not in dev_files]

test_files = random.sample(train_90_files, math.ceil(split_size*len(all_files)))

train_files = [f for f in train_90_files if f not in test_files]

for file in train_files:
    cmd = "cp "+file+" ./../data/train/"
    _ = os.system(cmd)


for file in dev_files:
    cmd = "cp "+file+" ./../data/dev/"
    _ = os.system(cmd)


for file in test_files:
    cmd = "cp "+file+" ./../data/test/"
    _ = os.system(cmd)