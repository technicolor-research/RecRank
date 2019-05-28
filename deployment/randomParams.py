"""
  Copyright (c) 2019 - Present – Thomson Licensing, SAS
  All rights reserved.
 
  This source code is licensed under the Clear BSD license found in the
  LICENSE file in the root directory of this source tree.
 """


"""Creates random tuples for hyperparameter tuning"""

import argparse
import sys
import csv
import itertools
from random import shuffle

def parseRow(row):
  res = []
  for elem in row:
    try:
      res.append(int(elem))
    except:
      res.append(float(elem))
  return res

def main(args):
  parser = argparse.ArgumentParser(description='Creates random tuples for \
      hyperparameter tuning', formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument("--gridFile", action='store', required=True, help= \
      '.csv file with one row per hyperparameter')
  parser.add_argument("--output", action='store', required=True, help='output filename')

  args=parser.parse_args(args)

  print("Hyperparameters values:")
  params = []
  with open(args.gridFile, "rb") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
      params.append(parseRow(row))
      print(params[-1])

  combinations = list(itertools.product(*params))
  shuffle(combinations)
  with open(args.output, 'wb') as f:
    writer = csv.writer(f, delimiter=',')
    for combo in combinations:
      writer.writerow(combo)

if __name__ == "__main__":
  main(sys.argv[1:])
