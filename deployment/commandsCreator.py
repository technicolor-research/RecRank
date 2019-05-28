"""
  Copyright (c) 2019 - Present – Thomson Licensing, SAS
  All rights reserved.
 
  This source code is licensed under the Clear BSD license found in the
  LICENSE file in the root directory of this source tree.
"""


surpriseAlgo=["SVD","PMF","NMF","SVDpp","KNNWithMeans"]

f = open('commands.txt', 'w')
gitPath="~/black-box_recom/git/"
dataPath="~/black-box_recom/datasets/ML100K/"
configPath="~/black-box_recom/config/"
outputPath="~/temp/"


# surprise algo
for i in range(0, len(surpriseAlgo)):

  # unknown_train + unknown_test
  cmd = "export name=" + surpriseAlgo[i] + \
      ",python "+gitPath+"recom.py --pickleSavePath "+outputPath+"${name}ut.pickle" + \
      " --dataset "+dataPath+"unknown_train.csv  --validSize 0" + \
      " --testSet "+dataPath+"unknown_test.csv" + \
      " --surprise_algo ${name} --proc 15" + \
      " --evalTrain --topN_list 5 10 20 30 50" + \
      " > "+outputPath+"${name}ut.out 2>&1"
  f.write("%s\n" % cmd)

  # known_train + unknown_test
  cmd = "export name=" + surpriseAlgo[i] + \
      ",python "+gitPath+"recom.py --pickleSavePath "+outputPath+"${name}kt.pickle" + \
      " --dataset "+dataPath+"known_train.csv  --validSize 0" + \
      " --testSet "+dataPath+"unknown_test.csv" + \
      " --surprise_algo ${name} --proc 15" + \
      " --evalTrain --topN_list 5 10 20 30 50" + \
      " > "+outputPath+"${name}kt.out 2>&1"
  f.write("%s\n" % cmd)

  # known_train + known_valid
  cmd = "export name=" + surpriseAlgo[i] + \
      ",python "+gitPath+"recom.py --pickleSavePath "+outputPath+"${name}kv.pickle" + \
      " --dataset "+dataPath+"known_train.csv  --validSize 0" + \
      " --testSet "+dataPath+"known_val.csv" + \
      " --surprise_algo ${name} --proc 15" + \
      " --evalTrain --topN_list 5 10 20 30 50" + \
      " > "+outputPath+"${name}kv.out 2>&1"
  f.write("%s\n" % cmd)

  # known_train + known_query
  cmd = "export name=" + surpriseAlgo[i] + \
      ",python "+gitPath+"recom.py --pickleSavePath "+outputPath+"${name}kq.pickle" + \
      " --dataset "+dataPath+"known_train.csv  --validSize 0" + \
      " --testSet "+dataPath+"known_query.csv" + \
      " --surprise_algo ${name} --proc 15" + \
      " --evalTrain --topN_list 5 10 20 30 50" + \
      " > "+outputPath+"${name}kq.out 2>&1"
  f.write("%s\n" % cmd)
