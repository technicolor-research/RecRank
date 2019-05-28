"""
  Copyright (c) 2019 - Present – Thomson Licensing, SAS
  All rights reserved.
 
  This source code is licensed under the Clear BSD license found in the
  LICENSE file in the root directory of this source tree.
 """


"""Calculates and visualizes the distance of recom algorithms""" 

from future.standard_library import install_aliases
install_aliases()

import argparse
import numpy as np
import os, sys
from collections import OrderedDict
# %matplotlib nbagg # uncomment for jupyter notebook
import pandas as pd
import matplotlib
from scipy import stats
import scipy.spatial
from sklearn import manifold, datasets, preprocessing
from orderedset import OrderedSet
import subprocess
import pickle
import math
from operator import itemgetter
import tempfile
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from mpldatacursor import datacursor, HighlightingDataCursor
import matplotlib.pyplot as plt

from networkx import *

def getFeatures(graphPath, verbose=False):
  """Derives manual features from graph 
  Args:
    graphPath (str): path to .pickle file
  Returns:
    list: features
    list: names
  """

  g = read_gpickle(graphPath)
  
  x = [] # input features
  names = [] # feature names
  
  # number of vertices
  x.append(g.number_of_nodes())
  names.append("vertices")
  if verbose: print("Vertices: ", g.number_of_nodes())

  # number of edges
  x.append(g.number_of_edges())
  names.append("edges")
  if verbose: print("Edges: ", g.number_of_edges())

  # vertex in-degree
  # TODO define bins edges i.e., x-axis positions => len(counts) = len(bins) - 1
  edges = range(0, 13)
  edges.append(g.number_of_edges())
  counts, bins, patches = plt.hist(list(dict(degree(g)).values()), bins=edges)
  if verbose:
    print("Counts: %s" % counts)
    print("Bins: %s" % bins)
  for i in range(0, len(counts)):
    x.append(counts[i])
    names.append("in-degree bin_" + str(i))

  avg = np.average(list(dict(degree(g)).values()))
  std = np.std(list(dict(degree(g)).values()))
  if verbose: print("In-degree: ", avg, std)
  x.append(avg)
  names.append("in-degree avg")
  x.append(std)
  names.append("in-degree std")
  
  # page rank
  i = 0
  ls = []
  nanCount = 0
  pr = pagerank(g)
  for v in pr:
    if (math.isnan(pr[v])):
      nanCount += 1
    else:
      ls.append(pr[v])

  if verbose: 
    print("Page rank: %s" % ls)
    if nanCount > 0:
      print("%d nan value(s) detected. Ignoring as a feature..." % nanCount)
  ls = sorted(ls, reverse=True)
  ls = np.array(ls)
  avg = np.average(ls)
  std = np.std(ls)
  x.append(avg)
  names.append("pr avg")
  x.append(std)
  names.append("pr std")

  # TODO define number of top ranked elements
  rank = 0
  for val in ls[:1]:
    x.append(val)
    names.append("page rank value %s" % rank)
    rank += 1
  # number of highly (>2*avg) ranked vertices
#  if verbose: print("Page rank Outlier: ", ls[ls > 2 * avg])
#  if verbose: print("Number of page rank outliers: ", len(ls[ls > 2 * avg]))
#  x.append(len(ls[ls > 2 * avg]))
#  names.append("page rank")

  # eignevector centrality
  temp = Graph(g)
  i = 0
  nanCount = 0
  ls = []
  ec = eigenvector_centrality(temp, weight="weight", \
      max_iter=10000)
  for v in ec:
    if (math.isnan(ec[v])):
      nanCount += 1
    else:
      ls.append(ec[v])
#    print(g.vertex_properties["text"][i], v)
    i += 1
 
  if verbose: 
    print("Eigenvector centrality: %s" % ls)
    if nanCount > 0:
      print("%d nan value(s) detected. Ignoring as a feature..." % nanCount)
  ls = sorted(ls, reverse=True)
  ls = np.array(ls)
  avg = np.average(ls)
  std = np.std(ls)
  x.append(avg)
  names.append("eigenvector avg")
  x.append(std)
  names.append("eigenvector std")

  # TODO define number of top ranked elements
  rank = 0
  for val in ls[:1]:
    x.append(val)
    names.append("eigenvector %s" % rank)
    rank += 1
  
  # number of vertices with high (>2*avg) centrality
#  if verbose: print("Eigenvector Outlier: ", ls[ls > 2 * avg])
#  if verbose: print("Number of eigenvector outliers: ", len(ls[ls > 2 * avg]))
#  x.append(len(ls[ls > 2 * avg]))
#  names.append("eigenvector")

  # betweenness centrality 
  temp = Graph(g)
  i = 0
  nanCount = 0
  ls = []
  bc = betweenness_centrality(temp, weight="weight")
  for v in bc:
    if (math.isnan(bc[v])):
      nanCount += 1
    else:
      ls.append(bc[v])
#    print(g.vertex_properties["text"][i], v)
    i += 1
 
  if verbose: 
    print("Betweeness centrality: %s" % ls)
    if nanCount > 0:
      print("%d nan value(s) detected. Ignoring as a feature..." % nanCount)

  ls = sorted(ls, reverse=True)
  ls = np.array(ls)
  avg = np.average(ls)
  std = np.std(ls)
  x.append(avg)
  names.append("betweeness avg")
  x.append(std)
  names.append("betweeness std")

  # TODO define number of top ranked elements
  rank = 0
  for val in ls[:1]:
    x.append(val)
    names.append("betweeness %s" % rank)
    rank += 1
  
  # number of vertices with high (>2*avg) centrality
#  if verbose: print("Betweenness Outlier: ", ls[ls > 2 * avg])
#  if verbose: print("Number of Betweenness outliers: ", len(ls[ls > 2 * avg]))
#  x.append(len(ls[ls > 2 * avg]))
#  names.append("betweeness")

  # closeness centrality 
  temp = Graph(g)
  i = 0
  nanCount = 0
  ls = []
  cc = closeness_centrality(temp, distance="weight")
  for v in cc:
    if (math.isnan(cc[v])):
      nanCount += 1
    else:
      ls.append(cc[v])
#    print(g.vertex_properties["text"][i], v)
    i += 1
   
  if verbose: 
    print("Closeness centrality: %s" % ls)
    if nanCount > 0:
      print("%d nan value(s) detected. Ignoring as a feature..." % nanCount)
  ls = sorted(ls, reverse=True)
  ls = np.array(ls)
  avg = np.average(ls)
  std = np.std(ls)
  x.append(avg)
  names.append("closeness avg")
  x.append(std)
  names.append("closeness std")

  # TODO define number of top ranked elements
  rank = 0
  for val in ls[:1]:
    x.append(val)
    names.append("closeness %s" % rank)
    rank += 1
 
  # katz centrality (issue with true_divide)
#  temp = Graph(g)
#  i = 0
#  ls = []
#  for v in katz(temp, weight=g.edges()["weight"]):
#    ls.append(v)
##    print(g.vertex_properties["text"][i], v)
#    i += 1
#   
#  if verbose: print("Katz centrality: %s" % ls)
#  ls = sorted(ls, reverse=True)
#  ls = np.array(ls)
#  avg = np.average(ls)
#  std = np.std(ls)
#  x.append(avg)
#  names.append("katz avg")
#  x.append(std)
#  names.append("katz std")
#
#  # TODO define number of top ranked elements
#  rank = 0
#  for val in ls[:1]:
#    x.append(val)
#    names.append("katz %s" % rank)
#    rank += 1
  

  # assortativity
  temp = Graph(g)
  assort = degree_assortativity_coefficient(g)
  # variance = degree_assortativity_coefficient(g)
  if verbose: print("(assortativity, variance): ", assort) #, variance)
  x.append(assort)
  names.append("assortativity")
  #x.append(variance)
  #names.append("assortativity var")

  # shortest distance (average of average shortest distances)
  temp = Graph(g)
  avgDist = [] # average of shortest_distances for a vertice
  sp = shortest_path_length(temp, weight="weight")
  for v in dict(sp).values():
    s = 0 # summation of shortest_distances
    count = 0 # number of reachable nodes
    for dist in v.values():
      if dist < 1e+308: # if node is reachable
        s += dist
        count += 1
    avgDist.append(s / float(count))
  if verbose: print("Average of average shortest distance: ", np.average(avgDist))
  x.append(np.average(avgDist))
  names.append("shortest distance")

  return x, names

def visualizer(pos_list, colors_list, labels_list, marker_list, legend_list):
  """Visualizes in 2D plot

  SN : nth set
  SN1 : first point of nth set
  Args:
    pos_list (list): 2D array with (x,y) coordinates for each point
        [array([x_s11, y_s11], [x_s12, y_s12], ...])
         array([x_s21, y_s21], [x_s22, y_s22], ...])
         ...]
    colors_list (list): [[c_s11, c_s12, ...]
                         [c_s21, c_s22, ...] ... ]
                         e.g., ['red', 'blue', ...]
    labels_list (list): same structure as colors_list
    marker_list (list): [m_s1, m_s2, ...]
                        e.g., ['o', '*', 'x', ...]
    legend_list (list): same structure as marker_list
  """
  # Next line to silence pyflakes. This import is needed.
  Axes3D
  
  #n_points = 1000
  #X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
  
  fig = plt.figure() #(figsize=(30, 30))
  S = len(pos_list) # number of sets
 
  # plot values
  for i in range(0, S): # for each set
    for j in range(0, len(pos_list[i])): # for each example
      x, y = pos_list[i][j]
      plt.scatter(x, y, marker=marker_list[i], color=colors_list[i][j], \
        label=labels_list[i][j], cmap=plt.cm.Spectral)
  plt.axis('tight')
  plt.title('Each algorithm has a different color\nClick on the points to annotate their label')

  # make symmetric axes limits
  flatten_pos = np.concatenate(pos_list).ravel()
  plt.xlim([max(flatten_pos), min(flatten_pos)])
  plt.ylim([max(flatten_pos), min(flatten_pos)])
 
  angleStep = 2*np.pi / float(len(flatten_pos))
  angle = 0 # polar coordinates angle [0, 2pi]
  R = 30 # outter
  for i in range(0, S): # for each set
    for j in range(0, len(pos_list[i])): # for each example
      x, y = pos_list[i][j]
      label = labels_list[i][j]
      plt.annotate(
        label,
        #xy=(x, y), xytext=(R * np.cos(angle), R * np.sin(angle)),
       xy=(x, y), xytext=(-20, 20),
       textcoords='offset points', ha='right', va='bottom',
     #   bbox=dict(boxstyle='round,pad=0.1', fc='blue', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
      angle += angleStep
    
  # Use a DataCursor to interactively display the label for a selected line...
  dc = datacursor(formatter='{label}'.format, bbox=dict(fc='white', alpha=1))

  # custom legend
  proxies = [create_proxy(marker) for marker in OrderedSet(marker_list)]
  plt.legend(proxies, legend_list, numpoints=1, markerscale=2)

  plt.show()

def values2dist(values, relative=False):
  """Derive distance matrix
  Args:
    values (list): e.g., [[x11,x12, ... x1n], 
                          [x21,x22, ... x2n], ...]
    relative (boolean): divide the L2 distance by the norm of the first value
  Returns:
    np.array: NxN matrix with the L2 distances
  """
  N = len(values)
  X = np.zeros((N, N))
  for i in range(0, N):
    for j in range(0, N):
      X[i,j] = scipy.spatial.distance.euclidean(values[i], values[j])
      if relative:
        X[i,j] /= np.linalg.norm(values[i]) 

  return X

def create_proxy(marker):
#    line = matplotlib.lines.Line2D([0], [0], linestyle='none', mfc='black',
#                mec='none', marker=r'$\mathregular{{{}}}$'.format(label))
    scatter = plt.scatter([], [], color='black', marker=marker)
    return scatter

def pairwiseSetDist(recomA, recomB, thres):
  """Calculates the set intersection between two output lists
  ! pairwiseSetDist(x,y) != pairwiseSetDist(y,x) 
  Args:
    recomA, recomB (lists): see recom.py#surprise_recom()
    thres (int): see setDist()
  Returns:
    float: set distance
  """
  # derive the per-user and per-item recommendation sets
  userRecA = {}
  itemRecA = {}
  for uid, iid, ts, topN in recomA:
    if uid not in userRecA:
      userRecA[uid] = set([])
    if iid not in itemRecA:
      itemRecA[iid] = set([])
    userRecA[uid].update(map(lambda x: x[0], topN[:thres]))
    itemRecA[iid].update(map(lambda x: x[0], topN[:thres]))

  userRecB = {}
  itemRecB = {}
  for uid, iid, ts, topN in recomB:
    if uid not in userRecB:
      userRecB[uid] = set([])
    if iid not in itemRecB:
      itemRecB[iid] = set([])
    userRecB[uid].update(map(lambda x: x[0], topN[:thres]))
    itemRecB[iid].update(map(lambda x: x[0], topN[:thres]))

  # calculate set intersection
  # user set intersection
  ucommon = 0
  utotal = 0
  for user in userRecA.keys():
    if user in userRecB:
      ucommon += len(userRecA[user].intersection(userRecB[user]))
    utotal += len(userRecA[user])
  # item set intersection
  icommon = 0
  itotal = 0
  for item in itemRecA.keys():
    if item in itemRecB:
      icommon += len(itemRecA[item].intersection(itemRecB[item]))
    itotal += len(itemRecA[item])
  
  sim = (ucommon/float(utotal) + icommon/float(itotal)) / 2.0
  dist = 1 - sim # distance

  return dist

def setDist(folder_path, thres=5, extra=None, cmpRank_path=None):
  """Extracts features for all the recom outputs in the given folder path
  by using the average(per-user set intersection, per-item set intersection)
  Args:
    folder_path (str): folder path with N .pickle output files
      see recom.py#surprise_recom() for the file structure
    thres (int): threshold for topN recommendations
    extra (str): extra .pickle file (usefull for black-box)
    cmpRank_path (str): see main#args.cmpRank_path
  Returns: (corresponding to each graph: )
    np.array: Nx2 coordinates (for plotting)
    list: N labels
    list: N colors
    np.array: NxN distance matrix
  """
  recs = []
  labels = []
  colors = []
  
  colorValues = ['red', 'blue', 'green', 'black'] # one per class
  filenames = map(lambda f: os.path.join(folder_path, f), os.listdir(folder_path))
  if not extra is None:
    filenames.append(extra)

  for filename in set(filenames):
    if filename.endswith(".pickle"):
      # prune attributes
      if filename.find('cold') != -1 \
          or filename.find('drop') != -1: 
        continue
      print("Getting output from: %s" % filename)
      with open(filename, 'rb') as handle:
        recom = pickle.load(handle)
      recs.append(recom)

      # color the classes
      if filename.find('KNN') != -1:
        colors.append(colorValues[0])
      elif filename.find('MF') != -1 or filename.find('SVD') != -1:
        colors.append(colorValues[1])
      elif filename.find('RBM') != -1:
        colors.append(colorValues[2])
      else:
        colors.append(colorValues[-1])

      labels.append(filename.rsplit('/', 1)[-1])

  # calculate set intersection
  X = np.zeros([len(recs), len(recs)]) # distance matrix
  for i in range(0, len(X)):
    for j in range(i, len(X)):
      # user set intersection
      dist = pairwiseSetDist(recs[i], recs[j], thres)

      X[i,j] = dist
      X[j,i] = dist # FIXME symmetric isn't equal (see pairwiseSetDist())

  print(X)
  tempX = X
  # print sorted distances
  if extra:
    d = []
    i = labels.index(extra.rsplit('/',1)[-1]) # black-box index
    for j in range(0, len(tempX[i])):
      if i != j:
        d.append((labels[j], tempX[i,j]))
    d = sorted(d, key=itemgetter(1))
    print("Set distance from: %s" % labels[i])
    rank = []
    for algo, dist in d:
      alg = algo.rsplit('.',1)[0] # remove suffix
      print("%s\t%.3f" % (alg, dist))
      rank.append(alg)

    # compute spearman correlation
    if cmpRank_path: 
      cmpRank = []
      with open(cmpRank_path, 'r') as f:
        for line in f.readlines():
          cmpRank.append(line[:-1]) # append name without \n

      rankCompare(rank, cmpRank)



  # FIXME filter inf values from X
  model = manifold.MDS(n_components=2, dissimilarity="precomputed", \
      random_state=42) # features
  pos = model.fit_transform(X)
  print("Position matrix:\n%s" % pos)

  return pos, labels, colors, X

def rankCompare(a, b):
  """Compare the ranking between two lists of string elements"""
  i = 0
  mapping = {}
  for ls in [a, b]:
    for s in ls:
      if s not in mapping:
        mapping[s] = i
        i += 1
  
  l1 = map(lambda x: mapping[x], a)
  l2 = map(lambda x: mapping[x], b)
  print("Ranking correlation between:\n%s\n%s" % (a, b))
  print(stats.spearmanr(l1, l2))
  print(stats.kendalltau(l1, l2))


def manualDist(folder_path, extra=None, cmpRank_path=None, \
    setDistFeature=False, thres=None, verbose=False):
  """Computes distances by using manual graph features
  Args:
    folder_path (str): folder path with N .graph graph files
    cmpRank_path (str): see main#args.cmpRank_path
    extra (str): extra .graph graph file (usefull for black-box)
    setDistFeature (boolean): append the set distance as a feature
      must give: extra, thres
      folder_path must additionally contain .pickle output files
    thres (int): set setDist()
    verbose (boolean): print feature extraction info
  Returns: (corresponding to each graph: )
    np.array: Nx2 coordinates (for plotting)
    list: N labels
    list: N colors
    np.array: NxM feature matrix
  """
  X = []
  labels = []
  colors = []
  
  colorValues = ['red', 'blue', 'green', 'yellow', 'magenta', 'black'] # one per class
  filenames = map(lambda f: os.path.join(folder_path, f), os.listdir(folder_path))
  if not extra is None and extra not in filenames:
    filenames.append(extra)

  if setDistFeature:
    # get black-box output
    f = extra.rsplit('.', 1)[-2] + '.pickle'
    with open(f, 'rb') as handle:
      bb_recom = pickle.load(handle)

  for filename in set(filenames):
    if filename.endswith(".graph"):
      # prune attributes
      if filename.find('cold') != -1 \
          or filename.find('drop') != -1: 
        continue
      print("Extracting features from: %s" % filename)
      features, feature_names = getFeatures(filename, verbose)
      
      if setDistFeature:
        f = filename[:-3] + 'pickle'
        print("Getting output from: %s" % f)
        with open(f, 'rb') as handle:
          recom = pickle.load(handle)
        features.append(pairwiseSetDist(bb_recom, recom, thres))
        feature_names.append('setDistance from black-box')

      # append if no nan features
      if True in map(lambda x: math.isnan(x), features):
        print("Nan feature(s) detected. Ignoring...")
        continue
      X.append(features)

      # color the classes
      if filename.find('KNN') != -1:
        colors.append(colorValues[0])
      elif filename.find('MF') != -1 or filename.find('SVD') != -1:
        colors.append(colorValues[1])
      elif filename.find('RBM') != -1:
        colors.append(colorValues[2])
      else:
        colors.append(colorValues[-1])

      labels.append(filename.rsplit('/', 1)[-1])
  
  X = np.array(X)

  # print feature variance
  print(pd.Series(labels))
  df = pd.DataFrame({'Variation Coefficient': stats.variation(X, 0), 
    'Std': np.std(X, 0), 'Mean': np.mean(X, 0), 'Feature': feature_names})
  print("Feature Variance")
  print(df)

  # print feature variance among specified examples
  comp = [] # comparison list with file ids
  try:
    id1 = labels.index('149.graph')
    id2 = labels.index('bb.graph')
    comp.append(id1)
    comp.append(id2)
    print("Feature Variance among: ", comp)
    df = pd.DataFrame({'Variation Coefficient': stats.variation(X[comp,:], 0), 
      id1: X[id1,:], id2: X[id2,:], 'Feature': feature_names})
    print(df)
  except ValueError:
    print("ValueError for the requested comparison")

  # z-score normalization (aka standardization)
  X = preprocessing.scale(X)

  tempX = values2dist(X)
  print("Distance matrix:\n%s" % tempX)
  # print sorted distances
  if extra:
    d = []
    i = labels.index(extra.rsplit('/',1)[-1]) # black-box index
    for j in range(0, len(tempX[i])):
      if i != j:
        d.append((labels[j], tempX[i,j]))
    d = sorted(d, key=itemgetter(1))
    print("Manual distance from: %s" % labels[i])
#    print("[%s]" % ', '.join(map(lambda x: x[0][:-6], d))) # print in list format
    rank = []
    for algo, dist in d:
      alg = algo.rsplit('.',1)[0] # remove suffix
      print("%s\t%.3f" % (alg, dist))
      rank.append(alg)
    
    # compute spearman correlation
    if cmpRank_path: 
      cmpRank = []
      with open(cmpRank_path, 'r') as f:
        for line in f.readlines():
          cmpRank.append(line[:-1]) # append name without \n

      rankCompare(rank, cmpRank)

  # features
  model = manifold.MDS(n_components=2, max_iter=300, n_jobs=-1, random_state=42)
#  model = manifold.TSNE(n_components=2, random_state=42)
  
  pos = model.fit_transform(X)
  
  # enforce same position for graphs with the same features (not done by MDS)
  for i in range(0, len(X)):
    for j in range(0, len(X)):
      if i != j and np.logical_and.reduce(np.equal(X[i], X[j])):
        print("Same features for: %s %s" % (labels[i], labels[j]))
        pos[i] = pos[j]

  return pos, labels, colors, X

# TODO make arguments similar to manualDist
def kernelDist(folder_path, cmpRank_path=None, extra=None):
  """Computes distances by using manual graph Kernels
  Args:
    folder_path (str): folder path with N .graph graph files
    cmpRank_path (str): see main#args.cmpRank_path
    extra (str): extra .graph graph file (usefull for black-box)
  Returns: (corresponding to each graph: )
    np.array: Nx2 coordinates
    list: N labels
    list: N colors
  """
  sys.path.append(os.path.abspath(os.path.join(os.path.realpath(__file__), \
    os.pardir)) + '/graph_kernels')
  import parse_graphs, wl_functions
  
  colorValues = ['red', 'blue', 'green', 'yellow', 'magenta', 'black'] # one per class
  idx = 0
  indices = []
  colors = []
  labels = []
  filenames = map(lambda f: os.path.join(folder_path, f), os.listdir(folder_path))
  if not extra is None and extra not in filenames:
    filenames.append(extra)
  for filename in set(filenames):
    if filename.endswith(".graph") or filename.endswith(".graph"):
      indices.append(idx)
      idx += 1

      # color the classes
      if filename.find('KNN') != -1:
        colors.append(colorValues[0])
      elif filename.find('MF') != -1 or filename.find('SVD') != -1:
        colors.append(colorValues[1])
      elif filename.find('RBM') != -1:
        colors.append(colorValues[2])
      else:
        colors.append(colorValues[-1])

      labels.append(filename.rsplit('/', 1)[-1])
  

  filenames_graphs = tempfile.NamedTemporaryFile() # create temp for graph filenames
  print("Saving/Loading graph filenames to: " + filenames_graphs.name)
  with open(filenames_graphs.name, 'wb') as handle:
    wr = csv.writer(handle, delimiter='\n')
    wr.writerow(labels)

  filename_indices = tempfile.NamedTemporaryFile() # create temp for graph targets
  print("Saving/Loading graph targets to: " + filename_indices.name)
  with open(filename_indices.name, 'wb') as handle:
    wr = csv.writer(handle, delimiter=' ')
    wr.writerow(indices)

  node_label, ad_list, G, Y = parse_graphs.load_and_process(
    filenames_graphs.name, filename_indices.name, folder_path)
  
  # Apply WL graph kernel
  # Get a list of h kernel matrices: K
  # get a list of h features maps: phi 
  K, phi = wl_functions.WL_compute(ad_list, node_label, 100)
  X = K[-1]
  
  vfunc = np.vectorize(lambda x: np.true_divide(1, x))
  X = vfunc(X) # distance matrix
  np.fill_diagonal(X, 0) # set self node distances to 0

  tempX = X
  print("Distance matrix:\n%s" % tempX)
  # print sorted distances
  if extra:
    d = []
    i = labels.index(extra.rsplit('/',1)[-1]) # black-box index
    for j in range(0, len(tempX[i])):
      if i != j:
        d.append((labels[j], tempX[i,j]))
    d = sorted(d, key=itemgetter(1))
    print("Manual distance from: %s" % labels[i])
#    print("[%s]" % ', '.join(map(lambda x: x[0][:-6], d))) # print in list format
    rank = []
    for algo, dist in d:
      alg = algo.rsplit('.',1)[0] # remove suffix
      print("%s\t%.3f" % (alg, dist))
      rank.append(alg)
    
    # compute spearman correlation
    if cmpRank_path: 
      cmpRank = []
      with open(cmpRank_path, 'r') as f:
        for line in f.readlines():
          cmpRank.append(line[:-1]) # append name without \n

    rankCompare(rank, cmpRank)

  model = manifold.MDS(n_components=2, dissimilarity="precomputed", \
      random_state=42) # features
  pos = model.fit_transform(X)
  
  return pos, labels, colors

# TODO make arguments similar to manualDist
def idealDist(folder_path):
  """Computes distances by using F1 scores
  Args:
    folder_path (str): folder path with N .out log files
  Returns: (corresponding to each graph: )
    np.array: Nx2 coordinates
    list: N labels
    list: N colors
  """
  labels = []
  colors = []
  values = []

  colorValues = ['red', 'blue', 'green', 'black'] # one per class
  for filename in os.listdir(folder_path):
    if filename.endswith(".out"):
      # prune attributes
      if filename.find('cold') != -1 \
          or filename.find('drop') != -1: 
        continue
      # keep only the classes
      if filename.find('KNN') != -1:
        colors.append(colorValues[0])
      elif filename.find('NMF') != -1:
        colors.append(colorValues[1])
      elif filename.find('PMF') != -1:
        colors.append(colorValues[2])
      elif filename.find('SVD') != -1:
        colors.append(colorValues[3])
      else:
        continue
      # get F1 score
      logPath = os.path.join(folder_path, filename)
      cmd = ("awk -F: '/F1:/ {print $2}' " + logPath)
      out = subprocess.getoutput(cmd)
      try:
        print("F1 score of %s: %s" % (logPath, float(out)))
      except:
        print("Problem with getting F1 score of %s" % logPath)
        continue
      values.append(float(out))
      labels.append(filename[:-4])
    
  tempX = values2dist(values)
  print("Distance matrix:\n%s" % tempX)

  # print sorted distances
  for i in range(0, len(tempX)):
    d = []
    for j in range(0, len(tempX[i])):
      if i != j:
        d.append((labels[j], tempX[i,j]))
    d = sorted(d, key=itemgetter(1))
    print("Ideal distance from: %s" % labels[i])
    for algo, dist in d:
      print("%s\t%.3f" % (algo, dist))

  X = values2dist(values)
  print("Distance matrix:\n%s" % X)
  model = manifold.MDS(n_components=2, dissimilarity="precomputed", \
      random_state=42) # features
  pos = model.fit_transform(X)
  print("Position matrix:\n%s" % pos)

  return pos, labels, colors


def main(args): 
  parser = argparse.ArgumentParser(description= \
      'Calculates and visualizes the distance of recom algorithms', \
      formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument("--folder_paths", nargs='+', help='Paths to directories with' \
      + 'input files.')
  parser.add_argument("--blackbox_path", type=str, action='store', \
      help='Full path to bb input file. Can be inside folder_paths or outside.')
  parser.add_argument("--cmpRank_path", type=str, action='store', \
      help='Path path to \'true\' ranking file.\n' + \
      'Example: \n\tPMFlib\n\tSVDpp\n\tKNNlib\n\tSVD\n' + \
      'Must give --blackbox_path.')
  parser.add_argument("--kernel", dest='kernel', action='store_true', \
      help='If set => compute distances with graph_kernels.\n'
      + 'input files: .graph files.')
  parser.add_argument("--manual", dest='manual', action='store_true', \
      help='If set => compute distances from graph properties.\n' \
      + 'input files: .graph files.')
  parser.add_argument("--ideal", dest='ideal', action='store_true', \
      help='If set => compute distances from F1 scores.\n'
      + 'input files: .out log files.')
  parser.add_argument("--setDist", dest='setDist', action='store_true', \
      help='If set => compute distances from user-item set intersections.\n'
      + 'input files: .pickle output files. Must set --topN')
  parser.add_argument("--manualSet", dest='manualSet', action='store_true', \
      help='If set => compute distances from graph properties combined with set intersections.\n' \
      + 'input files: .graph + .pickle files.\n' \
      + '!must give --blackbox_path')
  parser.add_argument("--topN", type=int, default=5, action='store', \
      help='Threshold for pruning topN list; default=5.')
  parser.add_argument("--sim_path", type=str, action='store', \
      help='Path for .csv file containing the similarity matrix. e.g.:\n' \
      + '1,simAB\nsimAB,1')
  parser.add_argument("--labels", type=str, action='store', \
      help='Path for .csv file with labels corresponding ' \
      + 'to the given similarity matrix. e.g.:\nA,B')
  parser.add_argument("--visualize", dest='visualize', action='store_true', \
      help='Visualize computed distances in a 2D MDS plot.')
  parser.add_argument("--verbose", dest='verbose', action='store_true', \
      help='Print feature info.')
  parser.add_argument("--fileLoadPath", type=str, action='store', help= \
      'If set=> Compare with distances loaded from the specified path.')
  parser.add_argument("--folderSavePath", type=str, action='store', help= \
      'If set => Save the distances for each input file in a .pickle file.')

  args = parser.parse_args(args)
  if args.folder_paths:  
    markerValues = ['o', '*', 'x', '^'] # one per input folder
    #print("openMP enabled: %s" % openmp_enabled())
    pos_list = []
    colors_list = []
    marker_list = []
    legend_list = []
    labels_list = []
    markerIdx = 0
    for path in args.folder_paths:
      if args.manual or args.manualSet:
        pos, labels, colors, _ = manualDist(path, thres=args.topN,
            setDistFeature=args.manualSet, extra=args.blackbox_path, 
            cmpRank_path=args.cmpRank_path, verbose=args.verbose)
      elif args.kernel:
        pos, labels, colors = kernelDist(path,
            extra=args.blackbox_path, cmpRank_path=args.cmpRank_path)
      elif args.ideal:
        pos, labels, colors = idealDist(path)
      elif args.setDist:
        pos, labels, colors, _ = setDist(path, args.topN,
            extra=args.blackbox_path, cmpRank_path=args.cmpRank_path)
      else:
        print("Must give --manual or --kernel or --ideal")
        sys.exit(1)
      
      # scale values to [0,1]
      pos += abs(min(0, min(pos.ravel()))) # offset
      pos /= max(pos.ravel()) # scale
 
      pos_list.append(pos)
      labels_list.append(labels)
      colors_list.append(colors)
      marker_list.append(markerValues[markerIdx])
      legend_list.append(os.path.basename(os.path.normpath(path)))
      markerIdx += 1

      # save the distances
      if not args.folderSavePath is None:
        print("Saving to: %s.pickle" % (args.folderSavePath+legend_list[-1]))
        with open(args.folderSavePath+legend_list[-1]+'.pickle', 'wb') as handle:
          pickle.dump(pos, handle)
         
      # load and compare the distances
      if not args.fileLoadPath is None:
        print("Loading from: %s" % args.fileLoadPath)
        with open(args.fileLoadPath, 'rb') as handle:
          poscmp = pickle.load(handle)
        diff = values2dist(pos) - values2dist(poscmp)
        print("Diff Matrix:\n%s" % diff)
        print("MAE: %s" % np.abs(diff).sum())

    if args.visualize:
      visualizer(pos_list, colors_list, labels_list, marker_list, legend_list)

    
  elif args.sim_path and args.labels: # get similarity matrix path
    labels = np.genfromtxt(args.labels, dtype='str', delimiter=',')
    X = np.genfromtxt(args.sim_path, delimiter=',')
    print(labels)
    print("Similarity matrix:\n%s" % X)
    vfunc = np.vectorize(lambda x: 1-x)
    X = vfunc(X) # distance matrix
    colors = []
    for i in range(0, len(labels)):
      colors.append('red')
    
    model = manifold.MDS(n_components=2, dissimilarity="precomputed", \
        random_state=42) # features
    pos = model.fit_transform(X)
    if args.visualize:
      visualizer([pos], [colors], [labels], ['o'], ['Similarities'])

  else:
    print("Must give --folder_paths or (--sim_path and --labels)")
    sys.exit(1)

if __name__ == "__main__":
  main(sys.argv[1:])
