"""
  Copyright (c) 2019 - Present – Thomson Licensing, SAS
  All rights reserved.
 
  This source code is licensed under the Clear BSD license found in the
  LICENSE file in the root directory of this source tree.
 """


"""Transforms the output of the recommendation algorithm into a graph"""


from surprise import SVD, KNNBasic
import numpy as np
import random
import sys
import argparse
import os
import pickle
from networkx import *

def gexfFormat(profile, recom, filename):
  """Outputs the recommendation graph in gexf format
  
  Parameters:
    profile (list): user profile as [(iid1, r1, ts1), ...] sorted by timestamp
    recom (list): topN recom for every click 
      [[(iid11, pred11), ... (iid1N, pred1N)],
       [(iid21, pred21), ... (iid2N, pred2N)],
       ...]
  """
  # ! ids are raw -> strings
  clickedItems = set(map(lambda x: str(x[0]), profile)) # set of clicked items
  recomItems = set() # set of recommended items
  
  with open(filename, 'w') as f:
    # write header
    f.write("""<?xml version="1.0" encoding="UTF-8"?>\n""")
    f.write("""<gexf xmlns:viz="http:///www.gexf.net/1.2/viz" version="1.2" xmlns="http://www.gexf.net/1.2">\n""")
    f.write("""<graph defaultedgetype="undirected" idtype="string" type="static">\n""")
    
    # write edges
    f.write("""<edges>\n""")
    id = 0
    for click in range(0, len(profile)): # for all the clicks
      print("Number of processed clicks: ", click)
      for rec in recom[click]: # for the topN recommendations
        f.write("<edge id=\"" + str(id) + "\" source=\"" + str(rec[0]) + "\" target=\"" + str(profile[click][0]) + "\" weight=\"" + str(rec[1]) + "\"/>\n")
        recomItems.add(str(rec[0]))
        id += 1
    
    f.write("""</edges>\n""")
    
    f.write("""<nodes>\n""")
    # write clicked item-nodes in an outter ring
    angleStep = 2*np.pi / float(len(clickedItems)) # polar coordinates angle step
    angle = 0 # polar coordinates angle [0, 2pi]
    R = 1000 # outter
    for item in clickedItems: # for all the clicks
      target = str(item)
      f.write("<node id=\"" + target + "\">\n")
      f.write("\t")
      f.write("""<viz:color r="255" g="0" b="0"></viz:color>\n""") # red color
      f.write("<viz:position x=\"" + str(R * np.cos(angle)) + "\" y=\"" + str(R * np.sin(angle)) + "\" z=\"0.0\"/>") # ring position
      f.write("</node>\n")
      angle += angleStep
    
    # write the rest item-nodes in an inner ring
    angleStep = 2*np.pi / float(len(recomItems - clickedItems)) # polar coordinates angle step
    angle = 0 # polar coordinates angle [0, 2pi]
    R = 600 # outter
    for item in recomItems - clickedItems: # for the rest of the items
      target = str(item)
      f.write("<node id=\"" + target + "\">\n")
      f.write("\t")
      f.write("<viz:position x=\"" + str(R * np.cos(angle)) + "\" y=\"" + str(R * np.sin(angle)) + "\" z=\"0.0\"/>") # ring position
      f.write("</node>\n")
      angle += angleStep
      
    f.write("""</nodes>\n""")
    f.write("""</graph>\n</gexf>""")

def graph_2hop(profile, recom, filename):
  """Outputs the recommendation graph as a Graph() object
  Graph has two rings (2hop)
  Vertices (red): outer ring contains the clicked items
  Vertices (blue): inner ring contains the recommended \setminus clicked items
  Edge between item A, B <=> B was recommended to the user after click A
  edge weight = rating prediction

  Args:
    profile (list): user profile as [(iid1, r1, ts1), ...] sorted by timestamp
    recom (list): topN recom for every click 
      [[(iid11, pred11), ... (iid1N, pred1N)],
       [(iid21, pred21), ... (iid2N, pred2N)],
       ...]
    filename (string): path with extension specifing the output type (e.g. out.xml)
  
  Returns:
    Graph: recommendation graph
  """
  g = Graph()

  # ! ids are raw -> strings
  clickedItems = set(map(lambda x: str(x[0]), profile)) # set of clicked items
  recomItems = set() # set of recommended items

  # get recommended items
  for click in range(0, len(profile)): # for all the clicks
    for rec in recom[click]: # for the topN recommendations
      recomItems.add(str(rec[0]))

  # write clicked item-nodes in an outter ring
  angleStep = 2*np.pi / float(len(clickedItems)) # polar coordinates angle step
  angle = 0 # polar coordinates angle [0, 2pi]
  R = 1000 # outter
  for item in clickedItems: # for all the clicks
    target = str(item)
    g.add_node(target)
    g.nodes[target]['color'] = [255,0,0,1] # RGBA format
    g.nodes[target]['pos'] = [R * np.cos(angle), R * np.sin(angle)]
    g.nodes[target]['text'] = target

    angle += angleStep
    
  # write the rest item-nodes in an inner ring
  angleStep = 2*np.pi / float(len(recomItems - clickedItems)) # polar coordinates angle step
  angle = 0 # polar coordinates angle [0, 2pi]
  R = 600 # outter
  for item in recomItems - clickedItems: # for the rest of the items
    target = str(item)
    g.add_node(target)
    g.nodes[target]['color'] = [0,0,255,1] # RGBA format
    g.nodes[target]['pos'] = [R * np.cos(angle), R * np.sin(angle)]
    g.nodes[target]['text'] = target

    angle += angleStep
      
  # construct edges
  edges = {} # dictionary: (source_iid, target_iid) -> Vertex object
  weight_prop = g.new_edge_property('float')
  
  for click in range(0, len(profile)): # for all the clicks
    for rec in recom[click]: # for the topN recommendations
      target= str(rec[0])
      source = str(profile[click][0])
      weight = rec[1]

      g.add_edge(source, target)
      g.edges(source, target)['weight'] = weight
  
  return g

def itemGraphUpdate(clicked_iid, recom, graph=None):
  """Updates the recommendation graph as a Graph() object
  Vertices: recommended and/or clicked items
  Edge A, B <=> itemB is at least in one user recom list triggered by click A
  EdgeAB scoreSum = \sum_{users} prediction for B after clicked A
  EdgeAB scoreCount = # predictions for B after clicked A
  if not weight:
    The scores for each recommendation are ignored.
    Multiple recommendations -> multiple edges between the same nodes

  Args:
    g (Graph): previous recommendation graph.
      if None then a new graph is constructed.
    clicked_uid (str)
    clicked_iid (str)
    recom (list): topN recom that the clicked item triggered
       [(iid1, pred1), ... (iidN, predN)]
    weight (boolean)

  Returns:
    Graph: updated recommendation graph

  """
  if graph is None:
    g = Graph()
  else:
    g = graph

  # ! ids are raw -> strings
  clicked_iid = str(clicked_iid)

  # add vertices
  rec_ids = set(map(lambda x: str(x[0]), recom))
  for iid in rec_ids.union(set([clicked_iid])):
    # append if vertex does not exist
    if graph is None or iid not in g.nodes:
      g.add_node(iid)
      g.nodes[iid]['text'] = iid

  # add edges
  src = g.nodes[clicked_iid]['text']
  
  for iid, pred in recom:
    dst = g.nodes[iid]['text']
    g.add_edge(src, dst)  
#    print("Old scoreSum for edge (%s -> %s): %.2f" % (clicked_iid, iid, scoreSum_prop[e]))
    if 'scoreSum' not in g.edges[(src, dst)]:
      g.edges[(src, dst)]['scoreSum'] = pred
      g.edges[(src, dst)]['scoreCount'] = 1 
    else:
      g.edges[(src, dst)]['scoreSum'] += pred
      g.edges[(src, dst)]['scoreCount'] += 1
  
  return g

def userItemGraphUpdate(clicked_uid, clicked_iid, recom, graph=None):
  """Updates the recommendation graph as a Graph() object
  Vertices: recommended items and users
  Edge A, B <=> item A \in topn for user B
  EdgeAB scoreSum = score sum for item B from user A
  EdgeAB scoreCount = number of times item B was recommended to user A
  if not weight:
    The scores for each recommendation are ignored.
    Multiple recommendations -> multiple edges between the same nodes

  Args:
    g (Graph): previous recommendation graph.
      if None then a new graph is constructed.
    clicked_uid (str)
    clicked_iid (str)
    recom (list): topN recom that the clicked item triggered
       [(iid1, pred1), ... (iidN, predN)]
    weight (boolean)

  Returns:
    Graph: updated recommendation graph

  """
  if graph is None:
    g = Graph()
  else:
    g = graph



  # ! ids are raw -> strings
  clicked_iid = str(clicked_iid)
  clicked_uid = str(clicked_uid)

  # add item vertices
  rec_ids = set(map(lambda x: str(x[0]), recom))
  for iid in rec_ids:
    # append if vertex does not exist
    if graph is None or 'i_' + iid not in g.nodes:
      g.add_node('i_' + iid)
      g.nodes['i_' + iid]['text'] = 'i_' + iid

  # add user vertice
  if graph is None or 'u_' + clicked_uid not in g.nodes:
    g.add_node('u_' + clicked_uid)
    g.nodes['u_' + clicked_uid]['text'] = 'u_' + clicked_uid 

  # add edges
  src = g.nodes['u_' + clicked_uid]['text']
  for iid, pred in recom:
    dst = g.nodes['i_' + iid]['text']
    g.add_edge(src, dst)
#    print("Old scoreSum for edge (%s -> %s): %.2f" % (clicked_uid, iid, scoreSum_prop[e]))
#    print(g.num_edges())
    if 'scoreSum' not in g.edges[(src, dst)]:
      g.edges[(src, dst)]['scoreSum'] = pred
      g.edges[(src, dst)]['scoreCount'] = 1 
    else:
      g.edges[(src, dst)]['scoreSum'] += pred
      g.edges[(src, dst)]['scoreCount'] += 1
  
  return g


def normalizedWeight(g, weight=True):
#  """EdgeAB weight = scoreSum - (\sum_{scoreSum} / \sum_{scoreCount}) + 1
  """
  if weight:
    EdgeAB weight = scoreSum / (\sum_{scoreSum} / \sum_{scoreCount})
    i.e., show the deviation of each prediction from the mean to make the 
    weights from different graphs comparable
  else:
    EdgeAB weight = 1
  """
  
  s = 0
  c = 0 
  for e in g.edges():
	s += g.edges[e]['scoreSum']
	c += g.edges[e]['scoreCount']

  avgPred = s / float(c)
  print("Average score: ", avgPred)

  weights = []
  for e in g.edges():
    if weight:
      tmp = g.edges[e]['scoreSum'] / avgPred * 100
      weights.append(tmp)
      g.edges[e]['weight'] = tmp
    else:
      tmp = 1
      weights.append(tmp)
      g.edges[e]['weight'] = tmp

  print("Number of edges: %s" % len(weights))
  print("Weights avg: %.4f and std: %.4f" % (np.mean(weights), np.std(weights)))


  return g

def graph_toolFull(recom, filename, thres=5):
  """Outputs the full recommendation graph as a Graph() object

  Args:
    recom (list): topN recom for every click 
      [(uid1, iid1, ts1, [(iid11, pred11), ... (iid1N, pred1N)]),
       ...]
    filename (string): 
    thres (int): threshold for topN recommendations

  Returns:
    Graph: full recommendation graph
  """
  g = None
  for uid, iid, ts, recomList in recom:
    g = itemGraphUpdate(iid, recomList[:thres], g)
#    g = userItemGraphUpdate(uid, iid, recomList, g)

  g = normalizedWeight(g, True)
  write_gpickle(g, filename)

  return g

def main(args):

  parser = argparse.ArgumentParser(description='Recommender output TO graph')
  parser.add_argument("--output", action='store', default=True, \
      help='Output graph filename; extension defines the type (e.g., .xml)')
  parser.add_argument("--topN", type=int, default=5, action='store', \
      help= 'topN Threshold when creating recommendation graph; default=5')
  parser.add_argument("--pickleLoadPath", type=str, default=True, \
      action='store', help='Pickle file to load topN recoms list')

  args = parser.parse_args()
  
  random.seed(42) # reproducibility
  np.random.seed(42)

  with open(args.pickleLoadPath, 'rb') as handle:
    recs = pickle.load(handle)

  graph_toolFull(recs, args.output)

if __name__ == "__main__":
  main(sys.argv[1:])

