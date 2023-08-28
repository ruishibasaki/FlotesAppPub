import networkx as nx
import osmnx as ox
import preprocessors as pp
from sklearn import tree
import numpy as np
import math

from sklearn.model_selection import GridSearchCV, StratifiedKFold
import RegularDecisionTree as rdt

#=================================================================================

def discretize_flow(nodeFlowSimulations, simModelInfo):
    flow_array = [[] for i in range(len(simModelInfo))]
    for i in range(len(simModelInfo)):
        flow_array[i].append(nodeFlowSimulations[i])
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(flow_array, simModelInfo)
    return(clf, pp.getLeafNodes(clf), clf.apply(flow_array))

#=================================================================================
def buildClassifier(flow, simModelInfo, observed_nodes):
    flow_array = [[] for i in range(len(simModelInfo))]
    for i in range(len(simModelInfo)):
        for n in observed_nodes:
            flow_array[i].append(flow[n][i])

    leafsz = max(1,round(0.1*len(simModelInfo)))
    clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=leafsz, min_samples_split=2*leafsz)
    clf = clf.fit(flow_array, simModelInfo)
    return(clf)

#=================================================================================

def predictWithClassifier(G, clf, flow, restriction=[]):
    flow_array = [[]]
    if len(restriction)>0:
        nodes_list = restriction
    else:
        nodes_list = list(G.nodes())
    for n in nodes_list:            
        flow_array[0].append(flow[n])
    model = clf.predict_proba(flow_array)
    return(model)

#=================================================================================

def calibrate(c, flow_array, simModelInfo, hist_observed_nodes, parameters):
    reg_model = rdt.RegularizedDecisionTreeClassifier(max_n_features=c, min_samples_leaf=5)
    skf = StratifiedKFold(n_splits=5)
    
    grid_reg = GridSearchCV(reg_model, parameters, cv = skf, n_jobs= 6)
    grid_reg.fit(flow_array, simModelInfo, default_selected=hist_observed_nodes)

    print(grid_reg.best_params_)
    print(grid_reg.best_score_)
    return grid_reg.best_params_

#=================================================================================

def placeSensors(G, simulations, simNodesFlow, hist_observed_nodes, c):
    flow_array, simModelInfo = pp.prepareData(G,simulations, simNodesFlow)
    reg_model = rdt.RegularizedDecisionTreeClassifier(delta=0.8, max_n_features=c, min_samples_leaf=5)
    reg_model.fit(flow_array, simModelInfo, default_selected=hist_observed_nodes)

    print("depth: ", reg_model.depth)
    print(reg_model.selected_features_, 'hist: ', hist_observed_nodes)
    places = list(reg_model.selected_features_)
    hist_observed_nodes.extend(places)
    #print(nodes_list[places[0]], nodes_list[places[1]])

    nodes_list = list(G.nodes())
    places = [nodes_list[i] for i in places]
    

    return places, reg_model

#=================================================================================

def placeSensorsHF(G,simNodesFlow, simulations, hist_observed_nodes, c):
    avgsimNodesFlow = pp.avgSimFlow(simNodesFlow)
    tabu =[]
    cc=0
    while(cc<c):
        key= max(avgsimNodesFlow, key=avgsimNodesFlow.get)
        if key not in hist_observed_nodes and key not in tabu:
            hist_observed_nodes.append(key)
            node = ox.distance.nearest_nodes(G, G.nodes()[key]['x'], G.nodes()[key]['y'])
            subg = ox.truncate.truncate_graph_dist(G, node, max_dist=100, weight='length')
            tabu.append(list(subg.nodes()))
            cc = cc+1
        avgsimNodesFlow.pop(key)

    flow_array, simModelInfo = pp.prepareData(G,simulations, simNodesFlow, hist_observed_nodes)
    reg_model = rdt.RegularizedDecisionTreeClassifier(delta=1.0, max_n_features=len(hist_observed_nodes), min_samples_leaf=5)
    reg_model.fit(flow_array, simModelInfo)

    print("depth: ", reg_model.depth)
    print(reg_model.selected_features_, 'hist: ', hist_observed_nodes)
    return hist_observed_nodes, reg_model 

#=================================================================================

def placeSensorsHFR(G,simNodesFlow, simulations, hist_observed_nodes, c):
    avgsimNodesFlow = pp.avgSimFlow(simNodesFlow)
    tabu =[]
    cc=0
    while(cc<c):
        key= max(avgsimNodesFlow, key=avgsimNodesFlow.get)
        if key not in hist_observed_nodes:
            block =False
            for e in G.edges(key,data=True):
                if 'name' in e[2] and e[2]['name'] in tabu:
                    block =True
                    break

            if(not block):
                hist_observed_nodes.append(key)
                cc = cc+1
                for e in G.edges(key, data=True):
                    if 'name' in e[2]  and e[2]['name'] not in tabu:
                        tabu.append(e[2]['name'])            
        avgsimNodesFlow.pop(key)

    flow_array, simModelInfo = pp.prepareData(G,simulations, simNodesFlow, hist_observed_nodes)
    reg_model = rdt.RegularizedDecisionTreeClassifier(delta=1.0, max_n_features=len(hist_observed_nodes), min_samples_leaf=5)
    reg_model.fit(flow_array, simModelInfo)

    print("depth: ", reg_model.depth)
    print(reg_model.selected_features_, 'hist: ', hist_observed_nodes)
    return hist_observed_nodes, reg_model 
