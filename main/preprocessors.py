import networkx as nx

#=================================================================================

def readModels(file):
    count =0
    f_model_res = open(file)
    models = dict()
    reading_m = False
    nb_types=0
    class_c = ""
    count=0
    for l in f_model_res:
        sline = l.rstrip().split(" ")
        if sline[0] == "model":
            class_c = sline[1]
            nb_types = int(sline[2])
            models[class_c] = [nb_types, []]
            reading_m = True
            count=0
        elif reading_m :
            models[class_c][1].append((float(sline[1]), float(sline[2])))
            count+=1
        if count == nb_types:
            reading_m = False
    return models

#=================================================================================

def printModels(file, models):
    count =0
    f_model_res = open(file, 'w')
    for m,val in models.items():
        nbtypes = val[0]
        client_types = val[1]
        f_model_res.write('model '+ str(m)+' '+str(nbtypes)+'\n')
        for i in range(nbtypes):
            f_model_res.write('client_type '+ str(client_types[i][0])+' '+str(client_types[i][1])+'\n')
        f_model_res.write('\n')


#=================================================================================

def printFeedbackData(file, clients, observed_nodes, realFlow, tourlist):
    Vi = clients.copy()
    
    for n in observed_nodes:
        if n not in Va.keys():
            Vi[n] = realFlow[n]
        else:
            Vi[n] = max(realFlow[n],Vi[n])


    affct = dict()
    for n in Vi.keys():
        affct[n]=[]
    for t in range(len(tourlist)):
        setTour = set(tourlist[t])
        for n in Vi.keys():
            if n in setTour:
                affct[n].append(t)

    f_model_res = open(file, 'w')
    f_model_res.write(str(len(Vi))+' '+str(len(tourlist))+'\n')
    keys = Vi.keys()
    for n in keys:
        f_model_res.write(str(Vi[n])+'\n')
        for t in affct[n]:
            f_model_res.write(str(t)+' ')
        f_model_res.write('\n')
        
        
    

#=================================================================================
def readSimulations(file) :
    f_model_res = open(file)
    dictSim = {}
    simulations = []
    current = {}
    depot = 0
    reading_tours = False
    for l in f_model_res:
        sline = l.rstrip().split(" ")
        if sline[0]=="simulation": 
            current_num = int(sline[1])
            reading_tours = False
            if current_num not in dictSim.keys():
                simulations.append(dict())
                dictSim[current_num] = len(simulations)-1
            current = simulations[dictSim[current_num]]
        elif not reading_tours and sline[0]!="solutions":
            if sline[0]=="class" :
                current[sline[0]] = sline[1]
            elif sline[0]=="depot":
                depot = (int(sline[1]))
        elif not reading_tours and sline[0]=="solutions":
            reading_tours = True
            if "solutions" not in current.keys():
                current["solutions"] = {}
            current["solutions"].update({depot: []})
            current["solutions"][depot].append([])
        elif sline[0]=="solutions":
            reading_tours = True
            current["solutions"][depot].append([])
        else:
            current["solutions"][depot][-1].append([int(x) for x in sline])
    
    return simulations

#=================================================================================
def tourToNodeList(G,tour, pathDB) :
    ## List of OSM nodes id to list of nodes in the shortest-path between 
    ## consecutive nodes in the tour.
    nodes_tour = []
    ptour = list(tour)
    ptour.append(tour[0])
    for i in range(1,len(ptour)):
        n1, n2 = ptour[i-1], ptour[i]
        if (n1,n2) in pathDB:
            if i == 1:
                nodes_tour.extend(pathDB[(n1,n2)])
            else:
                nodes_tour.extend(pathDB[(n1,n2)][1:])
            continue

        sp = nx.shortest_path(G, n1, n2, weight="travel_time")
        g_sp = nx.path_graph(sp)
        pathDB[(n1,n2)] = list(g_sp.nodes)
        if i == 1:
            # add first node only for the first sortest path 
            # (avoid repeatition of nodes)
            nodes_tour.extend(pathDB[(n1,n2)])
        else:
            nodes_tour.extend(pathDB[(n1,n2)][1:])
    
    return(nodes_tour)

#=================================================================================
def countNodesFlow(G,depot_solutions, pathDB):
    ## Return flow of each node
    ## for the given 'tours' (list of list of node id)
    dict_flow = dict()
    edgesflow_partial = dict()
    edgesflow = dict()
    for solutions in depot_solutions.values():
        nsolutions = len(solutions)
        for sol in solutions:
            for tour in sol:
                nodes_tour = tourToNodeList(G,tour, pathDB)
                for i in range(1,len(nodes_tour)):
                    n1, n2 = nodes_tour[i-1], nodes_tour[i]
                    if (n1,n2) not in edgesflow_partial.keys():
                        edgesflow[(n1,n2)] = 0
                        edgesflow_partial[(n1,n2)] = 0
                    edgesflow_partial[(n1,n2)] += 1

        for pair in edgesflow_partial.keys():
            edgesflow_partial[pair] /= float(nsolutions)
            edgesflow[pair] += edgesflow_partial[pair]
            edgesflow_partial[pair] = 0.0

    for pair in edgesflow.keys():
        n = pair[1]
        if n not in dict_flow.keys():
            dict_flow[n] = 0.0
        dict_flow[n] += edgesflow[pair]

        
    # nx.set_edge_attributes(G, dict_flow, name)
    return(dict_flow, edgesflow)

#=================================================================================
def countNodeFlowAllSimulations(G, simulations, pathDB):
    flow = dict()
    nodesFlow = [ dict()]* len(simulations) 
    edgesFlow = [ dict()]* len(simulations) 
    for i in range(len(simulations)):
        nodesFlow[i], edgesFlow[i]  = countNodesFlow(G,simulations[i]["solutions"], pathDB)
    return(nodesFlow, edgesFlow)

#=================================================================================

def clientList(G, solution):
    
    clientlist = dict()
    for tour in solution:
        tour = tour[1:]
        for n in tour:
            if n not in clientlist.keys():
                clientlist[n] = 0.0
            clientlist[n] += 1.0    
    return clientlist


#=================================================================================

def readRealFlow(G, file, pathDB):
    realData = readSimulations(file)
    nsamples = len(realData)
    realFlow = dict()
    realEgdeFlow = dict()
    weight = 1.0/float(nsamples)
    for n in G.nodes():
        realFlow[n] = 0.0
    for i in range(nsamples):
        instance_flow, edgesflow = countNodesFlow(G,realData[i]["solutions"], pathDB)
        for n, val in instance_flow.items():
            realFlow[n] += weight*val 

        for pair, val in edgesflow.items():
            if pair not in realEgdeFlow.keys():
                realEgdeFlow[pair] = 0.0
            realEgdeFlow[pair] += weight*val

    return realFlow, realEgdeFlow #, clientList(G,realData[0]["solutions"][0])


#=================================================================================

def getLeafNodes(classifier):
    leafs = dict()
    nnodes = classifier.tree_.node_count
    classes = classifier.classes_
    for n in range(nnodes):
        if classifier.tree_.children_left[n] == -1:
            maxval = 0
            leafClass = -1
            values = classifier.tree_.value[n][0]
            for c in range(len(classes)):
                if maxval < values[c]:
                    maxval = values[c]
                    leafClass = c
            leafs[n] = leafClass
            
    return leafs

#=================================================================================

def prepareData(G,simulations, simNodesFlow, restriction=[]):
    simModelInfo = [i['class'] for i in simulations]
    flow_array = [[] for i in range(len(simModelInfo))]
    if len(restriction)>0:
        nodes_list = restriction
    else:
        nodes_list = list(G.nodes())
    for i in range(len(simulations)):
        for n in nodes_list: 
            if n in simNodesFlow[i].keys():        
                flow_array[i].append(simNodesFlow[i][n])
            else:
                flow_array[i].append(0.0)
    
    return flow_array, simModelInfo

#=================================================================================

def avgData(simulations, weights, simNodesFlow,simEdgesFlow ):
    nbSims = len(simNodesFlow)
    avgSimNodesFlow = dict()
    avgSimEdgesFlow = dict()
    nbSimsPerModel= dict()
    for i in range(nbSims):
        class_ = simulations[i]['class'] 
        if class_ not in nbSimsPerModel.keys():
            nbSimsPerModel[class_]=0
        nbSimsPerModel[class_] +=1

    for i in range(nbSims):
        class_ = simulations[i]['class'] 
        if class_ in weights.keys():
            w = weights[class_]/float(nbSimsPerModel[class_])
        elif len(weights)==0:
            w =1/float((nbSims))
        else:
            w=0
        for n, val in simNodesFlow[i].items(): 
            if n not in avgSimNodesFlow.keys():
                avgSimNodesFlow[n] =0.0
            avgSimNodesFlow[n] +=w*val
        for n, val in simEdgesFlow[i].items(): 
            if n not in avgSimEdgesFlow.keys():
                avgSimEdgesFlow[n] =0.0
            avgSimEdgesFlow[n] +=w*val
            
    return avgSimNodesFlow, avgSimEdgesFlow

#=================================================================================

def avgSimFlow(simFlow ):
    nbSims = len(simFlow)
    avgSimFlow = dict()   
    for i in range(nbSims):
        w =1/float((nbSims))
        for n, val in simFlow[i].items(): 
            if n not in avgSimFlow.keys():
                avgSimFlow[n] =0.0
            avgSimFlow[n] +=w*val
            
    return avgSimFlow