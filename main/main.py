import osmnx as ox
import os
import sensor_location as sloc
import preprocessors as pp
import math
import folium 
from folium import plugins # this line is needed for BeautifyIcon
import branca.colormap as cm
import time
import sys

#=================================================================================

def filterModel(simModels, probPredict, classes):
    nclasses = len(classes)
    dim = list(simModels.values())[0][0]
    newModels= dict()
    weights = dict()
    for i in range(nclasses):
        if probPredict[i] >0:
            newModels.update({classes[i]: simModels[classes[i]]})
            weights[classes[i]] = probPredict[i]
        
    return newModels, weights

#=================================================================================

def predictModel(simModels, probPredict, classes):
    dim = list(simModels.values())[0][0]
    nclasses = len(classes)
    predModel =[]
    for d in range(dim):
        wmean = 0
        wstd = 0
        for i in range(nclasses):
            tpl = simModels[classes[i]][1][d]
            wmean = wmean +probPredict[i]*tpl[0]
            wstd = wstd + probPredict[i]*tpl[1]
        
        wmean = round(wmean)
        wstd = math.ceil(wstd)
        predModel.append((wmean,wstd))
    

    return [dim, predModel]
    

#=================================================================================
#Bhattacharyya distance or KL Divergence (choose the last one)
def KLDivergence(m1, m2):
    kld = math.log(m2[1]/m1[1]) + (math.pow(m1[1], 2) + math.pow((m1[0]-m2[0]), 2))/(2.0*math.pow(m2[1],2))-0.5
    return kld

#=================================================================================
#Bhattacharyya distance, KL Divergence or Jensen–Shannon divergence (choose the last one)
def checkConvergence( simModels):
    n = 1.0/float(len(simModels))
    dim = list(simModels.values())[0][0]
    maxkld=0
    for i in range(dim):
        centralMu = centralDev = 0.0
        for m1, val1 in simModels.values():
            model1 = val1[i]
            centralMu += n*(model1[0])
            centralDev += n*model1[1]
        centroid = (centralMu, centralDev)
        kld=0
        for m1, val1 in simModels.values():
            kld += n*(KLDivergence(val1[i], centroid))
        if kld > maxkld:
            maxkld = kld
    
    
    return maxkld

#=================================================================================
#Bhattacharyya distance or KL Divergence (choose the last one)
def diverge2Real(simModels, real):
    maxdiv = 0
    for key,val in simModels.items():
        comp = dict(real)
        comp.update({key: val})
        val = checkConvergence( comp)
        if maxdiv < val:
            maxdiv = val
    return maxdiv

#=================================================================================
def checkSolutionQuali(G,simEdgeFlow, realEgdeFlow):
    maxReal = max(realEgdeFlow.values())
    maxSim = max(simEdgeFlow.values())
    avgval=maxval=0.0
    avgNom=maxNom=0.0
    count=0
    for e in G.edges():
        diff = 0.0 
        diffNominal = 0.0 
        hasFlow =False
        if e in simEdgeFlow.keys():
            diff = simEdgeFlow[e]/maxSim
            diffNominal = simEdgeFlow[e]
            hasFlow =True
        if e in realEgdeFlow.keys():
            diff -= realEgdeFlow[e]/maxReal
            diffNominal -= realEgdeFlow[e]
            hasFlow =True
        if hasFlow:
            count +=1
        diff = abs(diff)
        avgval += diff
        if maxval < diff:
            maxval = diff

        diffNominal = abs(diffNominal)
        avgNom += diffNominal
        if maxNom < diffNominal:
            maxNom = diffNominal
    quali = {'avgDiffNorm' : float("{:.4f}".format(avgval/float(count))), 'maxDiffNorm' : float("{:.4f}".format(maxval))}
    quali.update({'avgDiffNom' : float("{:.4f}".format(avgNom/float(count))), 'maxDiffNom' : float("{:.4f}".format(maxNom))})
    quali.update({'avgNomR': float("{:.4f}".format(avgNom/float(count*maxReal))), 'maxNomR':  float("{:.4f}".format(maxNom/maxReal))})
    return quali


#=================================================================================

def printStatsToFile(filestart, quali, finalmodels, realmodel, convergence, itcount, nbcams, avgtime):
    fileout = open("./fileout.txt", 'a')
    fileout.write(filestart+" ")
    fileout.write(str(quali))
    fileout.write(" jsd: "+"{:.4f}".format(convergence))
    fileout.write(" jsd_real: " +"{:.4f}".format(diverge2Real(finalmodels, realmodel)))
    fileout.write(" iters: "+ str(itcount))
    fileout.write(" cams: "+ str(nbcams))
    fileout.write(" avgtimeit: "+"{:.2f}".format(avgtime)+"\n")
    fileout.close()

#=================================================================================

def printFoliumTours(G, fmap, realEgdeFlow):
    maxflow = max(realEgdeFlow.values())
    #colormap = cm.linear.Paired_06.scale(0.1, 1)
    colormap = cm.LinearColormap(colors=["limegreen","darkgreen", "gold", "orange", "red"],vmin=0.1, vmax=1.0)
    for e, val in realEgdeFlow.items() :
        if (e[0], e[1]) in realEgdeFlow.keys():
            val = max(val, realEgdeFlow[(e[0], e[1])])
        ox.plot_route_folium(G, e,route_map=fmap, color=colormap(max(val/maxflow,0.1)),
                                        weight=4)
    return colormap

#=================================================================================
def printFoliumSensors(id, gdf_nodes, mes_list_observed, fmap):

    for i in range(len(mes_list_observed)) :
        n = mes_list_observed[i]
        # print(gdf_nodes["osmid"][n])
        folium.Marker(
                location=(gdf_nodes["y"][n], gdf_nodes["x"][n]),
                icon=folium.plugins.BeautifyIcon(border_color='Blue',
                                    text_color='Blue',
                                    number=id,
                                    icon_shape='marker')
        ).add_to(fmap)
    
#=================================================================================

def main(G, pathDB, realModel, realFlow, realEgdeFlow, filestart):
    
    simModels = pp.readModels(filestart)
    pp.printModels("./models.txt", simModels)
    print("graph loaded, target model: ", realModel)
    print("trial models: ",simModels)

    c=3
    #parameters={'delta': [0.5+0.1*i for i in range(5)], 'max_depth': [c+1, 2*(c+1), 3*(c+1)]}
    #best_param = {'delta': 0.8, 'max_depth': c+1}
    hist_observed_nodes = []
    countModels = len(simModels)
    itcount =0
    avgtime = 0
    while True:
        walltime = time.time()
        print("make simulations...")
        os.system("../simulation/flotesApp 0")
        simulations = pp.readSimulations('./simulation.txt')

        print("load data...")
        simNodesFlow, simEdgesFlow  = pp.countNodeFlowAllSimulations(G,simulations, pathDB)
        print("data loaded.")

        # if count==0:
        #     print("GridSearchCV...")
        #     best_param = sloc.calibrate(c, flow_array, simModelInfo, hist_observed_nodes, parameters)
        print("placing sensors...")
        #mes_list_observed, clf=sloc.placeSensorsHF(G,simNodesFlow, simulations, hist_observed_nodes, c)
        mes_list_observed, clf = sloc.placeSensors(G ,simulations, simNodesFlow, hist_observed_nodes, c)
        print("place cameras on: ", mes_list_observed, 'vec_pos: ', hist_observed_nodes)
        
        #probPredict = sloc.predictWithClassifier(G, clf, realFlow, hist_observed_nodes) #USE THIS WITH placeSensorsHF OR placeSensorsHFR
        probPredict = sloc.predictWithClassifier(G, clf, realFlow)
        print("predicted model: ",probPredict," dict: ",list(clf.classes_labels_))
        newModels, weights = filterModel(simModels, probPredict[0], list(clf.classes_labels_))

        avgSimNodesFlow, avgSimEdgesFlow = pp.avgData(simulations, weights, simNodesFlow, simEdgesFlow )
        quali  = checkSolutionQuali(G,avgSimEdgesFlow, realEgdeFlow)
        print("quality of sim:", quali)

        avgtime += time.time()-walltime
        itcount += 1 
        convergence = checkConvergence( newModels)
        print("JS-Divergence ", convergence)
        if convergence < 1e-8 or len(newModels) == 1 or itcount >= 4:
            simModels=newModels
            break

        predm =  predictModel(simModels, probPredict[0], list(clf.classes_labels_))
        if predm not in newModels.values():
            newModels['class'+str(countModels+1)] = predm
            countModels +=1
        simModels=newModels
        print("predicted model: ",predm)
        print("new models: ",simModels)
        pp.printModels("./models.txt", simModels)
        
    
    print("final models: ",simModels)
    avgSimNodesFlow, avgSimEdgesFlow = pp.avgData(simulations, weights, simNodesFlow, simEdgesFlow )
    quali  = checkSolutionQuali(G,avgSimEdgesFlow, realEgdeFlow)
    printStatsToFile(filestart, quali, simModels, realModel, convergence, itcount, len(hist_observed_nodes), avgtime/float(itcount))


    fMap2 = folium.Map(location=[47.205, -1.55], zoom_start=15, tiles = "cartodbpositron", kwargs={'zoomSnap': 0.1})
    ftours2 = folium.map.FeatureGroup(name="tours", show=1)

    G = ox.load_graphml(filepath="./graphs/nantes_graph_truncated_wspeeds.xml")
    gdf_nodes = ox.graph_to_gdfs(G,nodes=True,edges=False)

    pathDB =dict()
    print("make simulations...")
    os.system("../simulation/flotesApp 1")
    simulations = pp.readSimulations('./simulation.txt')
    print("load data...")
    simNodesFlow, simEdgesFlow  = pp.countNodeFlowAllSimulations(G,simulations, pathDB)
    avgSimNodesFlow, avgSimEdgesFlow = pp.avgData(simulations, weights, simNodesFlow, simEdgesFlow )

    printFoliumTours(G,ftours2, avgSimEdgesFlow).add_to(fMap2)
    ftours2.add_to(fMap2)
    fMap2.save("mapsensorsfinal.html")

if __name__ == "__main__":
    ## OSMNx Parameters
    ox.settings.log_console=False
    ox.settings.use_cache=True
    print("load graph")
    G = ox.load_graphml(filepath="./graphs/nantes_graph_simplified_wspeeds.xml")

    pathDB =dict()
    realModel = pp.readModels("./target/target_model_4.txt")
    realFlow, realEgdeFlow = pp.readRealFlow(G, './target/target_flow.txt', pathDB)
    for i in range(1):
        print("#======================================================")
        main(G, pathDB, realModel, realFlow, realEgdeFlow, str(sys.argv[1]) )  
    

