
#include "value_gen.hpp"

#include <random>
#include <math.h> 
#include <chrono>

#include <iostream>



//=================================================================
//=================================================================
//=================================================================

void InstanceModel::copyBaseInfo(InstanceData& data){
    data.depotTimeWindow = _base->depotTimeWindow;
    data.depotCoordLoc = _base->depotCoordLoc;
    data.depotNodeCoordLoc = _base->depotNodeCoordLoc;
    data.clients = _base->clients;

    size_t nb_clients = data.clients.size();
    for(size_t i = 0; i<nb_clients; ++i){
        data.clients[i].service = 5*60;              //in sec.
        data.clients[i].timeWindow.first = 5*3600;   //in sec.
        data.clients[i].timeWindow.second = 8*3600; //in sec.
    }
    data.capacitiesByVehiculeType = _base->capacitiesByVehiculeType;
    data.distanceTable = _base->distanceTable;
    data.durationTable = _base->durationTable;
    data.maxDist = _base->maxDist;
    data.depotNodeID = _base->depotNodeID;
}
//=================================================================
//=================================================================
//=================================================================


void RandomDemandsPerTypeInstanceModel::generate(InstanceData& data){
    copyBaseInfo(data);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    size_t nbClients = data.clients.size();
    for (size_t i = 0; i < nbClients; i++){
        size_t type = RAND_SEQ4[i%MAX_CLIENTS]-1; 
        std::normal_distribution<double> norm_class(_m_demands[type],_sd_demands[type]);
        double random_dmand = std::max(1.0,ceil(norm_class(generator)));
        data.clients[i].demand = std::min(random_dmand, 95.0);
        
    }
}


//=================================================================
//=================================================================
//=================================================================

void RandomVehicleCapacityInstanceModel::generate(InstanceData& data){
    copyBaseInfo(data);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    


    // // Randomly pick the parameters for the distribution
    // unsigned int nb_class    = _m_vehicleCapa.size();
    // std::uniform_int_distribution<int> unif_distr(0,nb_class-1);
    // unsigned int class_distr =  unif_distr(generator);
    
    // // Get Mean and Std. Deviation of the picked class
    // double m_class  = _m_vehicleCapa[class_distr];
    // double sd_class = 1.;
    // if(class_distr<_sd_vehicleCapa.size()){
    //     sd_class = _sd_vehicleCapa[class_distr];
    // } 
    // std::cout << " sd_class: " << sd_class << std::endl;
    std::normal_distribution<double> norm_class(_m_vehicleCapa,_sd_vehicleCapa);

    double random_capa = ceil(norm_class(generator));
    std::vector<double> capacities;
    capacities.push_back(random_capa);
    data.capacitiesByVehiculeType = capacities;
}
