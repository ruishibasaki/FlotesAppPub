#include "greedy_heuristic.hpp"
#include "route_engine.hpp"
#include "lns.hpp"

#include "value_gen.hpp"
#include <cassert> 
#include <sstream> 
#include <iostream> 
#include <time.h>
#include <chrono>
#include <vector>
#include <unordered_map>
//#include <omp.h>

void appendToursToFile(int iteration, int depot, 
                       std::unordered_map<std::string,std::string> paramsModel, 
                       std::vector<std::vector<Tour> > & solutions,
                       bool append=true);

void readModels(std::string modelsFile, const InstanceData &data,
                 std::vector<InstanceModel*>& models);
//=================================================================
//=================================================================
//=================================================================

void generate_random_instance(int depot, std::string instance, bool append){
    InstanceData data;
    readData( data, instance);

    std::vector<InstanceModel*> models;
    readModels("../main/models.txt", data, models);

    size_t nbInstance = 10;

    #pragma omp parallel for collapse(2) num_threads(10)
    for(size_t m=0; m < models.size();++m){
        for(size_t t=0; t < nbInstance ; ++t){
            InstanceData random_data;
            models[m]->generate(random_data);
            
            GreedyHeuristic constructer(random_data);
            LNSMetaheuristic lns(random_data); 

           
            constructer.solve(lns.bestSol);
            lns.solve();

            std::vector<std::vector<Tour> >  vecTours;
            for (size_t i = lns.vecSol.size(); i--;){
                if(!lns.vecSol[i].feas()) continue;
                if((lns.vecSol[i].cost()-lns.bestSol.cost())/lns.bestSol.cost() < 0.01){
                    vecTours.push_back(std::vector<Tour>());
                    solution2Tours(random_data, lns.vecSol[i], vecTours.back());
                }
            }
            
            
            
            #pragma omp critical
            {
            // std::cout << " Model: " << models[m]->_class << std::endl;
            // std::cout<<" Solution value: ";
            // for (size_t i = lns.vecSol.size(); i--;){
            //     if(lns.vecSol[i].feas()){
            //         std::cout<<std::setprecision(10)<<(lns.vecSol[i].cost()-lns.bestSol.cost())/lns.bestSol.cost()<<" ";
            //     }
            // }
            // std::cout<<std::endl;
                std::unordered_map<std::string, std::string> params;
                    params.insert(std::make_pair("class",models[m]->_class));
                //{"capa",std::to_string(random_data.capacitiesByVehiculeType[0])}
            
            appendToursToFile(m*nbInstance+t+1, depot, params,vecTours,append);
            append=true;
            }
            
            
        }
    }
    for(size_t m=0; m < models.size();++m){
        delete models[m];
    }
}

//=================================================================
//=================================================================
//=================================================================


// Generate random instances for a single depot
int main(int argc, const char *argv[]){   
    srand(std::chrono::system_clock::now().time_since_epoch().count());
    bool append=false;
    for(size_t i=1;i<=25;++i){
        //std::string instance(argv[1]);
        std::string instance("../instances/instanceLoc");
        if(*argv[1] == '0')
            instance += std::to_string(i)+".dat";
        else instance += std::to_string(i)+".nx";
        generate_random_instance(i, instance, append);
        append=true;
    }
    
    return EXIT_SUCCESS;
}


//=================================================================
//=================================================================
//=================================================================

// Output tours as list of coordinates since 
// I was not able to recoverer OSM Node corresponding to the
// Clients and Depots nodeID
void appendToursToFile(int iteration, int depot,
                       std::unordered_map<std::string,std::string> paramsModel, 
                       std::vector<std::vector<Tour> > & solutions,
                       bool append){
    if(solutions.empty()) return;

    std::ofstream of;
    if(append)
        of.open("../main/simulation.txt", std::ios_base::app);
    else
        of.open("../main/simulation.txt", std::ofstream::out);

    of <<"simulation " << iteration << "\n";
    of <<"depot " << depot << "\n";
    for(auto  ite = paramsModel.begin(); ite!=paramsModel.end(); ++ite){
        of << ite->first << " " << ite->second << "\n"; 
    }
    for (size_t i = 0; i<solutions.size();i++){
        of <<"solutions "<<i<<"\n";
        std::vector<Tour>& tours = solutions[i];
        size_t nbTours = tours.size();
        // std::cout << "Nb Tours :" << nbTours << std::endl;
        for(size_t t=0;t<nbTours;++t){
            size_t sz = tours[t].geolocs.size();
            for (size_t i = 0; i<sz; ++i){
                of << tours[t].nodeIDs[i];
                // of <<tours[t].geolocs[i].first << "," << tours[t].geolocs[i].second;
                if(i<(sz-1)){of<<" ";}else{of<<"\n";}
            }
        }
    }
    // of<<"\n";
    of.close();
}

//=================================================================

void readModels(std::string modelsFile, const InstanceData &data,
                 std::vector<InstanceModel*>& models){
    std::ifstream file(modelsFile);
    assert(file.is_open());
    double m, sd;
    size_t nb_types, count;
    std::string s, name;
    std::istringstream ss;
    bool reading_m = false;

    std::vector<double> m_demands;  
    std::vector<double> sd_demands;

    while(getline(file,s)){
        ss.str(s);
        ss>>s;
        if(s=="model"){
            reading_m = true;
            ss>>name>>nb_types;
            count=0;
            m_demands.resize(nb_types, 0);
            sd_demands.resize(nb_types, 0);
        }else if(reading_m){
            ss>>m_demands[count]>>sd_demands[count];
            ++count;
        }
        if(count==nb_types){
            if(reading_m) 
                models.push_back(new RandomDemandsPerTypeInstanceModel(&data, name, nb_types, m_demands,sd_demands));
            reading_m = false;
        } 
        ss.clear();
        s.clear();
    }
    file.close();

}
