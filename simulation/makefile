# compiler
CC = g++

# module name
NAME = flotesApp

# basic directory
DIR = ..

# debug switches
#SW = -Wall -ggdb3
# production switches
SW = -w -O3 

# default target- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

default: $(NAME)

# clean target- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

clean::
	rm -f $(DIR)/bin/*.o $(DIR)/bin/*~ $(NAME)


# cplex- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

DIRINC = -I$(DIR)/include

INSTANCE_DATA_O = $(DIR)/bin/instance_data.o
GREEDY_HEURISTIC_O = $(DIR)/bin/greedy_heuristic.o
ROUTE_ENGINE_O = $(DIR)/bin/route_engine.o
DESTROY_NEIGHBORHOODS_O = $(DIR)/bin/destroy_neighborhood.o
REPAIR_NEIGHBORHOODS_O = $(DIR)/bin/repair_neighborhood.o
OPERATIONS_O = $(DIR)/bin/operations.o
EJECTION_O = $(DIR)/bin/ejection_search.o
LNS_O = $(DIR)/bin/lns.o
VALUE_GEN_O = $(DIR)/bin/value_gen.o

OTHEROBJ =$(GREEDY_HEURISTIC_O) $(ROUTE_ENGINE_O) \
			$(DESTROY_NEIGHBORHOODS_O) $(LNS_O) $(INSTANCE_DATA_O) \
			$(OPERATIONS_O) $(REPAIR_NEIGHBORHOODS_O) $(EJECTION_O) $ \
			$(VALUE_GEN_O)

# ---------------------------------------------------------------------


$(VALUE_GEN_O): ./value_gen.cpp
	$(CC) -c $< -o $@ $(DIRINC) $(SW) -std=c++11

$(REPAIR_NEIGHBORHOODS_O): $(DIR)/lns/repair_neighborhood.cpp
	$(CC) -c $< -o $@ $(DIRINC) $(SW) -std=c++11

$(EJECTION_O): $(DIR)/lns/ejection_search.cpp
	$(CC) -c $< -o $@ $(DIRINC) $(SW) -std=c++11

$(OPERATIONS_O): $(DIR)/lns/operations.cpp 
	$(CC) -c $< -o $@  $(DIRINC) $(SW) -std=c++11

$(INSTANCE_DATA_O): ./instance_data.cpp
	$(CC) -c $< -o $@ $(DIRINC) $(SW) -std=c++11

$(LNS_O): $(DIR)/lns/lns.cpp
	$(CC) -c $< -o $@ $(DIRINC) $(SW) -std=c++11

$(DESTROY_NEIGHBORHOODS_O): $(DIR)/lns/destroy_neighborhood.cpp
	$(CC) -c $< -o $@ $(DIRINC) $(SW) -std=c++11

$(ROUTE_ENGINE_O): ./route_engine.cpp
	$(CC) -c $< -o $@ $(DIRINC) $(SW)

$(GREEDY_HEURISTIC_O): $(DIR)/lns/greedy_heuristic.cpp
	$(CC) -c $< -o $@ $(DIRINC) $(SW)

$(DIR)/bin/simulator.o: ./simulator.cpp
	$(CC) -c $< -o $@ $(DIRINC) $(SW)

$(NAME): $(DIR)/bin/simulator.o  $(OTHEROBJ) 
	$(CC) -o $(NAME) $< $(OTHEROBJ) 