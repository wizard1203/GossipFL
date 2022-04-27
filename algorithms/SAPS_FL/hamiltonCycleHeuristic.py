import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

route = []

"""https://github.com/willsyo/hamilton_cycle/tree/88a89276f1f20ce0e43ce9a386f91367047198cf"""

def createRoute(G):
    V = list(G.nodes)
    R = nx.Graph()
    for v in route:
        R.add_node(V[v])
    for i in range(len(route)-1):
        R.add_edge(V[route[i]], V[route[i+1]])
    return R

def isValid(G, v, ncount, path):
    A = nx.to_numpy_array(G)
    # Check if current vertex and last vertex in the path are adjacent
    if A[path[ncount-1]][v] == 0:
        return False

    # Check if current vertex not already in path
    for vertex in path:
        if vertex == v:
            return False
    return True

# def hamiltonCycleHeuristic(G, path, ncount):
#     global route
#     A = nx.to_numpy_array(G)
#     # If all vertices are in the path, then...
#     if ncount == G.number_of_nodes():
#         # If the last vertex is adjacent to the first vertex in path to make the cycle, there is a cycle
#         if A[path[ncount-1]][path[0]] == 1:
#             return True
#         else:
#             return False
    
#     # Otherwise we try different vertices and test if they are valid next nodes for hamilton cycle.
#     for v in range(0, G.number_of_nodes()):
#         if isValid(G, v, ncount, path) == True:
#             path[ncount] = v # Add valid vertex to the path
#         # Recursively call until the hamilton cycle is complete 
#         if hamiltonCycleHeuristic(G, path, ncount+1) == True:
#             route = path
#             return True
        
#         # If we're here, then the current vertex is not a part of the solution
#         path[ncount] = -1
#     return False

# def hamiltonCycle(G,start=0):
#     path = [-1]*G.number_of_nodes()
#     # Start at node 0
#     path[0] = start
#     if hamiltonCycleHeuristic(G, path, 1) == False:
#         # There is no hamilton cycle
#         print("No Hamilton Cycle")
#         return False
#     route.append(path[0])
#     R = createRoute(G)
#     return R


def hamiltonCycleHeuristic(G, path, ncount, time_before, time_limit=4):
    time_s = time.time()
    print("time_before: ", time_before)
    global route
    A = nx.to_numpy_array(G)
    # If all vertices are in the path, then...
    if ncount == G.number_of_nodes():
        # If the last vertex is adjacent to the first vertex in path to make the cycle, there is a cycle
        time_before += time.time() - time_s
        if A[path[ncount-1]][path[0]] == 1:
            return time_before, True
        else:
            return time_before, False
    # Otherwise we try different vertices and test if they are valid next nodes for hamilton cycle.
    time_now = time_before
    for v in range(0, G.number_of_nodes()):
        time_s = time.time()
        if isValid(G, v, ncount, path) == True:
            path[ncount] = v # Add valid vertex to the path
        # Recursively call until the hamilton cycle is complete
        time_now += time.time() - time_s
        time_now, if_hamil = hamiltonCycleHeuristic(G, path, ncount+1, time_now, time_limit)
        # if hamiltonCycleHeuristic(G, path, ncount+1, time) == True:
        print("time_now: ", time_now)
        time_s = time.time()
        if if_hamil:
            route = path
            return time_now, True
        time_now += time.time() - time_s
        if time_now > time_limit:
            return time_now, False
        # If we're here, then the current vertex is not a part of the solution
        path[ncount] = -1
    return time_now, False

def hamiltonCycle(G,start=0, time_limit=4):
    path = [-1]*G.number_of_nodes()
    # Start at node 0
    path[0] = start
    time_now = 0
    time_now, if_hamil = hamiltonCycleHeuristic(G, path, 1, time_now, time_limit)
    if not if_hamil:
    # if hamiltonCycleHeuristic(G, path, ncount+1, time) == False:
        # There is no hamilton cycle
        print("No Hamilton Cycle")
        return None
    route.append(path[0])
    R = createRoute(G)
    return R




if __name__ == '__main__':
    # Create matrix
    G = nx.grid_graph(dim=[6,6])
    R = hamiltonCycle(G)
    edge_colors = ["orange" if R.has_edge(u,v) else "gray" for u,v in G.edges]
    edge_widths = [5 if R.has_edge(u,v) else 0.5 for u,v in G.edges]

    #print graph
    graph,ax = plt.subplots(1,1,figsize=(10,10))
    nx.draw(G, 
            pos=nx.kamada_kawai_layout(G), 
            ax=ax,
            with_labels=True, 
            node_color='#444444',
            font_color="white",
            edge_color=edge_colors,
            width=edge_widths)