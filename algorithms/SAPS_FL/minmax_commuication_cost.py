import copy
import numpy as np
import copy
from collections import deque

import networkx as nx

from .utils import Bandwidth_Decrease_Factor

from .hamiltonCycleHeuristic import hamiltonCycle

def hasPath(Gf, s, t, path):
    # BFS algorithm
    V = len(Gf)
    visited = list(range(V))
    for i in range(V):
        visited[i] = False
    visited[s] = True
    queue = deque([s])
    while queue:
        temp = queue.popleft()
        if temp == t:
            return True
        # print("temp =", temp)
        for i in range(V):
            if not visited[i] and (Gf[temp][i] > 0):
                queue.append(i)
                visited[i] = True
                path[i] = temp   # record precursor
    return visited[t]


def max_flow(graph, s, t):
    maxFlow = 0
    Gf = copy.deepcopy(graph)
    V = len(Gf)
    path = list(range(V))
    while hasPath(Gf, s, t, path):
        min_flow = float('inf')

        # find cf(p)
        v = t
        while v != s:
            u = path[v]
            min_flow = min(min_flow, Gf[u][v])
            v = path[v]
        # print(min_flow)

        # add flow in every edge of the augument path
        v = t
        while v != s:
            u = path[v]
            Gf[u][v] -= min_flow
            Gf[v][u] += min_flow
            v = path[v]

        maxFlow += min_flow
    return maxFlow, Gf

def generate_capacity(bandwidth, roles, bandwidth_threshold):
    nworkers = len(roles)
    capacity = np.zeros([nworkers+2, nworkers+2])
    for i in range(nworkers):
        for j in range(nworkers):
            # if (not bandwidth[i][j] > bandwidth_threshold) and ((int(roles[i])==1) ^ (int(roles[j])==1) == 1) and (i != j):
            if (bandwidth[i][j] > bandwidth_threshold) and (roles[i])==1 and roles[j]==0 and (i != j):
                capacity[i+1][j+1] = 1
    for i in range(nworkers):
        if roles[i] == 1:
            capacity[0][i+1] = 1
        else:
            capacity[i+1][nworkers+1] = 1

    return capacity


#############################################
class general_max_match:
    def __init__(self, e, n_workers):
        self.Nworkers = 100 # N
        # disjoint set
        # belong[nworkers]
        self.belong = [0] * self.Nworkers

        # int n, match[N];
        # vector<int> e[N];
        # int Q[N], rear;
        # int next[N], mark[N], vis[N];
        self.n_workers = n_workers
        self.match = [0] * self.Nworkers
        # e = [[]] * Nworkers  # [] object will be referenced but not created
        # self.e = [[] for i in range(self.Nworkers)]
        self.e = e
        self.Q = [0] * self.Nworkers
        self.rear = 0
        self.next1 = [0] * self.Nworkers
        self.mark = [0] * self.Nworkers
        self.vis = [0] * self.Nworkers

        self.LCA_static_t = 0

    def begin_match(self):
        for i in range(self.n_workers):
            self.match[i] = -1
        for i in range(self.n_workers):
            if (self.match[i] == -1):
                self.aug(i)
        return self.match

    # int findb(int x) { 
    # 	return belong[x] == x ? x : belong[x] = findb(belong[x]);
    # }
    # void unit(int a, int b) {
    # 	a = findb(a);
    # 	b = findb(b);
    # 	if (a != b) belong[a] = b;
    # }
    def findb(self, x):
        if self.belong[x] == x:
            return x
        else:
            self.belong[x] = self.findb(self.belong[x])
            return self.belong[x]

    def unit(self, a, b):
        a = self.findb(a)
        b = self.findb(b)
        if a != b:
            self.belong[a] = b

    # find the common ancester of x and y
    # int LCA(int x, int y) {
    #   static int t = 0; t++;
    #   while (true) {
    # 		if (x != -1) {
    # 			x = findb(x); // map the node to the corresponding blossom
    #  			if (vis[x] == t) return x;
    # 			vis[x] = t;
    # 			if (match[x] != -1) x = next[match[x]];
    # 			else x = -1;
    # 		}
    # 		swap(x, y);
    # 	}
    # }
    def swap(self, x, y):
        # print('111111')
        temp = copy.deepcopy(x)
        x = copy.deepcopy(y)
        y = copy.deepcopy(temp)


    def LCA(self, x, y):
        # print('running')
        self.LCA_static_t += 1
        while(True):
            if (x != -1):
                x = self.findb(x)
                if self.vis[x] == self.LCA_static_t:
                    return x
                self.vis[x] = self.LCA_static_t
                if (self.match[x] != -1):
                    x = self.next1[self.match[x]]
                else:
                    x = -1
            temp = copy.deepcopy(x)
            x = copy.deepcopy(y)
            y = copy.deepcopy(temp)


    def group(self, a, p):
        while(a != p):
            # print('222')
            b = self.match[a]
            c = self.next1[b]
            if (self.findb(c) != p):
                self.next1[c] = b
            if (self.mark[b] == 2):
                self.Q[self.rear] = b
                self.mark[self.Q[self.rear]] = 1
                self.rear += 1
            if (self.mark[c] == 2):
                self.Q[self.rear] = c
                self.mark[self.Q[self.rear]] = 1
                self.rear += 1
            self.unit(a, b)
            self.unit(b, c)
            a = c

    # augment
    # https://blog.csdn.net/yihuikang/article/details/10460997
    def aug(self, s):
        for i in range(self.n_workers):
            self.next1[i] = -1
            self.belong[i] = i
            self.mark[i] = 0
            self.vis[i] = -1
        self.mark[s] = 1
        self.Q[0] = s
        self.rear = 1
        front = 0
        while(self.match[s] == -1 and front < self.rear):
            x = self.Q[front]
            # print("333")
            for i in range(len(self.e[x])):
                y = self.e[x][i]
                if (self.match[x] == y): continue
                if (self.findb(x) == self.findb(y)): continue
                if (self.mark[y] == 2): continue
                if (self.mark[y] == 1):
                    r = self.LCA(x, y)
                    if (self.findb(x) != r): self.next1[x] = y
                    if (self.findb(y) != r): self.next1[y] = x
                    self.group(x, r)
                    self.group(y, r)
                elif (self.match[y] == -1):
                    self.next1[y] = x
                    u = y
                    while(u != -1):
                        v = self.next1[u]
                        mv = self.match[v]
                        self.match[v] = u
                        self.match[u] = v
                        u = mv
                    break
                else:
                    self.next1[y] = x
                    self.Q[self.rear] = self.match[y]
                    self.mark[self.Q[self.rear]] = 1
                    self.rear += 1 
                    self.mark[y] = 2
                front += 1


def general_graph_max_match(bandwidth, bandwidth_threshold, nworkers):
    nworkers = nworkers
    capacity = np.zeros([nworkers, nworkers])

    edges = {}
    for i in range(nworkers):
        edges[i] = []
        for j in range(nworkers):
            # if (not bandwidth[i][j] > bandwidth_threshold) and ((int(roles[i])==1) ^ (int(roles[j])==1) == 1) and (i != j):
            if (bandwidth[i][j] > bandwidth_threshold) and (i != j) and (i > j):
                capacity[i][j] = capacity[j][i] = 1
                edges[i].append(j)
                edges[j].append(i)

    match = matching(edges)
    total_match = len(match)

    return total_match/2, match

def limited_graph_max_match(bandwidth, bandwidth_threshold, nworkers):
    nworkers = nworkers
    edges = {}
    for i in range(nworkers):
        edges[i] = []
    # print("============\n", bandwidth, "============\n")
    for i in bandwidth:
        # edges[i] = []
        for j in bandwidth[i]:
            if (bandwidth[i][j] > bandwidth_threshold):
                edges[i].append(j)

    match = matching(edges)
    total_match = len(match)

    return total_match/2, match


if 'True' not in globals():
    globals()['True'] = not None
    globals()['False'] = not True

class unionFind:
    '''Union Find data structure. Modified from Josiah Carlson's code,
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
to allow arbitrarily many arguments in unions, use [] syntax for finds,
and eliminate unnecessary code.'''

    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        '''Find the root of the set that an object is in.
Object must be hashable; previously unknown objects become new singleton sets.'''

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object
        
        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
        
        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, *objects):
        '''Find the sets containing the given objects and merge them all.'''
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest

def matching(G, initialMatching = {}):
    '''Find a maximum cardinality matching in a graph G.
G is represented in modified GvR form: iter(G) lists its vertices;
iter(G[v]) lists the neighbors of v; w in G[v] tests adjacency.
The output is a dictionary mapping vertices to their matches;
unmatched vertices are omitted from the dictionary.

We use Edmonds' blossom-contraction algorithm, as described e.g.
in Galil's 1986 Computing Surveys paper.'''

    # Copy initial matching so we can use it nondestructively
    matching = {}
    for x in initialMatching:
        matching[x] = initialMatching[x]


    # Form greedy matching to avoid some iterations of augmentation
    for v in G:
        if v not in matching:
            for w in G[v]:
                if w not in matching:
                    matching[v] = w
                    matching[w] = v
                    break

    def augment():
        '''Search for a single augmenting path.
Return value is true if the matching size was increased, false otherwise.'''
    
        # Data structures for augmenting path search:
        #
        # leader: union-find structure; the leader of a blossom is one
        # of its vertices (not necessarily topmost), and leader[v] always
        # points to the leader of the largest blossom containing v
        #
        # S: dictionary of leader at even levels of the structure tree.
        # Dictionary keys are names of leader (as returned by the union-find
        # data structure) and values are the structure tree parent of the blossom
        # (a T-node, or the top vertex if the blossom is a root of a structure tree).
        #
        # T: dictionary of vertices at odd levels of the structure tree.
        # Dictionary keys are the vertices; T[x] is a vertex with an unmatched
        # edge to x.  To find the parent in the structure tree, use leader[T[x]].
        #
        # unexplored: collection of unexplored vertices within leader of S
        #
        # base: if x was originally a T-vertex, but becomes part of a blossom,
        # base[t] will be the pair (v,w) at the base of the blossom, where v and t
        # are on the same side of the blossom and w is on the other side.

        leader = unionFind()
        S = {}
        T = {}
        unexplored = []
        base = {}
        
        # Subroutines for augmenting path search.
        # Many of these are called only from one place, but are split out
        # as subroutines to improve modularization and readability.
        
        def blossom(v,w,a):
            '''Create a new blossom from edge v-w with common ancestor a.'''
            
            def findSide(v,w):
                path = [leader[v]]
                b = (v,w)   # new base for all T nodes found on the path
                while path[-1] != a:
                    tnode = S[path[-1]]
                    path.append(tnode)
                    base[tnode] = b
                    unexplored.append(tnode)
                    path.append(leader[T[tnode]])
                return path
            
            a = leader[a]   # sanity check
            path1,path2 = findSide(v,w), findSide(w,v)
            leader.union(*path1)
            leader.union(*path2)
            S[leader[a]] = S[a] # update structure tree

        topless = object()  # should be unequal to any graph vertex
        def alternatingPath(start, goal = topless):
            '''Return sequence of vertices on alternating path from start to goal.
Goal must be a T node along the path from the start to the root of the structure tree.
If goal is omitted, we find an alternating path to the structure tree root.'''
            path = []
            while 1:
                while start in T:
                    v, w = base[start]
                    vs = alternatingPath(v, start)
                    vs.reverse()
                    path += vs
                    start = w
                path.append(start)
                if start not in matching:
                    return path     # reached top of structure tree, done!
                tnode = matching[start]
                path.append(tnode)
                if tnode == goal:
                    return path     # finished recursive subpath
                start = T[tnode]
                
        def pairs(L):
            '''Utility to partition list into pairs of items.
If list has odd length, the final pair is omitted silently.'''
            i = 0
            while i < len(L) - 1:
                yield L[i],L[i+1]
                i += 2
            
        def alternate(v):
            '''Make v unmatched by alternating the path to the root of its structure tree.'''
            path = alternatingPath(v)
            path.reverse()
            for x,y in pairs(path):
                matching[x] = y
                matching[y] = x

        def addMatch(v, w):
            '''Here with an S-S edge vw connecting vertices in different structure trees.
Find the corresponding augmenting path and use it to augment the matching.'''
            alternate(v)
            alternate(w)
            matching[v] = w
            matching[w] = v
            
        def ss(v,w):
            '''Handle detection of an S-S edge in augmenting path search.
Like augment(), returns true iff the matching size was increased.'''
    
            if leader[v] == leader[w]:
                return False        # self-loop within blossom, ignore
    
            # parallel search up two branches of structure tree
            # until we find a common ancestor of v and w
            path1, head1 = {}, v
            path2, head2 = {}, w
    
            def step(path, head):
                head = leader[head]
                parent = leader[S[head]]
                if parent == head:
                    return head     # found root of structure tree
                path[head] = parent
                path[parent] = leader[T[parent]]
                return path[parent]
                
            while 1:
                head1 = step(path1, head1)
                head2 = step(path2, head2)
                
                if head1 == head2:
                    blossom(v, w, head1)
                    return False
                
                if leader[S[head1]] == head1 and leader[S[head2]] == head2:
                    addMatch(v, w)
                    return True
                
                if head1 in path2:
                    blossom(v, w, head1)
                    return False
                
                if head2 in path1:
                    blossom(v, w, head2)
                    return False    

        # Start of main augmenting path search code.

        for v in G:
            if v not in matching:
                S[v] = v
                unexplored.append(v)

        current = 0     # index into unexplored, in FIFO order so we get short paths
        while current < len(unexplored):
            v = unexplored[current]
            current += 1

            for w in G[v]:
                if leader[w] in S:  # S-S edge: blossom or augmenting path
                    if ss(v,w):
                        return True

                elif w not in T:    # previously unexplored node, add as T-node
                    T[w] = v
                    u = matching[w]
                    if leader[u] not in S:
                        S[u] = w    # and add its match as an S-node
                        unexplored.append(u)
                        
        return False    # ran out of graph without finding an augmenting path
                        
    # augment the matching until it is maximum
    while augment():
        pass

    return matching

#######################################################################
def get_limited_graph(bandwidth, bandwidth_threshold):
    nworkers = len(bandwidth)
    nworkers = nworkers
    match = {}
    # capacity = np.zeros([nworkers, nworkers])
    probability_edges = {}
    for i in range(nworkers):
        probability_edges[i] = {}
    print("in get_limited_graph bandwidth", bandwidth[0][4])
    print(bandwidth_threshold)
    for i in range(nworkers):
        #probability_edges[i] = {}
        match[i] = -1
        for j in range(nworkers):
            if (bandwidth[i][j] > bandwidth_threshold) and (bandwidth[j][i] > bandwidth_threshold) and (i != j) and (i > j):
                # capacity[i][j] = capacity[j][i] = 1
                if bandwidth[i][j] > bandwidth[j][i]:
                    #print(bandwidth[j][i])
                    probability_edges[i][j] = bandwidth[j][i]
                    probability_edges[j][i] = bandwidth[j][i]
                else:
                    #print(bandwidth[i][j])
                    probability_edges[i][j] = bandwidth[i][j]
                    probability_edges[j][i] = bandwidth[i][j]

    return probability_edges


def minmax_communication_match(bandwidth, roles, no_ask=1, islimited=True):
    nworkers = len(bandwidth)
    edges = nworkers*nworkers
    if no_ask:
        if islimited:
            bandwidth_sort = np.array([])
            #print("bandwidth input:", bandwidth)
            for i in bandwidth:
                for j in bandwidth[i]:
                    bandwidth_sort = np.append(bandwidth_sort, bandwidth[i][j])
            bandwidth_sort = np.sort(bandwidth_sort, kind='quicksort')
        else:
            bandwidth_sort = bandwidth.reshape(nworkers*nworkers)
            bandwidth_sort = np.sort(bandwidth_sort, kind='quicksort')
    else:
        bandwidth_sort = bandwidth.reshape(nworkers*nworkers)
        bandwidth_sort = np.sort(bandwidth_sort, kind='quicksort')
    #print("bandwidth_sort", bandwidth_sort)
    left = 0
    if no_ask:
        if islimited:
            right = len(bandwidth_sort)
        else:
            right = edges
    else:
        right = edges
    x = 0
    match = None
    while(1):
        # print("left: %d, right: %d"%(left, right))
        if x == int((left + right)/2):
            return match, bandwidth_threshold
        else:
            x = int((left + right)/2)
        bandwidth_threshold = bandwidth_sort[x]
        if no_ask:
            if islimited:
                flow, match_temp = limited_graph_max_match(bandwidth, bandwidth_threshold, nworkers)
            else:
                flow, match_temp = general_graph_max_match(bandwidth, bandwidth_threshold, nworkers)
        else:
            capacity = generate_capacity(bandwidth, roles, bandwidth_threshold)
            # print("capacity", capacity)
            flow, capacity_match = max_flow(capacity, 0, nworkers+1)

        if flow == int(nworkers/2):
            left = x
            if no_ask:
                match = match_temp
            else:
                match = (capacity[1:nworkers+1, 1:nworkers+1] - capacity_match[1:nworkers+1, 1:nworkers+1])==1
        else:
            right = x

def get_match_and_bandwidth(bandwidth, roles=None, no_ask=1, islimited=True):
    if no_ask:
        if islimited:
            match, bandwidth_threshold = minmax_communication_match(bandwidth, roles, 1, islimited)
        else:
            match, bandwidth_threshold = minmax_communication_match(bandwidth, roles, 1, islimited)
    else:
        match_matrix, bandwidth_threshold = minmax_communication_match(bandwidth, roles, 0, islimited)
        match = {}
        for i in range(workers):
            for j in range(workers):
                if match[i][j] == True:
                    match[i] = j
                    match[j] = i
                    # print('match (active :%d, passive: %d)' % (i, j))
    return match, bandwidth_threshold

def get_random_match_and_bandwidth(bandwidth, workers):
    index_roles = np.arange(workers)
    np.random.shuffle(index_roles)
    roles = np.ones(workers)
    roles[index_roles[0:int(workers/2)]]=0

    random_match = {}
    random_bandwidth_threshold = 100
    for i in range(int(workers/2)):
        random_match[index_roles[i]] = index_roles[i+int(workers/2)]
        random_match[index_roles[i+int(workers/2)]] = index_roles[i]
        if random_bandwidth_threshold > bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]:
            random_bandwidth_threshold = bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]
    return random_match, random_bandwidth_threshold

def get_ring_match_and_bandwidth(bandwidth, workers):

    ring_match = {}
    ring_bandwidth_threshold = 100
    for i in range(workers):
        ring_match[i] = []
        ring_match[i].append((i+1) % workers)
        ring_match[i].append((i-1) % workers)
        for j in ring_match[i]:
            if ring_bandwidth_threshold > bandwidth[i][j]:
                ring_bandwidth_threshold = bandwidth[i][j]
    return ring_match, ring_bandwidth_threshold


def build_nx_graph(bandwidth, ring_bandwidth_threshold):
    print(bandwidth)
    bandwidth_new = np.zeros(bandwidth.shape)
    bandwidth_new[bandwidth > ring_bandwidth_threshold] = 1
    print(bandwidth_new)
    # print(bandwidth_new)
    G =  nx.DiGraph(bandwidth_new)
    # print(G)
    return G

def hamilton(G):
    F = [(G,[list(G.nodes())[0]])]
    n = G.number_of_nodes()
    while F:
        graph,path = F.pop()
        confs = []
        neighbors = (node for node in graph.neighbors(path[-1]) 
                     if node != path[-1]) #exclude self loops
        for neighbor in neighbors:
            conf_p = path[:]
            conf_p.append(neighbor)
            conf_g = nx.Graph(graph)
            conf_g.remove_node(path[-1])
            confs.append((conf_g,conf_p))
        for g,p in confs:
            if len(p)==n:
                return p
            else:
                F.append((g,p))
    return None

def get_ring_match_and_bandwidth_optimum(bandwidth, workers, epsilon=0.01):

    bandwidth_min = bandwidth.min()
    bandwidth_max = bandwidth.max()

    left = bandwidth_min
    right = bandwidth_max
    middle = bandwidth_min + (bandwidth_max - bandwidth_min) / 2 

    precision = (bandwidth_max - bandwidth_min) * epsilon
    path = None
    # while abs(middle - left) <  precision or abs(middle - right) <  precision:
    while abs(right - left) > precision:
        print("Finding zone: [{}, {}]".format(left, right))
        bandwidth_new = build_nx_graph(bandwidth, ring_bandwidth_threshold=middle)
        # print(bandwidth_new)
        # path = hamilton(bandwidth_new)
        path = hamiltonCycle(bandwidth_new, 0)
        if path is None:
            print("Not Find hamilton")
            right = middle
        else:
            print("Find hamilton")
            save_path = copy.deepcopy(path)
            left = middle
        middle = left + (right - left) / 2

    bandwidth_threshold = left
    return save_path, bandwidth_threshold


def get_FedAVG_match_and_bandwidth(bandwidth, workers, choose_clients_num=None):

    min_raws = {}
    # print(bandwidth)
    # print(bandwidth[0])
    # print(bandwidth[0][1])
    for i in range(workers):
        min_raws[i] = 5000
        for j in range(workers):
            if min_raws[i] > bandwidth[i][j] and i!=j:
                min_raws[i] = bandwidth[i][j]

    server_rank = 0
    max_in_raws = 0
    for i in range(workers):
        if max_in_raws < min_raws[i]:
            server_rank = i
            max_in_raws = min_raws[i]

    if choose_clients_num is not None:
        pass
    else:
        choose_clients_num = workers / 2

    choose_list = np.array(range(workers))

    # at server rank the bandwidth is set as 0.
    choose_list = np.delete(choose_list, server_rank)

    np.random.shuffle(choose_list)
    chosen_clients = choose_list[:choose_clients_num]
    chosen_clients_bandwidth = bandwidth[server_rank][chosen_clients]
    bandwidth_threshold = min(chosen_clients_bandwidth)

    return None, bandwidth_threshold


def random_bandwidth(workers):
    #bandwidth = np.random.randint(1, 5, size=[workers, workers])
    bandwidth = np.random.rand(workers, workers)*5
    #bandwidth = (bandwidth)*5
    for i in range(workers):
        for j in range(workers):
            if i < j :
                bandwidth[i][j] = bandwidth[j][i]
            elif i == j :
                bandwidth[i][j] = 0
    return bandwidth

def fixed_bandwidth():
    bandwidth_list = [[0, 1.3, 1.5, 1.2, 1.6, 1.6, 1.5, 1.6, 1.7, 1.4, 1.7, 1.5, 1.6, 1.5, 1.1],
                         [1.3, 0, 1.5, 1.2, 1.5, 1.5, 1.5, 1.6, 1.5, 1.2, 1.5, 1.5, 1.4, 1.6, 1.5],
                         [1.4, 1.3, 0, 1.3, 1.5, 1.6, 1.4, 1.7, 1.3, 1.6, 1.7, 1.4, 1.6, 1.4, 1.3],
                         [1.2, 1.3, 1.4, 0, 1.5, 1.4, 1.5, 1.5, 1.5, 1.2, 1.5, 1.6, 1.6, 1.6, 1.4],
                         [11.0, 2.2, 27.7, 6.8, 0, 82.5, 73.1, 82.2, 132.5, 49.1, 69.5, 84.8, 98.0, 57.4, 51.7],
                         [6.8, 1.1, 20.2, 4.7, 82.6, 0, 129.2, 269.2, 78.3, 73.3, 147.1, 50.3, 54.4, 37.0, 36.4],
                         [27.3, 1.1, 15.1, 21.8, 83.2, 184.8, 0, 331.2, 86.4, 76.8, 261.1, 62.4, 70.6, 42.3, 39.0],
                         [0.2, 13.9, 27.6, 14.8, 60.8, 195.3, 276.2, 0, 63.3, 75.4, 323.1, 50.3, 62.6, 39.8, 36.4],
                         [0.2, 16.9, 5.7, 1.1, 166.8, 83.9, 64.0, 61.6, 0, 40.7, 54.0, 80.4, 65.9, 39.1, 41.6],
                         [36.2, 27.4, 1.7, 22.0, 37.5, 48.6, 54.7, 50.0, 35.8, 0, 45.0, 33.5, 39.0, 22.5, 49.7],
                         [36.0, 0.6, 16.8, 21.1, 27.9, 115.1, 247.8, 317.4, 51.6, 47.5, 0, 48.1, 36.8, 24.4, 28.8],
                         [15.6, 28.6, 10.6, 8.1, 94.8, 45.4, 43.8, 46.3, 70.4, 27.0, 45.8, 0, 172.9, 39.4, 61.5],
                         [2.3, 3.9, 22.5, 5.7, 78.3, 45.6, 32.7, 34.5, 47.3, 23.2, 23.7, 134.5, 0, 31.2, 38.5],
                         [0.1, 15.1, 8.2, 15.4, 41.8, 32.7, 39.9, 37.9, 59.6, 25.0, 38.4, 38.2, 39.9, 0, 32.5],
                         [27.9, 17.2, 18.4, 24.1, 54.3, 36.6, 36.0, 39.2, 39.1, 69.7, 33.8, 59.7, 65.9, 30.7, 0]]
    bandwidth = np.array(bandwidth_list[0])
    for i in range(1, len(bandwidth_list)):
        # print(bandwidth_list[i])
        # print(len(bandwidth_list[i]))
        bandwidth = np.vstack((bandwidth, np.array(bandwidth_list[i])))

    #print(bandwidth)
    for i in range(len(bandwidth[0])):
        for j in range(len(bandwidth[0])):
            if (i != j):
                # capacity[i][j] = capacity[j][i] = 1
                if bandwidth[i][j] > bandwidth[j][i]:
                    bandwidth[i][j] = bandwidth[j][i]
                    # print(bandwidth[i][j])
                    # print(bandwidth[j][i])
                else:
                    bandwidth[j][i] = bandwidth[i][j]
            else:
                bandwidth[i][j] = 0
    bandwidth = bandwidth[0:14,0:14]
    return bandwidth
# =================================================================================================
class Vertex(object):
    def __init__(self, label=''):
        self.label = label
    def __repr__(self):
        return 'Vertex(%s)' % repr(self.label)
    __str__=__repr__  

class Edge(tuple):
    def __new__(cls, e1, e2):
        return tuple.__new__(cls, (e1,e2))
    def __repr__(self):
        return "Edge(%s, %s)" % (repr(self[0]), repr(self[1]))
    __str__ = __repr__

class Graph(dict):
    def __init__(self, vs=[], es=[]):
        for v in vs:
            self.add_vertex(v)
        for e in es:
            self.add_edge(e)

    def add_vertex(self,v):
        self[v] = {}

    def add_edge(self, e):
        #  this implemention is a undirected graph
        v, w = e
        self[v][w] = e
        self[w][v] = e

    def get_edge(self,v1, v2):
        try:
            return self[v1][v2]
        except:
            return None

    def remove_edge(self,e):
        v, w = e
        self[v].pop(w)
        self[w].pop(v)

    def vertices(self):
        return list(self.keys())

    def edges(self):
        es = set()             # use set() to avoid repeated edge
        for v1 in self.vertices():
            for v2 in self.vertices():
                es.add(self.get_edge(v2, v1))
        es.discard(None)       
        return list(es)
        """
        es = []
        for v in self.vertices():
            es.extend(self[v].values())
        es = list(set(es))
        return es
        """

    def out_vertices(self,v):
        return list(self[v].keys())

    def out_edges(self,v):
        return list(self[v].values())

    def add_all_edges(self,vs=None):
        if vs == None:
            vs = self.vertices()
        for v1 in vs:
            for v2 in vs:
                if v1 is v2 : continue 
                self.add_edge(Edge(v1,v2))

    def is_connect(self):
        # pass
        vs = self.vertices()    # acquire all vertices

        q, s = [], set()        # search queue, and label set
        q.append(vs[0])         # search from no.1 node
        while q:                # when q is not empty:
            v = q.pop(0)       
            s.add(v)
            for w in self.out_vertices(v):
                if w not in s:
                    q.append(w)
        if len(s)==len(vs):
            return True
        else:
            return False

class SAPS_gossip(object):
    def __init__(self, B_origin, B_thres, T_thres):
        self.n_workers = len(B_origin[0])
        self.T_thres = T_thres
        self.R_stamps = np.zeros([self.n_workers, self.n_workers])
        self.B_origin = B_origin
        self.B_high = np.zeros([self.n_workers, self.n_workers])
        #print('before self.B_high============\n', self.B_high)
        self.B_high[np.where(self.B_origin > B_thres)] = 1
        #print('after self.B_high============\n', self.B_high)

    def generate_match(self, t):
        if self.ifConnected(self.R_stamps, self.T_thres, t):
            B_unmatch = self.B_high
            #print("is connected")
        else:
            #print("not connected")
            B_unmatch = self.GetOverTimeMatrix(self.R_stamps, self.T_thres, t)

        # here, matching randomBand produces randomMatch
        B_unmatch = self.randomBand(B_unmatch)
        index_roles = np.arange(self.n_workers)
        np.random.shuffle(index_roles)
        roles = np.ones(self.n_workers)
        roles[index_roles[0:int(self.n_workers/2)]]=0
        match, _ = minmax_communication_match(B_unmatch, roles, 1, islimited=False)

        match2 = {}
        if (match==None) or len(match) != self.n_workers:
            B_unmatch2 = self.GetUnMatch(self.B_origin, match)
            B_unmatch2 = self.randomBand(B_unmatch2)
            # limited_bandwidth2 = get_limited_graph(B_unmatch2, 0.5)
            # _, match2 = limited_graph_max_match(limited_bandwidth2, 0.5, self.n_workers)
            index_roles = np.arange(self.n_workers)
            np.random.shuffle(index_roles)
            roles = np.ones(self.n_workers)
            roles[index_roles[0:int(self.n_workers/2)]]=0
            match2, _ = minmax_communication_match(B_unmatch2, roles, 1, islimited=False)
            # print("=================B_unmatch2, match2", B_unmatch2, match2)
            # print("=================len(second_match) : %d", len(match2))

        if match:
            match.update(match2)
        else:
            match = match2

        for node1, node2 in match.items(): # here the match is double directions, its length is n  
            self.R_stamps[node1][node2] = t

        real_bandwidth_threshold = 1000000000
        for i in range(self.n_workers):
            # self.communication_record[rank][match[host]] += 1
            if real_bandwidth_threshold > self.B_origin[i][match[i]]:
                real_bandwidth_threshold = self.B_origin[i][match[i]]

        return match, real_bandwidth_threshold

    def ifConnected(self, R, T_thres, t):
        vs = []
        es = []
        for i in range(self.n_workers):
            v = Vertex(str(i))
            vs.append(v)

        # Q = np.zeros([self.n_workers, self.n_workers])
        for i in range(self.n_workers):
            # for j in range(self.n_workers):
            for j in range(i):
                #print("!!!!", R[i][j],t, T_thres)
                if R[i][j] > t - T_thres:
                    # Q[i][j] = 1
                    e = Edge(vs[i], vs[j])
                    es.append(e)

        g = Graph(vs, es)

        # self.Q[np.where(R > t - T_thres)] = 1
        # print('Q', g)
        return g.is_connect()

    def GetOverTimeMatrix(self, R, T_thres, t):
        Q = np.zeros([self.n_workers, self.n_workers])
        for i in range(self.n_workers):
            # for j in range(self.n_workers):
            for j in range(i):
                if R[i][j] < t - T_thres:
                    Q[i][j] = 1
        return Q

    def GetUnMatch(self, B, match):
        E = np.ones([self.n_workers, self.n_workers])
        if match==None:
            return E
        for node in match.keys():
            for j in range(self.n_workers):
                E[node][j] = 0
        return E

    def randomBand(self, B_unmatch):
        B = np.random.rand(self.n_workers, self.n_workers)
        B[np.where(B_unmatch==0)]=0.0
        return B


if __name__ == '__main__':
    # M=0
    # capacity = [
    # [0,16,13,M,M,M],
    # [M,0,10,12,M,M],
    # [M,4,0,M,14,M],
    # [M,M,9,0,M,20],
    # [M,M,M,7,0,4],
    # [M,M,M,M,M,0]
    # ]
    workers = 32
    Bandwidth_Decrease_Factor = 0.1

    if workers == 14:
        bandwidth = fixed_bandwidth()    
    else:
        bandwidth = random_bandwidth(workers)
    print("================\n")
    print(bandwidth)
    #print(bandwidth.shape)
    index_roles = np.arange(workers)
    np.random.shuffle(index_roles)
    roles = np.ones(workers)
    roles[index_roles[0:int(workers/2)]]=0
    #print("bandwidth\n", bandwidth)
    print("roles", roles)
    active_match = {}
    # match, bandwidth_threshold = minmax_communication_match(bandwidth, roles, 0)
    # bandwidth = get_limited_graph(bandwidth, 0)
    print("bandwidth after\n", bandwidth)
    noask_match, noask_bandwidth_threshold = minmax_communication_match(bandwidth, roles, 1, islimited=False)

    # random_match = {}
    # random_bandwidth_threshold = 100
    # for i in range(int(workers/2)):
    #     random_match[index_roles[i]] = index_roles[i+int(workers/2)]
    #     random_match[index_roles[i+int(workers/2)]] = index_roles[i]
    #     if random_bandwidth_threshold > bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]:
    #         random_bandwidth_threshold = bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]
    # for i in range(workers):
    #     for j in range(workers):
    #         if match[i][j] == True:
    #             active_match[i] = j
    #             active_match[j] = i
    #             print('match (active :%d, passive: %d)' % (i, j))
    print("bandwidth:",bandwidth)
    bandwidth_weighted = copy.deepcopy(bandwidth)
    limited_bandwidth = get_limited_graph(bandwidth, 1.45)
    limited_bandwidth_weighted = copy.deepcopy(limited_bandwidth)
    print("limited_bandwidth:",limited_bandwidth)
    B_thres=4
    T_thres=10
    SAPS_gossip1 = SAPS_gossip(bandwidth, B_thres=B_thres, T_thres=T_thres)

    Avg_random = 0
    Avg_ring = 0
    Avg_FedAvg = 0
    Avg_preparation = 0
    Avg_SAPS = 0

    for iteration in range(300):
        weighted_match, weighted_bandwidth_threshold = minmax_communication_match(bandwidth_weighted, roles, 1, islimited=False)
        real_bandwidth_threshold = 1000000000
        for i in range(workers):
            bandwidth_weighted[i][weighted_match[i]] = bandwidth_weighted[i][weighted_match[i]] - \
                        bandwidth[i][weighted_match[i]] * Bandwidth_Decrease_Factor 
            # self.communication_record[rank][match[host]] += 1
            if real_bandwidth_threshold > bandwidth[i][weighted_match[i]]:
                real_bandwidth_threshold = bandwidth[i][weighted_match[i]]
        # random_match = {}
        # random_bandwidth_threshold = 100
        # np.random.shuffle(index_roles)
        # for i in range(int(workers/2)):
        #     random_match[index_roles[i]] = index_roles[i+int(workers/2)]
        #     random_match[index_roles[i+int(workers/2)]] = index_roles[i]
        #     if random_bandwidth_threshold > bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]:
        #         random_bandwidth_threshold = bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]
        _, random_bandwidth_threshold = get_random_match_and_bandwidth(bandwidth, workers)
        _, ring_bandwidth_threshold = get_ring_match_and_bandwidth(bandwidth, workers)
        _, FedAVG_bandwidth_threshold = get_FedAVG_match_and_bandwidth(bandwidth, workers)
        preparation_match, preparation_bandwidth_threshold = minmax_communication_match(limited_bandwidth_weighted, roles, 1)
        preparation_bandwidth_threshold = 1000000000
        # print(limited_bandwidth_weighted)
        for i in range(workers):
            limited_bandwidth_weighted[i][preparation_match[i]] = limited_bandwidth_weighted[i][preparation_match[i]] - \
                        limited_bandwidth[i][preparation_match[i]] * Bandwidth_Decrease_Factor 
            # self.communication_record[rank][match[host]] += 1
            if preparation_bandwidth_threshold > limited_bandwidth[i][preparation_match[i]]:
                preparation_bandwidth_threshold = limited_bandwidth[i][preparation_match[i]]

        SAPS_match, SAPS_bandwidth = SAPS_gossip1.generate_match(iteration)


        Avg_random += random_bandwidth_threshold
        Avg_ring += ring_bandwidth_threshold
        Avg_FedAvg += FedAVG_bandwidth_threshold
        Avg_preparation += preparation_bandwidth_threshold
        Avg_SAPS += SAPS_bandwidth

        # # print("active  match =\n", match)
        # # print("active  bandwidth_threshold", bandwidth_threshold)
        # #print("random  match =\n", random_match)
        # print("============================")
        # print("random  bandwidth_threshold", random_bandwidth_threshold)
        # print("ring  bandwidth_threshold", ring_bandwidth_threshold)
        # #print("noask  match =\n", noask_match)
        # # print("noask  bandwidth_threshold", noask_bandwidth_threshold)
        # #print("weighted  noask  match =\n", weighted_match)
        # # print("weighted  noask  bandwidth_threshold", real_bandwidth_threshold)
        # print("FedAVG  bandwidth_threshold", FedAVG_bandwidth_threshold)
        # #print("weighted  noask  match =\n", preparation_match)
        # print("preparation_match   bandwidth_threshold", preparation_bandwidth_threshold)
        # #print("SAPS_match : %s,    SAPS_bandwidth_threshold = %s \n"%( SAPS_match, SAPS_bandwidth))
        # print("SAPS_bandwidth_threshold = %s \n"%(SAPS_bandwidth))

    print("\n=========!!!!!!!!!!!!==========\n")
    print("Avg_random  bandwidth_threshold = ", Avg_random/iteration)
    print("Avg_ring  bandwidth_threshold = ", Avg_ring/iteration)
    #print("noask  match =\n", noask_match)
    # print("noask  bandwidth_threshold", noask_bandwidth_threshold)
    #print("weighted  noask  match =\n", weighted_match)
    # print("weighted  noask  bandwidth_threshold", real_bandwidth_threshold)
    print("Avg_FedAvg  bandwidth_threshold = ", Avg_FedAvg/iteration)
    #print("weighted  noask  match =\n", preparation_match)
    print("Avg_preparation   bandwidth_threshold = ", Avg_preparation/iteration)
    #print("SAPS_match : %s,    SAPS_bandwidth_threshold = %s \n"%( SAPS_match, SAPS_bandwidth))
    print("    B_thres=%f, T_thres=%f\n" % (B_thres, T_thres))
    print("Avg_SAPS bandwidth_threshold = %s "%(Avg_SAPS/iteration))

    # f = open('14workers_B_T_thres.txt', 'w')
    # for B_thres in [0.5, 1, 1.45, 5, 10]:
    #     for T_thres in [1, 2, 5, 10, 15]:
    #         SAPS_gossip1 = SAPS_gossip(bandwidth, B_thres=B_thres, T_thres=T_thres)
    #         Avg_SAPS = 0  
    #         for iteration in range(300):
    #             SAPS_match, SAPS_bandwidth = SAPS_gossip1.generate_match(iteration)
    #             Avg_SAPS += SAPS_bandwidth
    #         f.writelines("    B_thres=%f, T_thres=%f" % (B_thres, T_thres))
    #         f.writelines("Avg_SAPS bandwidth_threshold = %s \n"%(Avg_SAPS/iteration))
    #         print("    B_thres=%f, T_thres=%f" % (B_thres, T_thres))
    #         print("Avg_SAPS bandwidth_threshold = %s \n"%(Avg_SAPS/iteration))
    # f = open('32workers_B_T_thres.txt', 'w')
    # for B_thres in [0.5, 1, 2, 3, 4]:
    #     for T_thres in [1, 2, 5, 10, 15]:
    #         SAPS_gossip1 = SAPS_gossip(bandwidth, B_thres=B_thres, T_thres=T_thres)
    #         Avg_SAPS = 0  
    #         for iteration in range(300):
    #             SAPS_match, SAPS_bandwidth = SAPS_gossip1.generate_match(iteration)
    #             Avg_SAPS += SAPS_bandwidth
    #         f.writelines("    B_thres=%f, T_thres=%f" % (B_thres, T_thres))
    #         f.writelines("Avg_SAPS bandwidth_threshold = %s \n"%(Avg_SAPS/iteration))
    #         print("    B_thres=%f, T_thres=%f" % (B_thres, T_thres))
    #         print("Avg_SAPS bandwidth_threshold = %s \n"%(Avg_SAPS/iteration))

    # f.close()

if False and __name__ == '__main__':
    # M=0
    # capacity = [
    # [0,16,13,M,M,M],
    # [M,0,10,12,M,M],
    # [M,4,0,M,14,M],
    # [M,M,9,0,M,20],
    # [M,M,M,7,0,4],
    # [M,M,M,M,M,0]
    # ]
    workers = 14
    # bandwidth = np.random.randint(10, 100, size=[workers, workers])
    # for i in range(workers):
    #     for j in range(workers):
    #         if i < j :
    #             bandwidth[i][j] = bandwidth[j][i]
    #         elif i == j :
    #             bandwidth[i][j] = 0

    bandwidth_list = [[0, 1.3, 1.5, 1.2, 1.6, 1.6, 1.5, 1.6, 1.7, 1.4, 1.7, 1.5, 1.6, 1.5, 1.1],
                         [1.3, 0, 1.5, 1.2, 1.5, 1.5, 1.5, 1.6, 1.5, 1.2, 1.5, 1.5, 1.4, 1.6, 1.5],
                         [1.4, 1.3, 0, 1.3, 1.5, 1.6, 1.4, 1.7, 1.3, 1.6, 1.7, 1.4, 1.6, 1.4, 1.3],
                         [1.2, 1.3, 1.4, 0, 1.5, 1.4, 1.5, 1.5, 1.5, 1.2, 1.5, 1.6, 1.6, 1.6, 1.4],
                         [11.0, 2.2, 27.7, 6.8, 0, 82.5, 73.1, 82.2, 132.5, 49.1, 69.5, 84.8, 98.0, 57.4, 51.7],
                         [6.8, 1.1, 20.2, 4.7, 82.6, 0, 129.2, 269.2, 78.3, 73.3, 147.1, 50.3, 54.4, 37.0, 36.4],
                         [27.3, 1.1, 15.1, 21.8, 83.2, 184.8, 0, 331.2, 86.4, 76.8, 261.1, 62.4, 70.6, 42.3, 39.0],
                         [0.2, 13.9, 27.6, 14.8, 60.8, 195.3, 276.2, 0, 63.3, 75.4, 323.1, 50.3, 62.6, 39.8, 36.4],
                         [0.2, 16.9, 5.7, 1.1, 166.8, 83.9, 64.0, 61.6, 0, 40.7, 54.0, 80.4, 65.9, 39.1, 41.6],
                         [36.2, 27.4, 1.7, 22.0, 37.5, 48.6, 54.7, 50.0, 35.8, 0, 45.0, 33.5, 39.0, 22.5, 49.7],
                         [36.0, 0.6, 16.8, 21.1, 27.9, 115.1, 247.8, 317.4, 51.6, 47.5, 0, 48.1, 36.8, 24.4, 28.8],
                         [15.6, 28.6, 10.6, 8.1, 94.8, 45.4, 43.8, 46.3, 70.4, 27.0, 45.8, 0, 172.9, 39.4, 61.5],
                         [2.3, 3.9, 22.5, 5.7, 78.3, 45.6, 32.7, 34.5, 47.3, 23.2, 23.7, 134.5, 0, 31.2, 38.5],
                         [0.1, 15.1, 8.2, 15.4, 41.8, 32.7, 39.9, 37.9, 59.6, 25.0, 38.4, 38.2, 39.9, 0, 32.5],
                         [27.9, 17.2, 18.4, 24.1, 54.3, 36.6, 36.0, 39.2, 39.1, 69.7, 33.8, 59.7, 65.9, 30.7, 0]]
    bandwidth = np.array(bandwidth_list[0])
    for i in range(1, len(bandwidth_list)):
        # print(bandwidth_list[i])
        # print(len(bandwidth_list[i]))
        bandwidth = np.vstack((bandwidth, np.array(bandwidth_list[i])))

    print(bandwidth)
    print(bandwidth.shape)
    bandwidth = bandwidth[0:14,0:14]
    index_roles = np.arange(workers)
    np.random.shuffle(index_roles)
    roles = np.ones(workers)
    roles[index_roles[0:int(workers/2)]]=0
    print("bandwidth\n", bandwidth)
    print("roles", roles)
    active_match = {}
    match, bandwidth_threshold = minmax_communication_match(bandwidth, roles, 0)
    noask_match, noask_bandwidth_threshold = minmax_communication_match(bandwidth, roles, 1, islimited=False)
    random_match = {}
    random_bandwidth_threshold = 100
    for i in range(int(workers/2)):
        random_match[index_roles[i]] = index_roles[i+int(workers/2)]
        random_match[index_roles[i+int(workers/2)]] = index_roles[i]
        if random_bandwidth_threshold > bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]:
            random_bandwidth_threshold = bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]
    for i in range(workers):
        for j in range(workers):
            if match[i][j] == True:
                active_match[i] = j
                active_match[j] = i
                print('match (active :%d, passive: %d)' % (i, j))
    print("bandwidth:",bandwidth)
    bandwidth_weighted = copy.deepcopy(bandwidth)
    limited_bandwidth = get_limited_graph(bandwidth, 1.45)
    limited_bandwidth_weighted = copy.deepcopy(limited_bandwidth)
    print("limited_bandwidth:",limited_bandwidth)

    SAPS_gossip1 = SAPS_gossip(bandwidth, B_thres=3, T_thres=3)
    for iteration in range(50):
        weighted_match, weighted_bandwidth_threshold = minmax_communication_match(bandwidth_weighted, roles, 1, islimited=False)
        real_bandwidth_threshold = 1000000000
        for i in range(workers):
            bandwidth_weighted[i][weighted_match[i]] = bandwidth_weighted[i][weighted_match[i]] - \
                        bandwidth[i][weighted_match[i]] * utils.Bandwidth_Decrease_Factor 
            # self.communication_record[rank][match[host]] += 1
            if real_bandwidth_threshold > bandwidth[i][weighted_match[i]]:
                real_bandwidth_threshold = bandwidth[i][weighted_match[i]]
        random_match = {}
        random_bandwidth_threshold = 100
        np.random.shuffle(index_roles)
        for i in range(int(workers/2)):
            random_match[index_roles[i]] = index_roles[i+int(workers/2)]
            random_match[index_roles[i+int(workers/2)]] = index_roles[i]
            if random_bandwidth_threshold > bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]:
                random_bandwidth_threshold = bandwidth[index_roles[i]][index_roles[i+int(workers/2)]]

        preparation_match, preparation_bandwidth_threshold = minmax_communication_match(limited_bandwidth_weighted, roles, 1)
        preparation_bandwidth_threshold = 1000000000
        # print(limited_bandwidth_weighted)
        for i in range(workers):
            limited_bandwidth_weighted[i][preparation_match[i]] = limited_bandwidth_weighted[i][preparation_match[i]] - \
                        limited_bandwidth[i][preparation_match[i]] * utils.Bandwidth_Decrease_Factor 
            # self.communication_record[rank][match[host]] += 1
            if preparation_bandwidth_threshold > limited_bandwidth[i][preparation_match[i]]:
                preparation_bandwidth_threshold = limited_bandwidth[i][preparation_match[i]]

        SAPS_match, SAPS_bandwidth = SAPS_gossip1.generate_match(iteration)

        # print("active  match =\n", match)
        print("active  bandwidth_threshold", bandwidth_threshold)
        #print("random  match =\n", random_match)
        print("random  bandwidth_threshold", random_bandwidth_threshold)
        #print("noask  match =\n", noask_match)
        print("noask  bandwidth_threshold", noask_bandwidth_threshold)
        #print("weighted  noask  match =\n", weighted_match)
        print("weighted  noask  bandwidth_threshold", real_bandwidth_threshold)
        print("weighted  noask  match =\n", preparation_match)
        print("preparation_match   bandwidth_threshold",preparation_match, preparation_bandwidth_threshold)
        print("SAPS_match : %s,    SAPS_bandwidth_threshold = %s \n"%( SAPS_match, SAPS_bandwidth))

    # v = Vertex('v')
    # w = Vertex('w')
    # e = Edge(v,w)
    # vs = []
    # es = []
    # eiiiii = [set([0, 1]), set([0, 2]), set([1, 3]), set([1, 5]), set([4, 5]), set([6, 5]), set([7, 8]), set([7, 9]), set([9, 10]), set([11, 12]), set([12, 13]), set([6, 7]), set([10, 11])]
    # for i in range(workers):
    #     v = Vertex(str(i))
    #     vs.append(v)
    #     for j in range(i):
    #         # v = Vertex(str(i))
    #         # w = Vertex('w')
    #         if set([i, j]) in eiiiii:
    #             e = Edge(vs[j], v)
    #             es.append(e)
            
    # g = Graph(vs, es)

    # print(g)
    # for i, _ in g.items():
    #     print("========================\n")
    #     print(g[i])
    # print(g.is_connect())




















































