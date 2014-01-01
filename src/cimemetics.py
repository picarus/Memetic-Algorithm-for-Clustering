import networkx as nx
import random

# list of graphs used in the experiment
karate_club="karate club"
dolphins="dolphins"

# TODO's
# review modularity formula
# modularity of the ground truth
# choose_parents--> fix to consider negative modularity

#parameters of the GA
population_size=10
init_iter_factor=0.2  # from the paper
init_iter=0
generations=0
MAX_GENERATIONS=1
#problem parameters
ln=0
graph_weight=0
best_q=-10.0
best=dict()

def calc_graph_weight(graph):
    global graph_weight
    graph_weight=0
    for e in graph.edges_iter():
        w=get_weight(e)
        graph_weight+=w
    graph_weight*=2

def reverse(element):
    reversedict = dict()
    for key, value in element.iteritems():
        if not reversedict.has_key(value):
            reversedict[value]=list()    
        reversedict[value].append(key)
    return reversedict

def get_weight(edge):
    w=1
    return w

def internal_edge(nodes, edge):
    if not edge[0] in nodes:
        return False
    return edge[1] in nodes

def modularity(graph, reversedict): # fitness function
    q=0.0
    for cluster_id in reversedict.keys():
        nodes=reversedict[cluster_id]
        edges=graph.edges_iter(nodes, data=True)
        d=0.0
        c=0.0
        for edge in edges:
            w=get_weight(edge)
            d+=w
            if internal_edge(nodes, edge): ## the edge belongs to the cluster
                c+=w   
        #c*=2 # all edges must be counted twice
        #d*=2
        q+=((c/graph_weight)-pow(d/graph_weight,2))
        #print cluster_id, nodes, c, q
    return q

def ERI():
    pass

def NMI():
    pass

def loadData(nameGraph):
    if (nameGraph==karate_club):
        graph=nx.karate_club_graph()
    elif (nameGraph==dolphins):
        graph=null
    return graph

def merge(element, graph):
    # randomly, nodes are put together
    sample=random.sample(element.keys(), init_iter)
    for i in sample:
        #find the cluster of the element identifier
        cluster_id=element[i]
        nghbrs=graph.neighbors(i)
        for n in nghbrs:
            element[n]=cluster_id

def create_initial_element(graph):
    element=dict();
    # each vertex is a community
    for node in graph:
        element[node]=node
    # they are mixed together
    merge(element, graph)
    return element;

def initialize_population(graph):
    # every individual in the population will be a:
    # dict, where:
    #   the first element is the node id
    #   the second element is the cluster id
    #   initially they are equal
    # [ {0:0}, {1:1}, {2:2}, {3:3} ]
    total_q=0
    population=list();
    for i in xrange(0,population_size):
        element=create_initial_element(graph)
        rev_element=reverse(element)
        q=modularity(graph,rev_element)
        print i, q, rev_element
        solution=(element,rev_element,q)
        update_best(solution, q)
        total_q+=q
        population.append(solution)
    return (population,total_q)

def choose_parent(population,total_q):
    r=random.random()*total_q
    q_acc=0.0
    for p in population:
        q_acc+=p[2]
        if (q_acc>=r):
            return p
    return None

def choose_parents(population, total_q):
    # the modularity can be negative!!!!
    
    p1=choose_parent(population,total_q)
    p2=p1
    while (p1==p2):
        p2=choose_parent(population,total_q)
    return (p1,p2)

def combine_parents(graph, p1, p2):
    return p1

def improve_offspring(graph, offspring):
    pass

def update_best(solution, q):
    global best_q
    if q > best_q:
        best_q=q
        global best
        best=solution
    pass

def update_population(population, off_pool):
    # the best will always survive
    pass

def end_criterion():
    global generations
    global MAX_GENERATIONS
    generations+=1
    return generations<MAX_GENERATIONS
    pass

def run(graph):
    (population,total_q)=initialize_population(graph)
    while (not end_criterion()):
        off_pool=list()
        for i in xrange(int(population_size/2)+1):
            (p1,p2)=choose_parents(population, total_q)
            print p1[2], p2[2]
            off=combine_parents(graph,p1,p2)
            improve_offspring(graph,off)
            update_best(off,off[2])
            off_pool.append(off)
        total_q=update_population(population,off_pool)
    pass

def results():
    pass

def init(graph):
    random.seed(0)              # initialization for repeatibility
    global ln
    ln=len(graph)
    global init_iter
    init_iter=int(ln*init_iter_factor)
    calc_graph_weight(graph)

def main():
    graph=loadData(karate_club) # load a graph
    init(graph)
    run(graph)
    results()

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
