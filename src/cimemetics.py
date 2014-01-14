import networkx as nx
import random
import copy
import pyparsing
import numpy

# list of graphs used in the experiment
karate_club="karate club"
#modularity=0.4198
ksgt2={ 1: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21],
       2: [8, 9, 15, 14, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]}
ksgt={ 1: [0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21],
       2: [4, 5, 6, 10, 16],
       3: [8, 9, 15, 14, 18, 20, 22, 26, 29, 30, 32, 33],
       4: [23, 24, 25, 27, 28, 31]}
dolphins="dolphins"
# modularity=0.5285
dgt2={1:[1,5,6,7,9,13,17,19,22,25,26,27,31,32,41,48,54,56,57,60],
      2:[0,2,3,4,8,10,11,12,14,15,16,18,20,21,23,24,28,29,30,33,34,35,36,37,38,39,40,42,43,44,45,46,47,49,50,51,52,53,55,58,59,61]
     }

dgt={1: [1, 5, 6, 7, 9, 13, 17, 19, 22, 25, 26, 27, 31, 32, 41, 48, 54, 56, 57, 60],
     2: [12, 14, 16, 33, 34, 37, 38, 40, 43, 46, 49, 50, 52, 53, 58, 61],
     3: [4, 11, 15, 18, 21, 23, 24, 29, 35, 45, 51, 55],
     4: [0, 2, 10, 20, 28, 30, 42, 44, 47],
     5: [3, 8, 36, 39, 59]}

dgt3={1: [1, 5, 6, 7, 9, 13, 17, 19, 22, 25, 26, 27, 31, 32, 41, 48, 54, 56, 57, 60],
     2: [12, 4, 16, 33, 34, 37, 38, 40, 43, 46, 49, 50, 52, 53, 58, 61],
     3: [14, 11, 15, 18, 21, 23, 24, 29, 35, 45, 51, 55],
     4: [0, 2, 10, 20, 28, 30, 42, 44, 47],
     5: [3, 8, 36, 39, 59]}

# TODO's
# max_intersect>1 would cause that splitted nodes are never added to a community if they only have one edge (leaves)
# cmin=10 to 0.01 --> makes the results much worse
# NMI
# max_eri: too high (76)
# consider large clusters on improve_offspring


#used parameters
population_size=0
MAX_GENERATIONS=0      
Pm=0
init_iter_factor=0.2  # from the paper
nreps=20
#problem parameters
problem_name=''
generations=0
mutations=0
creplacements=0
wreplacements=0
init_iter=0         # number of iterations in initialization (n_nodes*init_iter_factor)
f=0
ln=0
graph_weight=0
gt=dict()
gtmod=0.0
nedges=0

#parameters of the GA
#dejong parameters
#(population_size,MAX_GENERATIONS,Pm,cmin)
default=(10,200,0.2,'default.txt')
dejong=(50,1000,0.001,'dejong.txt')      #best option
dejong11=(50,2000,0.001,'dejong11.txt')  #MAX_GENERATIONS
dejong12=(50,3000,0.001,'dejong12.txt')
dejong21=(100,1000,0.001,'dejong21.txt') #population_size
dejong22=(150,1000,0.001,'dejong22.txt')
dejong23=(200,1000,0.001,'dejong23.txt')
dejong24=(300,1000,0.001,'dejong24.txt')
dejong31=(50,1000,0.01,'dejong31.txt')   #Pm
dejong32=(50,1000,0.05,'dejong32.txt')
dejong33=(50,1000,0.1,'dejong32.txt')
dejong34=(50,1000,0.9,'dejong34.txt')
dejong35=(50,1000,0.0,'dejong35.txt')
grefenstette=(30,1000,0.01,'grefenstette.txt') # all options
microga1=(5,100,0.04,'microga1.txt')
microga2=(5,100,0.02,'microga2.txt')

#paramsGA=[dejong,dejong01,dejong02,dejong03,dejong04,dejong05]
#paramsGA=[dejong06,dejong07,dejong08]
#paramsGA=[dejong11,dejong12]
#paramsGA=[dejong21,dejong22,dejong33,dejong24]
#paramsGA=[dejong31,dejong32,dejong33,dejong34,dejong35]
paramsGA=[default,dejong,grefenstette,microga1,microga2]


def calc_graph_weight(graph):
    global graph_weight
    graph_weight=0
    for e in graph.edges_iter():
        w=get_weight(e)
        graph_weight+=w
    graph_weight*=2

def rereverse(element):
    reversedict=dict()
    for key in element.keys():
        values=element[key]
        for v in values:
            reversedict[v]=key
    return reversedict

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

def modularity_cluster(graph, nodes):
    edges=graph.edges_iter(nodes, data=True)
    d=0.0
    c=0.0
    for edge in edges:
        w=get_weight(edge)
        d+=w
        if internal_edge(nodes, edge): ## the edge belongs to the cluster
            c+=w
            d+=w
    c*=2 # all edges must be counted twice
    qp=(c/graph_weight)-pow(d/graph_weight,2)
    return qp

def modularity(graph, offspring_dir, offspring_rev):
    (q,mod)=modularity_priv(graph, offspring_rev)
    return (offspring_dir,offspring_rev,q,mod)
    
def modularity_priv(graph, reversedict): # fitness function
    q=0.0
    mod=dict();
    for cluster_id in reversedict.keys():
        nodes=reversedict[cluster_id]
        qp=modularity_cluster(graph,nodes)
        q+=qp                       # total modularity
        mod[cluster_id]=qp          # modularity of the cluster              
    return (q+0.5,mod)  #+0.5 to make it positive

def ERI_member(graph, element):
    d=dict()
    t=element[0]
    for e in graph.edges_iter():
        d[e]=t[e[0]]==t[e[1]]
    return d

def ERI(do, dm):
    key_list=do.keys()
    d=0
    for key in key_list:
        d+=not(do[key]==dm[key])
        #print d, do[key], dm[key]
    return float(d)/float(nedges)
    
def ERI_selection(graph, population, offspring):
    c=2*graph.number_of_edges() #init to double max
    w=-1                        #init to minimum
    cm=tuple()
    wm=tuple()
    do=ERI_member(graph, offspring)
    for member in population:
        dm=ERI_member(graph, member)
        d=ERI(do,dm)
        if d>w:
            w=d
            wm=member
        if d<c:
            c=d
            cm=member
    return (cm,c,wm,w)

def NMI():
    pass

def loadData(nameGraph):
    global gt
    global problem_name
    problem_name=nameGraph
    if (nameGraph==karate_club):
        graph=nx.karate_club_graph()  #standard graph in networkx
        gt=ksgt
    elif (nameGraph==dolphins):
        graph=nx.read_gml('..\data\dolphins\dolphins.gml')
        gt=dgt
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
        solution=modularity(graph, element, rev_element)
        update_best(solution)
        total_q+=solution[2]
        population.append(solution)
    return (population,total_q)

def choose_parent(population,total_q):
    r=random.random()*total_q
    q_acc=0.0
    for p in population:
        q_acc+=p[2]
        if (q_acc>=r):
            population.remove(p)
            return p
    return None

def choose_parents(population, total_q):
    p1=choose_parent(population,total_q)  
    p2=choose_parent(population,total_q-p1[2])
    # we do not verify if the two elements are the same
    population.append(p2)
    population.append(p1)
    return (p1,p2)

def combine_lists(p1,p2):
    k1=p1.keys()
    l1=zip(p1.values(),k1,[0]*len(k1))
    k2=p2.keys()
    l2=zip(p2.values(),k2,[1]*len(k2))
    li=sorted(l1+l2, reverse=True)
    return li

def show_population(population):
    for sol in population:
        print sol[2]
    
def combine_parents(graph, p1, p2):
    #init offspring
    offspring=(dict(),dict())
    #aux structures
    p=[p1, p2]
    d=[copy.deepcopy(p1[1]),copy.deepcopy(p2[1])]
    li=combine_lists(p1[3],p2[3])
    cluster_nr=1                    #cluster number
    # get the clusters 1 by 1
    for c in li:
        nodes=d[c[2]][c[1]]
        if (len(nodes)>0):
            update_parent(p,d,c,nodes)
            update_offspring(graph,offspring,nodes,cluster_nr)
            cluster_nr+=1    
    return offspring  

def update_parent(p,d,c,nodes):
    wl=c[2]
    wln=1-wl
    index=p[wln][0] # n->c dictionary
    di=d[wln]
    for n in nodes:
        cluster=index[n]
        di[cluster].remove(n)

def update_offspring(graph,offspring,nodes,cluster_nr):
    offspring[1][cluster_nr]=list()
    for n in nodes:
        offspring[0][n]=cluster_nr
        offspring[1][cluster_nr].append(n)
        #print 'cluster', cluster_nr, n

def improve_offspring(graph, offspring):
    # take all lonely nodes
    # check if the majority of their edges
    # are in one of the clusters
    # what to do with large clusters???
    keys=offspring[1].keys()
    keys_shuffle=list(offspring[1].keys())
    random.shuffle(keys_shuffle)
    for cluster_id in keys:
        nodes=offspring[1][cluster_id]
        if ( len(nodes)==1 and Pm<random.random() ): # spare nodes
            max_intersect=-1
            max_cluster=-1
            e=nodes[0]
            nghbrs=graph.neighbors(e)
            nghbrs_set=set(nghbrs)
            for c in keys_shuffle:   # shuffle to avoid determinism and having to introduce a random
                if c<>cluster_id and c in offspring[1]:
                    n=offspring[1][c]
                    if (len(n)>1):
                        intersect=[val for val in n if val in nghbrs_set]
                        len_intersect=len(intersect)
                        if len_intersect>max_intersect:
                            max_intersect=len_intersect
                            max_cluster=c
            # change to this c-luster
            if (max_intersect>-1):
                global mutations
                mutations+=1
                offspring[1][max_cluster].append(e)
                del offspring[1][cluster_id]
                offspring[0][e]=max_cluster

def update_best(solution):
    global best_q
    q=solution[2]
    if q > best_q:
        best_q=q
        global best
        best=solution

def update_population(graph, population, offspring):
    # eri vs d:
    # if d(Io,Ic) <dmin and Q(Io) >= Q(Ic), then Io replaces Ic in P;
    # otherwise,if Q(Io) >= Q(Iw)then Io replaces Iw in P.
    # the best will always survive
    if offspring not in population:
        (cm,c,wm,w)=ERI_selection(graph, population, offspring)
        if (True):#(c<cmin):
            if (cm[2]<offspring[2]):  
                population.remove(cm)
                population.append(offspring)
                global creplacements
                creplacements+=1
        elif (wm[2]<offspring[2]):
            population.remove(wm)
            population.append(offspring)
            global wreplacements
            wreplacements+=1
    total_q=calculate_total_q(population)
    return total_q

def end_criterion():
    global generations
    generations+=1
    return generations>MAX_GENERATIONS

def calculate_total_q(population):
    total_q=0.0
    for p in population:
        total_q+=p[2]
    return total_q

def run(graph):
    (population,total_q)=initialize_population(graph)
    #print total_q, best_q
    while (not end_criterion()):
        (p1,p2)=choose_parents(population, total_q)
        offspring=combine_parents(graph,p1,p2)
        improve_offspring(graph,offspring)
        offspring=modularity(graph,offspring[0],offspring[1])
        update_best(offspring)    
        total_q=update_population(graph,population,offspring)
        #partial_results(graph, offspring, population)
        #show_population(population)
    pass

def partial_results(graph, offspring, population):
    min_mod=2
    max_mod=-1
    min_eri=nedges*4
    max_eri=-1
    min_k=1000
    max_k=-1
    l=len(population)
    eri_list=list() 
    for i in xrange(0,l):
        member=population[i]
        d_member=ERI_member(graph, member)
        mod=member[2]
        k=len(member[1])
        if mod<min_mod:
            min_mod=mod
        if mod>max_mod:
            max_mod=mod
        if k<min_k:
            min_k=k
        if k>max_k:
            max_k=k
        for d in eri_list:
            eri=ERI(d_member,d)
            if (eri<min_eri):
                min_eri=eri
            if (eri>max_eri):
                max_eri=eri       
        eri_list.append(d_member)    
    range_mod=max_mod-min_mod
    range_eri=max_eri-min_eri
    # print generations, 'offspring', offspring[2], 'best', best_q
    # n_gen min_mod best_mod /best_mod-min_mod/ eri_c eri_w /eri_c-eri_w/
    my_str='{0};{1};{2};{3};{4};{5};{6};{7};{8}'.format(generations,min_mod,max_mod,range_mod,min_eri,max_eri,range_eri,min_k,max_k)
    print my_str
    f.write(my_str+'\n')
    
def results(graph):
    d_best=ERI_member(graph, best)
    d_BKR=ERI_member(graph, (rereverse(gt),gt))
    eri=ERI(d_best,d_BKR)
    # eri dgt, dgt3 = 11/159 = 0.06918
    # eri dgt, dgt2 = 0.2012
    #d_BKR2=ERI_member(graph, (rereverse(dgt2), dgt2))
    #eri2=ERI(d_BKR,d_BKR2)
    results='BESTMOD;{0};;ERI;{1};;MUTATIONS;{2};CREPL;{3};WREPL;{4}'.format(best_q,eri,mutations,creplacements,wreplacements)
    global results_num
    show(results)
    results_best_q.append(best_q)
    results_eri.append(eri)

def show(mystr):
    print mystr
    f.write(mystr+'\n')

def fileresults(paramsGA):
    global f
    filename=problem_name+(paramsGA[3])
    f=open(filename,'w')
    show(filename)
    global results_best_q
    results_best_q=list()
    global results_eri
    results_eri=list()

def closefile():
    numpy_best_q=numpy.array(results_best_q)
    bestModMean=numpy.mean(numpy_best_q)
    bestModStdev=numpy.std(numpy_best_q)
    numpy_eri=numpy.array(results_eri)
    eriMean=numpy.mean(numpy_eri)
    eriStdev=numpy.std(numpy_eri)
    zeros=nreps-numpy.count_nonzero(numpy_eri)
    results='BESTMOD;ZEROS:{0};MEAN:{1};STDEV;{2};ERI;MEAN:{3};STDEV;{4}'.format(zeros,bestModMean,bestModStdev,eriMean,eriStdev)
    show(results)
    f.close()

def init_paramsGA(paramsGA):
    random.seed(0)                              # initialization for repeatibility
    global population_size
    population_size=paramsGA[0]
    global MAX_GENERATIONS
    MAX_GENERATIONS=paramsGA[1]
    global Pm
    Pm=paramsGA[2]
    params='SIZE;{0};;MAXGENS;{1};;Pm;{2};'.format(population_size, MAX_GENERATIONS, Pm)
    show(params)

def init_exec():
    global generations
    generations=0
    global mutations
    mutations=0
    global creplacements
    creplacements=0
    global wreplacements
    wreplacements=0
    global best
    best=dict()
    global best_q
    best_q=-10
    
def init(graph): 
    global ln
    ln=len(graph)
    global init_iter
    init_iter=int(ln*init_iter_factor)
    calc_graph_weight(graph)
    global gtmod
    (gtmod,b)=modularity_priv(graph,gt)
    global nedges
    nedges=graph.number_of_edges()
    mod_str='MAXMOD;{0};;NEDGES;{1}'.format(gtmod,nedges)
    show(mod_str)
    my_str='min_mod;max_mod;range_mod;min_eri;max_eri;range_eri;min_k;max_k;'
    show(my_str)

def problem(problem_name):
    
    graph=loadData(problem_name)                # load a graph
    for settings in xrange(len(paramsGA)):
        fileresults(paramsGA[settings])
        init_paramsGA(paramsGA[settings])
        init(graph)
        for i in xrange(0,nreps):
            init_exec()
            run(graph)
            results(graph)
        closefile()   

def main():
    
    problem(karate_club)
    #problem(dolphins)         
    

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
