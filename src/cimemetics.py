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
# average partial_results across repetitions 
# quick evolution of the modularity
# dynamic end criterion finishes less than 1000 (if 500 as change)
# consider large clusters on improve_offspring


#used parameters
population_size=0
MAX_GENERATIONS=0      
Pm=0
init_iter_factor=0.2  # from the paper
nreps=20
dyn=False
#problem parameters
problem_name=''
generations=0
nochange=0
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
#(population_size,MAX_GENERATIONS,Pm,cmin)
default=(10,200,0.2,'default.txt')
dejong=(50,1000,0.001,'dejong.txt')             #best option

grefenstette=(30,1000,0.01,'grefenstette.txt')  # all options

grefenstettedyn=(30,1000,0.01,'grefenstettedyn.txt')  # dyn

grefenstette11=(30,2000,0.01,'grefenstette11.txt')        #MAX_GENERATIONS
grefenstette12=(30,4000,0.01,'grefenstette12.txt')
grefenstette13=(30,8000,0.01,'grefenstette13.txt')
grefenstette14=(30,16000,0.01,'grefenstette14.txt')
grefenstette15=(30,32000,0.01,'grefenstette15.txt')
grefenstette16=(30,64000,0.01,'grefenstette16.txt')
grefenstette17=(30,128000,0.01,'grefenstette17.txt')

grefenstette20=(10,1000,0.01,'grefenstette20.txt')        #population_size
grefenstette21=(20,1000,0.01,'grefenstette21.txt')        
grefenstette22=(50,1000,0.01,'grefenstette22.txt')        
grefenstette23=(100,1000,0.01,'grefenstette23.txt')        
grefenstette24=(150,1000,0.01,'grefenstette24.txt')
grefenstette25=(200,1000,0.01,'grefenstette25.txt')
grefenstette26=(300,1000,0.01,'grefenstette26.txt')

grefenstette30=(30,1000,0.0,'grefenstette30.txt')          #Pm
grefenstette31=(30,1000,0.001,'grefenstette31.txt')   
grefenstette32=(30,1000,0.05,'grefenstette32.txt')
grefenstette33=(30,1000,0.1,'grefenstette33.txt')
grefenstette34=(30,1000,0.2,'grefenstette34.txt')
grefenstette35=(30,1000,0.4,'grefenstette35.txt')
grefenstette36=(30,1000,0.6,'grefenstette36.txt')
grefenstette37=(30,1000,0.9,'grefenstette37.txt')

microga1=(5,100,0.04,'microga1.txt')
microga2=(5,100,0.02,'microga2.txt')

bestparams011=(300,1000,0.01,'bestparams011.txt')
bestparams021=(300,1000,0.60,'bestparams021.txt')
bestparams01=(300,8000,0.01,'bestparams01.txt')
bestparams02=(300,8000,0.60,'bestparams02.txt')

#for dolphins
#paramsGA=[grefenstette,grefenstettedyn,grefenstette13,grefenstette26,grefenstette33,grefenstette36,bestparams01,bestparams02]
#paramsGA=[bestparams011,bestparams021]
paramsGA=[bestparams02]#,bestparams01]

#paramsGA=[default,dejong,grefenstette,microga1,microga2,grefenstette30,grefenstette31,grefenstette32,grefenstette33,grefenstette34,grefenstette35,grefenstette36,grefenstette37,grefenstette20,grefenstette21,grefenstette22,grefenstette23,grefenstette24,grefenstette25,grefenstette26,grefenstette11,grefenstette12,grefenstette13,grefenstette14,grefenstette15,grefenstette16,grefenstette17]
#paramsGA=[grefenstette20,grefenstette21,grefenstette22,grefenstette23,grefenstette24,grefenstette25,grefenstette26]
#paramsGA=[grefenstette30,grefenstette31,grefenstette32,grefenstette33,grefenstette34,grefenstette35,grefenstette36,grefenstette37]
#paramsGA=[bestparams01,bestparams02]
#paramsGA=[grefenstette]
#paramsGA=[grefenstettedyn]
#paramsGA=[grefenstette11,grefenstette12,grefenstette13]
#paramsGA=[grefenstette14,grefenstette15,grefenstette16,grefenstette17]
#paramsGA=[grefenstette20,grefenstette21,grefenstette22,grefenstette23,grefenstette24,grefenstette25,grefenstette26]
#paramsGA=[grefenstette30,grefenstette31,grefenstette32,grefenstette33,grefenstette34,grefenstette35,grefenstette36,grefenstette37]
#paramsGA=[default,dejong,grefenstette,microga1,microga2]

def calc_graph_weight(graph):
    global graph_weight
    graph_weight=0
    for e in graph.edges_iter():
        w=get_weight(e)         # we keep it like that to use it in weighted networks in the future
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
    # for each edge, we store if the edge has both ends (nodes) in the same cluster
    d=dict()  
    t=element[0]
    for e in graph.edges_iter():
        d[e]=t[e[0]]==t[e[1]]
    return d

def ERI(do, dm):
    # count how many times the edges are not in the same cluster for the two partitions(do,dm)
    key_list=do.keys()
    d=0
    for key in key_list:
        d+=not(do[key]==dm[key])
        #print d, do[key], dm[key]
    return float(d)/float(nedges)
    
def ERI_selection(graph, population, offspring):
    # find the closest and furthest ERI-wise elements in the population
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

def NMI(graph, individual1,individual2):
    cm=confussion_matrix(individual1[1],individual2[1])
    return NMICM(graph, cm)

def NMICM(graph,cm):
    (r,c)=cm.shape
    cmr=numpy.zeros(shape=(r,),dtype=numpy.float)
    cmc=numpy.zeros(shape=(c,),dtype=numpy.float)
    n=float(graph.number_of_nodes())

    for i in xrange(r):
        for j in xrange(c):
            cmr[i]+=cm[i][j]
            cmc[j]+=cm[i][j]
           
    denominator=0
    
    for i in xrange(r):
        if (cmr[i]>0):
            denominator+=cmr[i]*numpy.log(cmr[i]/n)
        
    for j in xrange(c):
        if (cmc[j]>0):
            denominator+=cmc[j]*numpy.log(cmc[j]/n)

    numerator=0    
    for i in xrange(r):
        for j in xrange(c):
            if (cm[i][j]>0):
                numerator+=cm[i][j]*numpy.log(cm[i][j]*n/cmr[i]/cmc[j])
    
    return -2*numerator/denominator

def confussion_matrix(individual1, individual2):
    l1=len(individual1)
    l2=len(individual2)
    confussion_matrix=numpy.zeros(shape=(l1,l2,), dtype=numpy.float)
    i=0
    k1=individual1.keys()
    k2=individual2.keys()
    for c1 in k1:
        j=0
        e1=individual1[c1]
        for c2 in k2:
            e2=individual2[c2]
            confussion_matrix[i][j]=len(set(e1) & set(e2))
            j+=1
        i+=1
    return confussion_matrix

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
                candidate=(copy.deepcopy(offspring[0]),copy.deepcopy(offspring[1]))
                candidate[1][max_cluster].append(e)
                del candidate[1][cluster_id]
                candidate[0][e]=max_cluster
                candidate=modularity(graph,candidate[0],candidate[1])
                if (False or candidate[2]>offspring[2]):
                    offspring=candidate
                    global mutations
                    mutations+=1
    return offspring
                     

def update_best(solution):
    global best_q
    q=solution[2]  # modularity
    if q > best_q:
        best_q=q
        global best
        best=solution
        if (dyn):
            global nochange
            nochange=0

def update_population3(graph, population, offspring):
    if offspring not in population:
        update_population2(graph, population, offspring)    

def update_population2(graph, population, offspring):
    # eri vs d:
    # if d(Io,Ic) <dmin and Q(Io) >= Q(Ic), then Io replaces Ic in P;
    # otherwise,if Q(Io) >= Q(Iw)then Io replaces Iw in P.
    # the best will always survive
    (cm,c,wm,w)=ERI_selection(graph, population, offspring)
    if ((c<0.01) and (cm[2]<offspring[2])): # really close  
        population.remove(cm)
        population.append(offspring)
        global creplacements
        creplacements+=1
    elif (wm[2]<offspring[2]):
        population.remove(wm)
        population.append(offspring)
        global wreplacements
        wreplacements+=1

def update_population1(graph, population, offspring):
    if offspring not in population:
        # returns the closest and furthest elements and respective ERI's
        (cm,c,wm,w)=ERI_selection(graph, population, offspring)
        if (cm[2]<offspring[2]):  
            population.remove(cm)
            population.append(offspring)
            global creplacements
            creplacements+=1

def end_criterion():
    global generations
    generations+=1
    global nochange
    nochange+=1
    if (dyn):
        return nochange>MAX_GENERATIONS
    else:
        return generations>MAX_GENERATIONS

def calculate_total_q(population):
    total_q=0.0
    for p in population:
        total_q+=p[2]
    return total_q

def run(graph):
    (population,total_q)=initialize_population(graph)
    #partial_results(graph, population)
    while (not end_criterion()):
        (p1,p2)=choose_parents(population, total_q)
        offspring=combine_parents(graph,p1,p2)
        offspring=modularity(graph,offspring[0],offspring[1])  # remove later
        offspring=improve_offspring(graph,offspring)
        offspring=modularity(graph,offspring[0],offspring[1])
        update_best(offspring)    
        total_q=update_population(graph,population,offspring)
        #partial_results(graph, population)
        #show_population(population)
    pass

def partial_results(graph, population):
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
    my_str='{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11}'.format(generations,min_mod,max_mod,range_mod,min_eri,max_eri,range_eri,min_k,max_k,mutations,creplacements,wreplacements)
    print my_str
    f.write(my_str+'\n')
    
def results(graph):
    d_best=ERI_member(graph, best)
    d_BKR=ERI_member(graph, (rereverse(gt),gt))
    eri=ERI(d_best,d_BKR)
    nmi=NMI(graph,best,(None,gt))
    results='BESTMOD;{0};;GENS:{5};;;ERI;{1};;MUTATIONS;{2};CREPL;{3};WREPL;{4};NMI;{6}'.format(best_q,eri,mutations,creplacements,wreplacements,generations,nmi)
    global results_num
    show(results)
    results_best_q.append(best_q)
    results_eri.append(eri)
    results_nmi.append(nmi)

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
    global results_nmi
    results_nmi=list()

def closefile():
    numpy_best_q=numpy.array(results_best_q)
    bestModMean=numpy.mean(numpy_best_q)
    bestModStdev=numpy.std(numpy_best_q)
    
    numpy_eri=numpy.array(results_eri)
    eriMean=numpy.mean(numpy_eri)
    eriStdev=numpy.std(numpy_eri)
    zeros=nreps-numpy.count_nonzero(numpy_eri)

    numpy_nmi=numpy.array(results_nmi)
    nmiMean=numpy.mean(numpy_nmi)
    nmiStdev=numpy.std(numpy_eri)
    
    results='ROUND;ZEROS:{0};MEAN:{1};STDEV;{2};ERI;MEAN:{3};STDEV;{4};NMI;MEAN:{5};STDEV;{6};'.format(zeros,bestModMean,bestModStdev,eriMean,eriStdev,nmiMean,nmiStdev)
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
    my_str='generation;min_mod;max_mod;range_mod;min_eri;max_eri;range_eri;min_k;max_k;mutations;crepl;wrepl;'
    show(my_str)

def problem(problem_name):
    graph=loadData(problem_name)                # load a graph
    for settings in xrange(len(paramsGA)):
        fileresults(paramsGA[settings])         # create the results files
        init_paramsGA(paramsGA[settings])       # initialize the GA parameters
        init(graph)                             # load the graph for teh problem
        for i in xrange(0,nreps):               # repeat execution nreps times
            init_exec()                         # init counters and statistics for the exec
            run(graph)                          # run the execution
            results(graph)                      # calculate final results
        closefile()                             # dump results to file

def update_population(graph,population, offspring):
    update_population1(graph,population, offspring)
    total_q=calculate_total_q(population)
    return total_q

def main():
    #problem(dolphins)
    problem(karate_club)
             
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
