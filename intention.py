import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from dijkstra import DijkstraSPF, Graph
import numpy as np

#the way the norms impact is thru the utility
#for a given desire and norm we have our utility, 
# we can find the utility and best descision
#we dont know the desires and norms from the influence diagram
#the general influence diagram is known
#find the probability of intention given an influence diagram
#given an intention you know what actions that person should take
#find the posterior through rejection sampling
#this part just gives us the desires
#not directly related to the intention

#moral permissibility section is fitting the model to the human data
#we jsut use their parameters
#run mini experiments on ppl and then fit the model

#JOINT
def life_value(norm, k, n_T = 1, reg = True, alpha_bro = 30):
    """
    inputs: 
    norm (bool): if we are following the norm that loved ones
    are more valued, norm = True.
    
    alpha_bro: norm when family is seen more valuable. 
    the brother is see as equal to alpha_norm number of ppl

    n: norm when all people are seen as equal

    k = -1 if they want to kill the people on the track, 1 overwise
    
    Returns: D_T: utility of the people on the track not being killed
    """
    if (not norm or reg == True) and n_T != "B":
        D_T = n_T*k*np.random.exponential()
    else:
        D_T = alpha_bro*k*np.random.exponential()
    return D_T

def life_value_5(norm, k, n_T = 1, reg = True, alpha_bro = 1.5):
    """
    inputs: 
    norm (bool): if we are following the norm that loved ones
    are more valued, norm = True.
    
    alpha_bro: norm when family is seen more valuable. 
    the brother is see as equal to alpha_norm number of ppl

    n: norm when all people are seen as equal

    k = -1 if they want to kill the people on the track, 1 overwise
    
    Returns: D_T: utility of the people on the track not being killed
    """
    if (not norm or reg == True) and n_T != "B":
        D_T = n_T*k*np.random.exponential()
    else:
        D_T = 5*alpha_bro*k*np.random.exponential()
    return D_T

# def life_value_mixed(norm, k, n_T = 1, reg = True, alpha_doc = 10):
#     """
#     inputs: 
#     norm (bool): if we are following the norm that doctors
#     are more valued, norm = True.
    
#     alpha_doc: norm when doctors is seen more valuable. 
#     the doctor is seen as equal to alpha_norm number of ppl

#     n: norm when all people are seen as equal

#     k = -1 if they want to kill the people on the track, 1 overwise
    
#     Returns: D_T: utility of the people on the track not being killed
#     """
#     if (not norm or reg == True) and n_T != "D":
#         D_T = n_T*k*np.random.exponential()
#     else:
#         D_T = alpha_doc*k*np.random.exponential()
#     return D_T

#print(life_value(True, 2, 0.5, 1))

def sample_P_DN(a_k = 0.05, a_b = 0.1, a_norm = 0.55, n_M = 1, n_S = 1, reg=True):
    """
    a_b : prob that the agent wants to kill as many people as possible
    
    a_k: prob that they want to kill the people on the track, 
    independent for each track
    
    a_norm: prob that they follow the norm
    """
    samples = []
    for i in range(0, 100):
        a_b = np.random.uniform(0,100)
        if a_b < 10:
            k = -1
        else:
            a_k = np.random.uniform(0,100)
            if a_k < 5:
                k = -1
            else:
                k=1
        a_norm = np.random.uniform(0,100)
        if a_norm < 55:
            norm = True
        else:
            norm = False
        
        life_val = (life_value(norm, k, n_T = n_M, alpha_bro = 30, reg=reg), life_value(norm, k, n_T = n_S, alpha_bro = 30, reg=reg))
        m = max(life_val, key = abs)
        
        if k == -1:
            if m == life_val[0]:
                l = False
            else:
                l = True
        else:
            if m == life_val[0]:
                l = True
            else:
                l = False

        if n_M == "B":
            bro_loc = "main"
            samples.append((l, k, bro_loc, n_M, n_S))
        elif n_S == "B":
            bro_loc = "side"
            samples.append((l, k, bro_loc, n_M, n_S))
        else:
            samples.append((l, k))
        

    return samples

#utility of people not being killed on main track

def sample_P_norm(action_done, action_samples):
    num_norm = 0
    num_total_samps = 0
    for samp in action_samples:
        if samp[0] == action_done:
            num_total_samps += 1
            if samp[3] == 5:
                lives_saved  = 5
            if samp[3] == 2:
                lives_saved  = 2
            if samp[3] == 1:
                lives_saved  = 1
            if samp[3] == "B":
                lives_saved = 1
            if samp[4] == 5:
                lives_lost = 5
            if samp[4] == 2:
                lives_lost  = 2
            if samp[4] == 1:
                lives_lost  = 1
            if samp[4] == "B":
                lives_lost = 1 
            net = lives_saved - lives_lost
            if samp[1] == -1:
                if net > 0:
                    num_norm += 1
            if samp[1] == 1:
                if net < 0:
                    num_norm += 1
            if net == 0:
                if samp[1] == 1:
                    if samp[3] =="B":
                        num_norm += 1
                if samp[1] == -1:
                    if samp[4] =="B":
                        num_norm += 1
                
    return num_norm/num_total_samps

def sample_P_I(action_done, action_samples):   
    num_i_kill = 0
    num_total_samps = 0
    for samp in action_samples:
        if samp[0] == action_done:
            num_total_samps += 1
            if samp[1] == -1:
                num_i_kill += 1
    return num_i_kill/num_total_samps


for i in range(1, 11):
    print(sample_P_norm(True, sample_P_DN(a_k = 0.05, a_b = 0.1, a_norm = 0.55, n_M = "B", n_S= 5, reg=False)))
    #action true means that the lever was pulled, so side track ppl were killled


#INTENTION
def max_util(utils):
    """
    decide on what course of action will yeild the maximum utilty
    
    args: utils, list in the form [[u1, u2, ..], [u3, u4, ...], [u5, u6, ..], ...]
    where every sublist represents a course of action that can be taken

    returns: the sublist with the maximum utilty

    this is definition 1 of intention in the paper
    """
    return max(utils, key=sum)


def inf_diagram1():
    """
    decide on what course of action will yeild the maximum utilty
    
    args: max_utils, list in the form [[u1, u2, ..], [u3, u4, ...], [u5, u6, ..], ...]
    where every sublist represents a course of action that can be taken

    returns: the sublist with the maximum utilty

    this is definition 2 of intention in the paper
    """
    # Create a decision network
    model = gum.InfluenceDiagram()
    # Add a decision node for test
    throw = gum.LabelizedVariable('Throw A','Throw the switch',2)
    throw.changeLabel(0,'Yes')
    throw.changeLabel(1,'No')
    model.addDecisionNode(throw)
    ut1to5 = gum.LabelizedVariable('Util1to5','Utility of 1 to 5',1)
    model.addUtilityNode(ut1to5)
    # Add an utility node 
    ut6 = gum.LabelizedVariable('Util6','Utility of 6',1)
    model.addUtilityNode(ut6)
    # Add connections between nodes
    model.addArc(model.idFromName('Throw A'), model.idFromName('Util1to5'))
    model.addArc(model.idFromName('Throw A'), model.idFromName('Util6'))

    # Add utilities
    model.utility(model.idFromName('Util1to5'))[{'Throw A':'Yes'}]=5
    model.utility(model.idFromName('Util6'))[{'Throw A':'No'}]=1
    model.utility(model.idFromName('Util1to5'))[{'Throw A':'No'}]=0
    model.utility(model.idFromName('Util6'))[{'Throw A':'Yes'}]= 0

    ie = gum.InfluenceDiagramInference(model)
    # Make an inference with default evidence
    ie.makeInference()
    print('--- Inference with default evidence ---')

    print('Maximum Expected Utility (MEU) : {0}'.format(ie.getMEU()))



def inf_diagram5v1():
    #method using a graph, this is definition 2 of intention in the paper
    #classic trolley problem
    graph = Graph()
    start = "throw_A"
    end1 = "util1to5"
    end2 = "util6"
    choice = "end result"
    graph.add_edge(start, end1, -5)
    graph.add_edge(start, end2, -1)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram5v1_loop():
    #method using a graph, this is definition 2 of intention in the paper
    #classic trolley problem
    #this is the special case of the loop track
    graph = Graph()
    start = "throw_A"
    end1 = "util1to5"
    end2 = "util6"
    mid = "hurt6"
    choice = "end result"
    graph.add_edge(start, end1, -5)
    graph.add_edge(start, mid, -1)
    graph.add_edge(mid, end2, 0)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

def inf_diagram5vB():
    #method using a graph, this is definition 2 of intention in the paper
    #classic trolley problem, but with a brother instead of 1 person
    graph = Graph()
    start = "throw_A"
    end1 = "util1to5"
    end2 = "utilB"
    choice = "end result"
    graph.add_edge(start, end1, -5)
    graph.add_edge(start, end2, -10)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram2v1():
    #method using a graph, this is definition 2 of intention in the paper
    #classic trolley problem, but with 2 ppl
    graph = Graph()
    start = "throw_A"
    end1 = "util1to2"
    end2 = "util3"
    choice = "end result"
    graph.add_edge(start, end1, -2)
    graph.add_edge(start, end2, -1)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram2vB():
    #method using a graph, this is definition 2 of intention in the paper
    #classic trolley problem, but with 2 ppl and brother
    graph = Graph()
    start = "throw_A"
    end1 = "util1to2"
    end2 = "util3"
    choice = "end result"
    graph.add_edge(start, end1, -2)
    graph.add_edge(start, end2, -10)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

def inf_diagram1v1():
    #method using a graph, this is definition 2 of intention in the paper
    #classic trolley problem, but with 1 person
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2"
    choice = "end result"
    graph.add_edge(start, end1, -1)
    graph.add_edge(start, end2, -1)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

def inf_diagram1vB():
    #method using a graph, this is definition 2 of intention in the paper
    #classic trolley problem, but with 1 person and brother
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2"
    choice = "end result"
    graph.add_edge(start, end1, -1)
    graph.add_edge(start, end2, -10)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

def inf_diagramBv1():
    #method using a graph, this is definition 2 of intention in the paper
    #classic trolley problem, but with brother and 1 person
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2"
    choice = "end result"
    graph.add_edge(start, end1, -10)
    graph.add_edge(start, end2, -1)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

def inf_diagram1v2():
    #method using a graph, this is definition 2 of intention in the paper
    #1 person on main track, 2 ppl on side track
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2to3"
    choice = "end result"
    graph.add_edge(start, end1, -1)
    graph.add_edge(start, end2, -2)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

def inf_diagramBv2():
    #method using a graph, this is definition 2 of intention in the paper
    #brother on main track, 2 people on side track
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2to3"
    choice = "end result"
    graph.add_edge(start, end1, -10)
    graph.add_edge(start, end2, -2)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram1v5():
    #method using a graph, this is definition 2 of intention in the paper
    #1 person on main track, 5 people on side track
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2to6"
    choice = "end result"
    graph.add_edge(start, end1, -1)
    graph.add_edge(start, end2, -5)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

def inf_diagramBv5():
    #method using a graph, this is definition 2 of intention in the paper
    #brother on main track, 5 people on side track
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2to6"
    choice = "end result"
    graph.add_edge(start, end2, -10)
    graph.add_edge(start, end1, -5)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram1v5_Side():
    #method using a graph, this is definition 2 of intention in the paper
    #1 person on main track, 5 people on side track, and then the side-side track that saves everyone
    #this is presented in the paper, but not tested with subjects
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2to6"
    end3 = "Side Side"
    choice = "end result"
    graph.add_edge(start, end1, -1)
    graph.add_edge(start, end2, -5)
    graph.add_edge(end2, end3, -1)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end3, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

#NEW SCENARIOS

def inf_diagram1v5_Mixed():
    #method using a graph, this is definition 2 of intention in the paper
    #1 person on main track, 5 people on side track
    #th 1 person is a doctor, other 5 are murderers
    graph = Graph()
    start = "throw_A"
    end1 = "Doctor"
    end2 = "Murderers"
    choice = "end result"
    graph.add_edge(start, end1, -10)
    graph.add_edge(start, end2, -2.5)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

def inf_diagram5v1_Mixed():
    #method using a graph, this is definition 2 of intention in the paper
    #5  on main track, 1 on side track
    #th 1 person is a doctor, other 5 are murderers
    graph = Graph()
    start = "throw_A"
    end1 = "Murderers"
    end2 = "Doctor"
    choice = "end result"
    graph.add_edge(start, end2, -10)
    graph.add_edge(start, end1, -2.5)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram5v5_Mixed():
    #method using a graph, this is definition 2 of intention in the paper
    #5  on main track, 5 on side track
    #th 5 person is a doctor, other 5 are murderers, murderers on main traick
    graph = Graph()
    start = "throw_A"
    end1 = "Murderers"
    end2 = "Doctors"
    choice = "end result"
    graph.add_edge(start, end2, -10)
    graph.add_edge(start, end1, -2.5)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

def inf_diagram5v5_Mixed_Switch():
    #method using a graph, this is definition 2 of intention in the paper
    #5  on main track, 5 on side track
    #th 5 person is a doctor, other 5 are murderers, murderers on side track
    graph = Graph()
    start = "throw_A"
    end1 = "Doctors"
    end2 = "Murders"
    choice = "end result"
    graph.add_edge(start, end1, -10)
    graph.add_edge(start, end2, -2.5)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))

#for the classic trolly problem (5 on main, 1 on side track)
inf_diagram5v1()



#graph!