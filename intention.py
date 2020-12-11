import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from dijkstra import DijkstraSPF, Graph
import numpy as np
#import cairosvg

#JOINT
def life_value(norm, alpha_norm, n, k):
    if norm == True:
        D_T = n*k*np.random.exponential()
    else:
        D_T = alpha_norm*k*np.random.exponential()
    return D_T

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

#print(life_value(True, 0.5, 0.5, 0.5))

# from scipy.stats import rv_discrete
# class life_value(rv_discrete):
#      "Life value distribution"
#      def _pmf(self, norm, alpha, alpha_norm, n, k):
#         if norm == True:
#             return n*k*np.random.exponential()
#         else:
#             return alpha_norm*k*np.random.exponential()
# #value = life_value(name="value")
# value= life_value()
# value.rvs(True, 0.5, 0.5, 0.5, 0.5)

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


#for the classic trolly problem (5 on main, 1 on side track)
inf_diagram5v1()



#graph!