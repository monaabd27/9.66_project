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

print(life_value(True, 0.5, 0.5, 0.5))

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



def inf_diagram2(utils):
    """
    method using a graph

    args: max_utils, list in the form [[u1, u2, ..], [u3, u4, ...], [u5, u6, ..], ...]
    where every sublist represents a course of action that can be taken

    returns: the sublist with the maximum utilty

    this is definition 2 of intention in the paper
    """

    def create_graph(start, end1, end2, choice):
        # create the graph so that the throw node connects to one util node
        #but the utils are added for each group, and then negated
        #each negated sum will serve as a directed edge to our end node
        negated_sums = [sum(u)*-1 for u in utils]
        graph = Graph()
        graph.add_edge(start, end2, -1)
        graph.add_edge(start, end1, -5)
        graph.add_edge(end2, choice, 0)
        graph.add_edge(end1, choice, 0)

        
        #add an edge to the graph, where the first arg is start node
        #second arg is end node, thirs arg is in form (weight, name)
        #name is the ind of the weight in negated_sums
        return graph

    def cost_func(u, v, edge, prev_edge):
        length, name = edge
        if prev_edge:
            prev_name = prev_edge[1]
        else:
            prev_name = None
        cost = length
        # if name != prev_name:
        #     cost += 0
        return cost

    start = "throw_A"
    end1 = "util1to5"
    end2 = "util6"
    choice = "end result"
    nodes = [start, end1, end2, choice]
    graph = create_graph(start, end1, end2, choice)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(end1))
    print(dijkstra.get_path(end1))
    print(dijkstra.get_distance(end2))
    print(dijkstra.get_path(end2))
    print(dijkstra.get_distance(choice))
    print(dijkstra.get_path(choice))


inf_diagram2([[1,1,1,1,1],[1]])
# inf_diagram1()
