import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
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
    # Add a decision node for drill
    # drill = gum.LabelizedVariable('Drill','Drill for oil',2)
    # drill.changeLabel(0,'Yes')
    # drill.changeLabel(1,'No')
    # model.addDecisionNode(drill)
    # # Add a chance node for result of test
    # result = gum.LabelizedVariable('Result','Result of test',4)
    # result.changeLabel(0,'NoS')
    # result.changeLabel(1,'OpS')
    # result.changeLabel(2,'ClS')
    # result.changeLabel(3,'NoR')
    # model.addChanceNode(result)
    # # Add a chance node for oil amount
    # amount = gum.LabelizedVariable('Amount','Oil amount',3)
    # amount.changeLabel(0,'Dry')
    # amount.changeLabel(1,'Wet')
    # amount.changeLabel(2,'Soak')
    # model.addChanceNode(amount)
    # Add an utility node for testing
    ut1to5 = gum.LabelizedVariable('Util1to5','Utility of 1 to 5',1)
    model.addUtilityNode(ut1to5)
    # Add an utility node for drilling
    ut6 = gum.LabelizedVariable('Util6','Utility of 6',1)
    model.addUtilityNode(ut6)
    # Add connections between nodes
    model.addArc(model.idFromName('Throw A'), model.idFromName('Util1to5'))
    model.addArc(model.idFromName('Throw A'), model.idFromName('Util6'))
    # model.addArc(model.idFromName('Test'), model.idFromName('UtilityOfTest'))
    # model.addArc(model.idFromName('Test'), model.idFromName('Drill'))
    # model.addArc(model.idFromName('Amount'), model.idFromName('Result'))
    # model.addArc(model.idFromName('Amount'), model.idFromName('UtilityOfDrill'))
    # model.addArc(model.idFromName('Result'), model.idFromName('Drill'))
    # model.addArc(model.idFromName('Drill'), model.idFromName('UtilityOfDrill'))

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


inf_diagram1()
