import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
#import cairosvg


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


def inf_diagram1(max_utils):
    """
    decide on what course of action will yeild the maximum utilty
    
    args: max_utils, list in the form [[u1, u2, ..], [u3, u4, ...], [u5, u6, ..], ...]
    where every sublist represents a course of action that can be taken

    returns: the sublist with the maximum utilty

    this is definition 2 of intention in the paper
    """
    def helper1(switch):
        if switch:
            hit1to5 = False
            if not hit1to5:
                kill1to5 = False
                if not kill1to5:
                    return max_utils[0]
        else:
            hit6 = False
            if not hit6:
                kill6 = False
                if not kill6:
                    return max_utils[1]
    options = [helper1(True), helper1(False)]
    max_util_list = []
    max_util = 0
    for option in options:
        if sum(option) > max_util:
            max_util = sum(option)
            max_util_list = option
    return max_util_list 

    
inf_diagram1([[1,1,1,1,1], [1]))
#main()