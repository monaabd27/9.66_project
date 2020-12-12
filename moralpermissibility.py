#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:52:12 2020

@author: gredondo
"""

import math

def logit(x, alpha_1, alpha_2):
    insexp = alpha_1 * (x-alpha_2)
    logit_def = (1+math.exp(insexp))**(-1)
    return logit_def
    
#p_harm = P(I_harm = Yes|A) = probability that the agent is inferred to have
#intended to hard someone given that they took action A
def intention_permissibility(p_harm, alpha_1 =7, alpha_2=0.7):
    return logit((1-p_harm), alpha_1, alpha_2)

def utility_permissibility(lives_lost, lives_saved, alpha_1 =0.3, alpha_2=0):
    delta_lives = lives_lost - lives_saved
    return logit(delta_lives, alpha_1, alpha_2)

def full_permissibility(p_harm, lives_lost, lives_saved, w= 0.8):
    per_intention = intention_permissibility(p_harm)
    per_utility = utility_permissibility(lives_lost, lives_saved)
    per_full = w*per_intention + (1-w)*per_utility
    return per_full
    
    
print(full_permissibility(0.5, 0.5, 5, 8))
# print(full_permissibility(0.5, 0.5, 5, 11))
# print(full_permissibility(0.5, 0.5, 5, 15))



###stuff for graphing

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# p_ap_mean = [81, 86, 90, 90, 63, 72]

# s_ap_mean = [74, 73, 74, 73, 73, 67]

# c_ap_mean = [84, 94, 90, 96, 56, 85]

# # Set position of bar on X axis
# barWidth = 0.1
# r1 = np.arange(len(p_ap_mean))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]

# bands =  ["log", "log fitted", "rf", "rf fitted", "knn", "mlp"]
 
# # Make the plot
# plt.bar(r1, p_ap_mean, width=barWidth, edgecolor='white', label='Embeddings')
# plt.bar(r2, s_ap_mean, width=barWidth, edgecolor='white', label='Other Modalities')
# plt.bar(r3, c_ap_mean, width=barWidth, edgecolor='white', label='All Modalities')
 
# # Add xticks on the middle of the group bars
# #plt.ylim([0.5, 1])
# plt.ylabel('Model Type', fontweight='bold')
# plt.xlabel('% Accuracy', fontweight='bold')
# plt.title("Accuracy for Different Classifiers before and after embeddings")
# plt.xticks([r + 1/3*barWidth for r in range(len(p_ap_mean))], bands, rotation = 30)

# plt.legend()
# sns.despine()

# # plt.subplot(324)
# # plt.scatter(x, y, s=80, c=z, marker=(5, 1))
# plt.show()
# plt.savefig('all_groups_bands')