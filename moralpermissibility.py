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