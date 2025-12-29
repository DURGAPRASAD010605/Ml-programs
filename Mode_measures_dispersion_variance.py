# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:22:12 2025

@author: LAB
"""

import statistics as stats
data=[12,15,12,18,20,15,18,15,20,25]
variance=stats.variance(data)
std_deviation=stats.stdev(data)
print("Data: ",data)
print("Variance: ",variance)
print("standard deviation: ", std_deviation)
