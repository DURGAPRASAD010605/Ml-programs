# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:22:10 2025

@author: LAB
"""

import statistics as stats
data=[12,15,12,18,20,15,18,15,20,25]
mean_value=stats.mean(data)
median_value=stats.median(data)
try:
    mode_value=stats.mode(data)
except stats.StatictisError:
    mode_value="no unique mode found"
print("Data: ",data)
print("Mean: ",mean_value)
print("Mode: ",mode_value)    