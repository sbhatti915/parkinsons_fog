#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 01:15:14 2023

@author: macbookpro

Conclusion: There are significantly more patients on medication than off
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from events_analysis import read_csv

def num_meds(dataframe):
    med = dataframe.loc[dataframe['Medication'] == 'on']
    print("The number of patients on medications are " + str(med.shape[0]) + 
          " people out of " + str(dataframe.shape[0]) + " patients.")

if __name__ == "__main__":
    
    defog_file = '/Users/Documents/kaggle/parkinsons_fog/data/defog_metadata.csv'
    tdcsfog_file = '/Users/Documents/kaggle/parkinsons_fog/data/tdcsfog_metadata.csv'
    
    defog_df = read_csv(defog_file)
    tdcsfog_df = read_csv(tdcsfog_file)
    
    num_meds(defog_df)
    num_meds(tdcsfog_df)
    
    
    
