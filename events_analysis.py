#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:20:53 2023

@author: Sameer Bhatti

Conclusion: Really high times means they are off medication but really low
            times does not necessarily mean they are on medication
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path_to_csv):
    df = pd.read_csv(path_to_csv)
    return df

def plot_times(dataframe):
    times = list(dataframe['Completion'] - dataframe['Init'])
    
    plt.figure()
    plt.title(str(dataframe['Type'].values[0]))
    plt.ylabel('Time (s)')
    plt.xlabel('Occurrence')
    plt.plot(np.linspace(1, len(times), len(times)), times)
    plt.show()


if __name__ == "__main__":
    
    input_file = '/Users/Documents/kaggle/parkinsons_fog/data/events.csv'
    
    df = read_csv(input_file)
    
    sh = df.loc[df['Type'] == 'StartHesitation']
    w = df.loc[df['Type'] == 'Walking']
    t = df.loc[df['Type'] == 'Turn']
    
    plot_times(sh)
    plot_times(w)
    plot_times(t)
    