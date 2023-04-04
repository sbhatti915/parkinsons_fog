#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 00:31:00 2023

@author: macbookpro

Analysis of time and accelerations of event with meds and no meds for defog
"""

import matplotlib.pyplot as plt
import numpy as np
from events_analysis import read_csv
import pandas as pd

def get_row(df_subjects, id_number):
    row = df_subjects.loc[df_subjects['Id'] == id_number]
    return row

def get_total_time(dataframe):
    times = dataframe['Completion'] - dataframe['Init']
    return times

def plot_times(dataframe, graph_title=None):
    
    plt.figure()
    plt.title(graph_title)
    plt.ylabel('Time (s)')
    plt.xlabel('Event Occurrence Count')
    plt.plot(np.linspace(1, len(dataframe), len(dataframe)), dataframe)
    plt.show()
    
    
if __name__ == "__main__":
    
    events_file_path = '/Users/Documents/kaggle/parkinsons_fog/data/events.csv'
    tdcsfog_file_path = '/Users/Documents/kaggle/parkinsons_fog/data/tdcsfog_metadata.csv'
    
    events_df = read_csv(events_file_path)
    tdcsfog_df = read_csv(tdcsfog_file_path)
    
    # Sort df based on subjects
    tdcsfog_subjects_df = tdcsfog_df.sort_values(by=['Subject'])
    
    tdcsfog_subjects = tdcsfog_subjects_df.Subject.unique()
    
    sh_events_med_on = pd.DataFrame()
    w_events_med_on = pd.DataFrame()
    t_events_med_on = pd.DataFrame()
    
    sh_events_med_off = pd.DataFrame()
    w_events_med_off = pd.DataFrame()
    t_events_med_off = pd.DataFrame()
    
    
    for subject_id in tdcsfog_subjects:
        # Visit Ids for a subject
        ids_list = tdcsfog_subjects_df.query("Subject == @subject_id")["Id"]
        
        for ID in ids_list:
            # Get 1 visit id of a subject
            id_row = get_row(tdcsfog_subjects_df, ID)
            
            if id_row['Medication'].values[0] == 'on':  
            
                # match visit id to visit id in events folder
                id_events_df = events_df.query("Id == @id_row['Id'].values[0]")
                
                sh_med_on = id_events_df.loc[id_events_df['Type'] == 'StartHesitation']
                w_med_on = id_events_df.loc[id_events_df['Type']  == 'Walking']
                t_med_on = id_events_df.loc[id_events_df['Type']  == 'Turn']
                
                if not sh_med_on.empty:
                    sh_events_med_on = pd.concat([sh_events_med_on, sh_med_on]) 

                if not w_med_on.empty:
                    w_events_med_on = pd.concat([w_events_med_on, w_med_on])
                    
                if not t_med_on.empty:
                    t_events_med_on = pd.concat([t_events_med_on, t_med_on])
                    
                
            if id_row['Medication'].values[0] == 'off':  
            
                # match visit id to visit id in events folder
                id_events_df = events_df.query("Id == @id_row['Id'].values[0]")
                
                sh_med_off = id_events_df.loc[id_events_df['Type'] == 'StartHesitation']
                w_med_off = id_events_df.loc[id_events_df['Type']  == 'Walking']
                t_med_off = id_events_df.loc[id_events_df['Type']  == 'Turn']
                
                if not sh_med_off.empty:
                    sh_events_med_off = pd.concat([sh_events_med_off, sh_med_off]) 

                if not w_med_off.empty:
                    w_events_med_off = pd.concat([w_events_med_off, w_med_off])
                    
                if not t_med_off.empty:
                    t_events_med_off = pd.concat([t_events_med_off, t_med_off])
                    
                    
    sh_times_med_on = get_total_time(sh_events_med_on)
    w_times_med_on = get_total_time(w_events_med_on)
    t_times_med_on = get_total_time(t_events_med_on)
    
    sh_times_med_off = get_total_time(sh_events_med_off)
    w_times_med_off = get_total_time(w_events_med_off)
    t_times_med_off = get_total_time(t_events_med_off)

    plot_times(sh_times_med_on, 'Start Hesitation Times w/ Meds')
    plot_times(sh_times_med_off, 'Start Hesitation Times w/out Meds')
    
    plot_times(w_times_med_on, 'Walking Times w/ Meds')
    plot_times(w_times_med_off, 'Walking Times w/out Meds')
    
    plot_times(t_times_med_on, 'Turn Times w/ Meds')
    plot_times(t_times_med_off, 'Turn Times w/out Meds')
    
    print("The mean time for lab subjects for Start Hesitation with meds is "
          + str(np.mean(sh_times_med_on)) + " seconds +/- " 
          + str(np.std(sh_times_med_on)) + " seconds and with meds off is " 
          + str(np.mean(sh_times_med_off)) + " seconds +/- "
          + str(np.std(sh_times_med_off)) + " seconds")
    
    print("The mean time for lab subjects for Walking with meds is "
          + str(np.mean(w_times_med_on)) + " seconds +/- " 
          + str(np.std(w_times_med_on)) + " seconds and with meds off is " 
          + str(np.mean(w_times_med_off)) + " seconds +/- "
          + str(np.std(w_times_med_off)) + " seconds")
    
    print("The mean time for lab subjects for Walking with meds is "
          + str(np.mean(t_times_med_on)) + " seconds +/- " 
          + str(np.std(t_times_med_on)) + " seconds and with meds off is " 
          + str(np.mean(t_times_med_off)) + " seconds +/- "
          + str(np.std(t_times_med_off)) + " seconds")