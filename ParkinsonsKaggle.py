#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:30:35 2023

@author: danikiyasseh
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from collections import defaultdict
import random
from tqdm import tqdm
import pickle
import copy
import scipy

CATEGORIES = ['StartHesitation','Turn','Walking'] # 3 classes
FEATNAMES = ['AccV','AccML','AccAP']
mlb = MultiLabelBinarizer()
lenc = LabelEncoder()
mlb.fit([CATEGORIES])
lenc.fit(CATEGORIES)

SAVE_DIR = '/kaggle/working/'
DATASOURCE = 'lab' # options: lab | realworld | lab_and_realworld
TASK = 'multiclass' # options: binary | multiclass
SAMPLES = 192
jump = 38

#gpus = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

#%%
# populate dataframe only with data associated with labels (0, 1, 2) for the 3 classes
def get_multiclass_df(FILENAMES,metadata_df):
    df = pd.DataFrame()
    for filename in tqdm(FILENAMES):
        data = pd.read_csv(filename)
        nevents = data[CATEGORIES].sum().sum()
        if nevents == 0: # skip if subject has no event whatsoever
            continue
        labels = mlb.inverse_transform(np.array(data[CATEGORIES])) # returns class tuple
        eventId = filename.split('/')[-1].split('.csv')[0]
        subjectId = metadata_df[metadata_df['Id']==eventId]['Subject'].item()
        subjectId = eventId
        
        for category in CATEGORIES:
            condition = data[category]==1
            category_df = data[FEATNAMES][condition]
            category_df['label'] = lenc.transform(pd.DataFrame(labels)[condition].iloc[:,0])
            category_df['subject'] = subjectId
            category_df['series'] = eventId
            category_df.reset_index(inplace=True,drop=True)
            df = pd.concat((df,category_df),0)

    df['label'] = df['label'].astype(int)
    return df

#%%
# obtain samples for the background class
#events_df = pd.read_csv('/home/danikiyasseh/datasets/tlvmc-parkinsons-freezing-gait-prediction/events.csv')
def get_background_df(FILENAMES,metadata_df):
    background_df = pd.DataFrame()
    for filename in tqdm(FILENAMES):
        data = pd.read_csv(filename)
        labels = [-1]*data.shape[0]
        eventId = filename.split('/')[-1].split('.csv')[0]
        subjectId = metadata_df[metadata_df['Id']==eventId]['Subject'].item()
    
        condition = ~data[CATEGORIES].any(axis=1)
        category_df = data[FEATNAMES][condition]
        category_df['label'] = pd.DataFrame(labels)[condition]
        category_df['subject'] = subjectId
        category_df['series'] = eventId
        category_df.reset_index(inplace=True,drop=True)
        background_df = pd.concat((background_df,category_df),0)
    
    background_df['label'] = background_df['label'].astype(int)
    return background_df

#%%
# prepare samples in dictionary format
def get_data_dict(df,unitConversion=1):
    data_dict = dict()
    for subject in tqdm(df['subject'].unique()):
        data_dict[subject] = defaultdict(list)
        subject_df = df[df['subject']==subject]
        for series in subject_df['series'].unique():
            series_df = subject_df[subject_df['series']==series]
            for category in series_df['label'].unique():
                #subject_bool = df['subject']==subject
                #series_bool = df['series']==series
                #category_bool = df['label']==category
                #combined_bool = subject_bool & series_bool & category_bool
                category_df = series_df[series_df['label']==category]
                if category_df.shape[0] >= SAMPLES: # at least this many samples for this subject from this category
                    #subject_df = df[combined_bool]
            
                    start = 0
                    end = start + SAMPLES
                    while end <= category_df.shape[0]:
                        chunk_category_df = category_df[start:end]
                        chunk_category_arr = np.array(chunk_category_df[FEATNAMES]) * unitConversion # SAMPLES x NFEATS
                        data_dict[subject][category].append(chunk_category_arr)
                        start = start + jump
                        end = start + SAMPLES
                
        for category in subject_df['label'].unique():
            if len(data_dict[subject][category]) == 0:
                data_dict[subject].pop(category)
            else:
                data_dict[subject][category] = np.stack(data_dict[subject][category]) # NCHUNKS x SAMPLES x NFEATS
    return data_dict

# get number of samples from each category
def get_sample_counts(data_dict):
    summary_df = pd.DataFrame(columns=['label','subject'])
    for subject in data_dict.keys():
        categories = data_dict[subject].keys()
        for category in categories:
            data = data_dict[subject][category]
            nsamples = data.shape[0]
            curr_df = pd.DataFrame([category]*nsamples,columns=['label'])
            curr_df['subject'] = subject
            summary_df = pd.concat((summary_df,curr_df),0)
    return summary_df

# calculate final number of samples to obtain a uniform distribution across the classes
def get_subsampled_sample_counts(summary_df,labels=[0,1,2]):
    new_summary_df = pd.DataFrame()
    min_nsamples = summary_df['label'].value_counts().min()
    for category in labels:
        category_df = summary_df[summary_df['label']==category]
        subsampled_category_df = category_df.sample(min_nsamples,random_state=0)
        new_summary_df = pd.concat((new_summary_df,subsampled_category_df),0)
    counts_df = new_summary_df.groupby(by=['subject'])['label'].value_counts()
    counts_df.name = 'count'
    counts_df = counts_df.reset_index()
    return new_summary_df, counts_df

def subsample_data_dict(data_dict,counts_df):
    # subsample data according to above calculated sample numbers
    new_data_dict = dict()
    for subject in data_dict.keys():
        new_data_dict[subject] = dict()
        for category in data_dict[subject].keys():
            combined_bool = (counts_df['subject']==subject) & (counts_df['label']==category)
            if combined_bool.sum() == 0:
                continue
            count = counts_df[combined_bool]['count'].item()
            subsampled_data = data_dict[subject][category][:count] 
            new_data_dict[subject][category] = subsampled_data

    # remove any empty entries
    subjects_to_keep = []
    for subject,data in new_data_dict.items():
        if data != dict():
            subjects_to_keep.extend([subject])
    new_data_dict = {subject:new_data_dict[subject] for subject in subjects_to_keep}
    return new_data_dict

#%%

if DATASOURCE == 'lab':
    DATA_DIR = '/home/danikiyasseh/datasets/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog'
    tdcsfog_metadata_df = pd.read_csv('/home/danikiyasseh/datasets/tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv')
    FILENAMES = [os.path.join(DATA_DIR,file) for file in os.listdir(DATA_DIR) if '.csv' in file]

    df = get_multiclass_df(FILENAMES,tdcsfog_metadata_df)
    
    multiclass_data_dict = get_data_dict(df)
    multiclass_summary_df = get_sample_counts(multiclass_data_dict)
    
elif DATASOURCE == 'realworld':
    DATA_DIR = '/home/danikiyasseh/datasets/tlvmc-parkinsons-freezing-gait-prediction/train/defog'
    defog_metadata_df = pd.read_csv('/home/danikiyasseh/datasets/tlvmc-parkinsons-freezing-gait-prediction/defog_metadata.csv')
    FILENAMES = [os.path.join(DATA_DIR,file) for file in os.listdir(DATA_DIR) if '.csv' in file]

    df = get_multiclass_df(FILENAMES,defog_metadata_df)
    
    multiclass_data_dict = get_data_dict(df,unitConversion=9.81)
    multiclass_summary_df = get_sample_counts(multiclass_data_dict)

elif DATASOURCE == 'lab_and_realworld':
    # LAB data
    DATA_DIR = '/home/danikiyasseh/datasets/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog'
    tdcsfog_metadata_df = pd.read_csv('/home/danikiyasseh/datasets/tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv')
    FILENAMES = [os.path.join(DATA_DIR,file) for file in os.listdir(DATA_DIR) if '.csv' in file]

    df = get_multiclass_df(FILENAMES,tdcsfog_metadata_df)
    
    tdcsfog_multiclass_data_dict = get_data_dict(df)
    tdcsfog_multiclass_summary_df = get_sample_counts(tdcsfog_multiclass_data_dict)
    
    # REALWORLD data
    DATA_DIR = '/home/danikiyasseh/datasets/tlvmc-parkinsons-freezing-gait-prediction/train/defog'
    defog_metadata_df = pd.read_csv('/home/danikiyasseh/datasets/tlvmc-parkinsons-freezing-gait-prediction/defog_metadata.csv')
    FILENAMES = [os.path.join(DATA_DIR,file) for file in os.listdir(DATA_DIR) if '.csv' in file]

    df = get_multiclass_df(FILENAMES,defog_metadata_df)
    
    defog_multiclass_data_dict = get_data_dict(df,unitConversion=9.81)
    defog_multiclass_summary_df = get_sample_counts(defog_multiclass_data_dict)
    
    multiclass_data_dict = {**tdcsfog_multiclass_data_dict,**defog_multiclass_data_dict}
    multiclass_summary_df = pd.concat((tdcsfog_multiclass_summary_df,defog_multiclass_summary_df),0)

#%%
""" Subsample the classes """
subsampled_multiclass_summary_df, subsampled_multiclass_counts_df = get_subsampled_sample_counts(multiclass_summary_df)
subsampled_multiclass_data_dict = subsample_data_dict(multiclass_data_dict, subsampled_multiclass_counts_df)

#%%
if TASK == 'binary': # convert problem to binary classification
    """ Get background data (from lab only for now) """
    background_df = get_background_df(FILENAMES,tdcsfog_metadata_df)
    background_data_dict = get_data_dict(background_df)
    
    # prepare data dict for binary classification (event vs. no event)
    background_summary_df = get_sample_counts(background_data_dict)
    multiclass_summary_df['label'] = multiclass_summary_df['label'].replace({0:1,1:1,2:1})
    background_summary_df['label'] = 0
    binary_summary_df = pd.concat((background_summary_df,multiclass_summary_df),0)
    subsampled_binary_summary_df, subsampled_binary_counts_df = get_subsampled_sample_counts(binary_summary_df,labels=[0,1])
    
    # add the background data to a combined data dict
    binary_data_dict = copy.deepcopy(multiclass_data_dict)
    for subject in binary_data_dict.keys():
        if subject in background_data_dict:
            binary_data_dict[subject][-1] = background_data_dict[subject][-1] # background originally labelled as -1 (to avoid overlapping with other classes)
    
    # aggregate the non background data into one category
    new_binary_data_dict = dict()
    for subject in binary_data_dict.keys():
        new_binary_data_dict[subject] = dict()
        categories = binary_data_dict[subject].keys()
        new_arr = []
        for category in categories:
            if category in [0,1,2]: # FOG event classes
                arr = binary_data_dict[subject][category]
                new_arr.append(arr)
        new_arr = np.stack(arr)
        new_binary_data_dict[subject][0] = binary_data_dict[subject][-1] # background data
        new_binary_data_dict[subject][1] = new_arr
    
    # need to get combine_dict (combine multiclass and background dict)
    subsampled_binary_data_dict = subsample_data_dict(new_binary_data_dict, subsampled_binary_counts_df)

#%%
with open('balanced_multiclass_data_dict','wb') as f:
    pickle.dump(subsampled_multiclass_data_dict,f)

# with open('balanced_binary_data_dict','wb') as f:
#     pickle.dump(subsampled_binary_data_dict,f)

#%%
with open('balanced_multiclass_data_dict','rb') as f:
    multiclass_data_dict = pickle.load(f)
    
# with open('balanced_binary_data_dict','rb') as f:
#     binary_data_dict = pickle.load(f)

#%%
""" inspect number of samples from each class """
counts = {i:0 for i in range(3)}
for key in multiclass_data_dict.keys():
    for cat in multiclass_data_dict[key].keys():
        counts[cat] += multiclass_data_dict[key][cat].shape[0]
print(counts)

#%%
def data_generator(subjects,data_dict):
    #random.shuffle(subjects)
    for subject in subjects:
        #subject = subject.decode("utf-8") # tf encodes input string to utf-8 (therefore you must decode it)
        #assert isinstance(data_dict,dict)
        categories_dict = data_dict[subject] 
        for category in categories_dict.keys():
            data = categories_dict[category]
            if isinstance(data,np.ndarray):
                nchunks = data.shape[0]
                for i in range(nchunks):
                    input_data = categories_dict[category][i] # 256 x 3
                    #channel_mean = np.mean(input_data,axis=0)
                    #channel_std = np.std(input_data,axis=0)
                    #input_data = (input_data - channel_mean)/channel_std
                    b,a = scipy.signal.butter(2, 15, 'low', fs=128)
                    input_data = scipy.signal.lfilter(b,a,input_data,axis=0)
                    output_data = [category]*SAMPLES # 256
                    yield tf.constant(input_data), tf.constant(output_data) 

#%%
FOLDS = 1
for fold in range(FOLDS):
    random.seed(fold)
    subjects = list(multiclass_data_dict.keys())
    random.shuffle(subjects)
    nsubjects = len(subjects)
    train_frac, val_frac = 0.7, 0.2
    train_nsubjects, val_nsubjects = int(train_frac*nsubjects), int(val_frac*nsubjects)
    train_subjects, val_subjects, test_subjects = subjects[:train_nsubjects], subjects[train_nsubjects:train_nsubjects+val_nsubjects], subjects[train_nsubjects+val_nsubjects:] 
    
    train_data = tf.data.Dataset.from_generator(lambda: data_generator(train_subjects,multiclass_data_dict), # args=[x,y,z]
                                                output_signature=(
                                                   tf.TensorSpec(shape=(SAMPLES,3), dtype=tf.float64),
                                                   tf.TensorSpec(shape=(SAMPLES), dtype=tf.int32))
                                               ) # shape is at the individual tensor level (not batch)
    
    val_data = tf.data.Dataset.from_generator(lambda: data_generator(val_subjects,multiclass_data_dict),
                                                output_signature=(
                                                   tf.TensorSpec(shape=(SAMPLES,3), dtype=tf.float64),
                                                   tf.TensorSpec(shape=(SAMPLES), dtype=tf.int32))
                                               ) # shape is at the individual tensor level (not batch)
    
    train_data = train_data.batch(batch_size=16) # 16
    train_data = train_data.shuffle(100,seed=fold)
    
    val_data = val_data.batch(batch_size=8) # 8
    val_data = val_data.shuffle(100,seed=fold)

    """ Make bidirectional """
    lstm_model = tf.keras.models.Sequential([ 
        tf.keras.layers.InputLayer(input_shape=(SAMPLES, 3)),
        # Shape [batch, time, features] 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)), # returns output at each time-step (i.e., many to many setup)
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=3)
    ])
    
    class multiclassAUPRC(tf.keras.metrics.AUC):
        
        def __init__(self,**kwargs): # you need to have the kwargs here to be able to load it in later
            super(multiclassAUPRC,self).__init__(from_logits=True,curve='PR')
        
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.one_hot(y_true,depth=3)
            super().update_state(y_true, y_pred)
    
    lstm_model.compile(
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy',multiclassAUPRC()])
    
    # lstm_model.compile(
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               metrics=['accuracy',tf.keras.metrics.AUC()])
    
    lstm_model.fit(
        x = train_data,
        validation_data = val_data,
        epochs = 50,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=5),
            tf.keras.callbacks.TensorBoard('./logs', update_freq=1),
            tf.keras.callbacks.ModelCheckpoint(
                                filepath='/tmp/checkpoint_fold%i' % fold,
                                save_weights_only=True,
                                monitor='val_accuracy',
                                mode='max',
                                save_best_only=True)
                                        ]
        )

#%%
lstm_model.save(os.path.join(SAVE_DIR,'lstm_parkinsons'))

#%%
lstm_model = tf.keras.models.load_model(os.path.join(SAVE_DIR,'lstm_parkinsons'),custom_objects={"multiclassAUPRC":multiclassAUPRC})

#%%
import glob
test_paths = glob.glob("test/**/**")

all_preds_df = pd.DataFrame()
for f in test_paths:
    df = pd.read_csv(f)
    df.set_index('Time', drop=True, inplace=True)
    df['Id'] = f.split('/')[-1].split('.')[0]
    df['Id'] = df['Id'].astype(str) + '_' + df.index.astype(str)
    
    start = 0
    end = start + jump 
    while end <= df.shape[0]:
        chunk_df = df[start:end]
        chunk_arr = np.array(chunk_df[FEATNAMES]) # SAMPLES x NFEATS
        chunk_arr = np.expand_dims(chunk_arr,0) # 1 x SAMPLES x NFEATS
        preds = lstm_model.predict(chunk_arr)
        preds_df = pd.DataFrame(preds.squeeze(0),columns=CATEGORIES,index=chunk_df.index)
        preds_df['Id'] = chunk_df['Id']
        all_preds_df = pd.concat((all_preds_df,preds_df),0)
        start += jump 
        end = start+jump
    
    # make sure to cover the final (smaller batch)
    final_nsamples = df.shape[0] - start
    chunk_df = df[-SAMPLES:]
    chunk_arr = np.array(chunk_df[FEATNAMES]) # SAMPLES x NFEATS
    chunk_arr = np.expand_dims(chunk_arr,0) # 1 x SAMPLES x NFEATS
    preds = lstm_model.predict(chunk_arr)
    preds_df = pd.DataFrame(preds.squeeze(0),columns=CATEGORIES,index=chunk_df.index)
    preds_df['Id'] = chunk_df['Id']
    preds_df = preds_df[-final_nsamples:]
    all_preds_df = pd.concat((all_preds_df,preds_df),0)

submission_df = all_preds_df[['Id','StartHesitation','Turn','Walking']]
submission_df.to_csv(os.path.join(SAVE_DIR,'submission.csv'),index=False)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    