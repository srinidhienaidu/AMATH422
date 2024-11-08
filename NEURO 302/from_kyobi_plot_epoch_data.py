#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:15:46 2024

@author: kyobi
"""

import numpy as np
import neuscitk as ntk
import matplotlib.pyplot as plt
import scipy as sc

#%% Import Labchart data and load into matrices
filename = "Downloads/BH_Lab_2_Part_2_10.mat"

dataset = ntk.LabChartDataset(filename)

experimental_data = np.zeros([10,len(dataset.get_block(1)[0])])
stimulus_data = experimental_data.copy()

for ii in range(10):
    experimental_data[ii,:] = dataset.get_block(ii+1)[0]*1000
    stimulus_data[ii,:] = dataset.get_block(ii+1)[1]

# plt.plot(stimulus_data.T)
# plt.show()

def get_trials_data(experimental_data,stimulus_data,SR,pre_time,
                    post_time,number_of_epochs,number_of_subpages,
                    start_first_epoch, start_final_epoch):
    
    trials_data = np.zeros([number_of_epochs,int((pre_time+post_time)*SR),number_of_subpages])
    trials_stimulus = trials_data.copy()
    time_vect = np.linspace(start_first_epoch,start_final_epoch,number_of_epochs)
    start_indexes = time_vect*SR
    for ii in range(10):
        for index, position in enumerate(start_indexes):
            start_index = int(position - (pre_time * SR))
            end_index = int(position + (post_time * SR))
            
            trials_data[index,:,ii] = experimental_data[ii,start_index:end_index]
            trials_stimulus[index,:,ii] = stimulus_data[ii,start_index:end_index]
            
    means = np.mean(trials_data[:,0:int(pre_time*SR),:],axis=1)
    mean_adj_trials_data = trials_data.copy()
    
    stimuli = np.round(np.mean(np.max(trials_stimulus,axis=1),axis=1),1)
    
    for ii in range(mean_adj_trials_data.shape[0]):
        for jj in range(mean_adj_trials_data.shape[2]):
            mean_adj_trials_data[ii,:,jj] = mean_adj_trials_data[ii,:,jj] - means[ii,jj]
    
    return mean_adj_trials_data, stimuli



#%% Define your parameters
SR = 40000 # sampling rate (samples per second)
pre_time = 0.15 # seconds before stimulus turns on
post_time = 1.25 # seconds after stimulus turns on
number_of_epochs = 11# number of different heights in your ramp experiment
number_of_subpages = 10
start_first_epoch = 1 # time in seconds
start_final_epoch = 21 # time in seconds

mean_adj_trials_data, stimuli = get_trials_data(experimental_data,stimulus_data,SR,pre_time,
                    post_time,number_of_epochs,number_of_subpages,
                    start_first_epoch, start_final_epoch)

#%% Plot Processed Data

ylabel = 'Response Amplitude (mV)'
xlabel = 'time (s)'
color_spread = 0.8

plt.clf()
plt.figure(figsize=(1,1),layout='constrained')
time_vals = np.linspace(-pre_time,post_time,mean_adj_trials_data.shape[1])
data = np.median(mean_adj_trials_data,axis=2)
for ii in range(mean_adj_trials_data.shape[0]):
    color = np.array([color_spread,color_spread,color_spread]) - (ii/11)*color_spread
    plt.plot(time_vals,data[ii,:],color=color,linewidth=0.25)
plt.yticks(np.linspace(-1000,500,num=5),fontsize=16,fontweight='bold')
plt.ylim([-800,600])
plt.ylabel(ylabel,fontsize=16,fontweight='bold')
plt.xlim([-pre_time-0.15,post_time+0.2])
plt.xticks([0,0.5,1,1.5],fontsize=16,fontweight='bold')
plt.xlabel(xlabel,fontsize=16,fontweight='bold')
plt.plot([0,0],[-1,1],color='black',linestyle=':',linewidth=0.5)
# plt.box(on=False)


plt.show()
plt.savefig('figures/Intensity_Ramp')


#%%

offset = 0.025
plt.clf()
for ii in range(mean_adj_trials_data.shape[0]):
    scaling_factor = (ii/11)
    offset_add = 0.06 - ii*0.012
    color = np.array([color_spread,color_spread,color_spread]) - scaling_factor*color_spread
    plt.plot(time_vals,data[ii,:],color=color,linewidth=0.25)
    plt.text(x=offset+offset_add,y=-100+1.4*data[ii,int((pre_time+offset)*SR)],
             s=str(stimuli[ii])+'V',
             c=color,fontsize=8,fontweight='bold')
plt.yticks(np.linspace(-100,500,num=5),fontsize=16,fontweight='bold')
plt.ylabel(ylabel,fontsize=16,fontweight='bold')
plt.xticks([0,0.1,0.2],fontsize=16,fontweight='bold')
plt.xlabel(xlabel,fontsize=16,fontweight='bold')
plt.plot([0,0],[-1,1],color='black',linestyle=':',linewidth=0.5)
plt.xlim([-0.05,0.2])
plt.ylim([-200,600])
plt.show()
plt.savefig('figures/Intensity_Ramp_L1_Closeup.pdf')

#%% Extracting L1, PR and L2
L_period = 0.1 # seconds
PR_timing = 1 # seconds after pre-time
xlabel = 'LED Intensity (V)'
ylabel = 'Median response size (mV)'
fontsize = 16
fontweight = 'bold'

L1_indexes = [int(pre_time*SR), int((pre_time+L_period)*SR)]
PR_indexes = [int(SR*(pre_time+PR_timing-L_period)),int(SR*(pre_time+PR_timing))]
L2_indexes = [PR_indexes[1], PR_indexes[1]+int(L_period*SR)]

L1 = np.max(mean_adj_trials_data[:,L1_indexes[0]:L1_indexes[1],:],axis=1)
PR = np.median(mean_adj_trials_data[:,PR_indexes[0]:PR_indexes[1],:],axis=1)
L2 = np.min(mean_adj_trials_data[:,L2_indexes[0]:L2_indexes[1],:],axis=1) - PR

L1vPR = L1/PR
L2vPR = L2/PR

def make_median_amplitude_plots(data,color):
    y = np.median(data,axis=1)
    max_y = np.max(y)
    y = y/max_y
    y_err = sc.stats.median_abs_deviation(data/max_y,axis=1)
    plt.plot(stimuli,y,color=color,marker='o')
    plt.fill_between(stimuli, y+y_err,
                               y-y_err,alpha=0.1,color=color)
    plt.xlim([2,5.5])
    plt.xlabel(xlabel,fontsize=fontsize,fontweight=fontweight)
    plt.ylabel(ylabel,fontsize=fontsize,fontweight=fontweight)
    plt.xticks([3,4,5],fontsize=fontsize,fontweight=fontweight)
    plt.yticks(fontsize=fontsize,fontweight=fontweight)
    if np.mean(y) > 0:
        plt.ylim([0,np.round(np.max(y)*1.2,2)])
    else:
        plt.ylim([np.round(np.min(y)*1.2,2),0])    
       
    # plt.show()

plt.clf()
make_median_amplitude_plots(L1,color='blue')
make_median_amplitude_plots(np.abs(L2),color='red')
make_median_amplitude_plots(np.abs(PR),color='green')
plt.ylim([0,1.5])
plt.show()

#%%
ylabel='Response Ratio'
make_median_amplitude_plots(np.abs(L1vPR))
make_median_amplitude_plots(L2vPR)

plt.show()
#%% Import Ramp Data
SR = 40000 # sampling rate (samples per second)
pre_time = 0.25 # seconds before stimulus turns on
post_time = 3 # seconds after stimulus turns on
number_of_epochs = 6 # number of different heights in your ramp experiment (number of pages here)
number_of_subpages = 10 # subpages per page
start_time = 1 # time when stimulus activates
max_time = 3 # seconds of the longest trial
    
length_times = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5])

max_size = [number_of_epochs*number_of_subpages,len(dataset.get_block(70)[0])]
experimental_data = np.zeros(max_size)
stimulus_data = experimental_data.copy()
for ii,val in enumerate(range(10,70)):
    data = dataset.get_block(val+1)[0]
    experimental_data[ii,:data.shape[0]] = data
    stimulus_data[ii,:data.shape[0]] = dataset.get_block(val+1)[1]
    
def get_trials_ramp_data(experimental_data,stimulus_data,SR,pre_time,
                    post_time,number_of_epochs,number_of_subpages,
                    start_first_epoch):
    
    trials_data = np.zeros([number_of_epochs,int((pre_time+post_time)*SR),number_of_subpages])
    trials_stimulus = trials_data.copy()
    constant_start_index = start_time*SR
    for jj in range(number_of_epochs):
        
        for ii in range(number_of_subpages):
                
            epoch_index = jj*number_of_subpages+ii
            start_index = int(constant_start_index - (pre_time * SR))
            end_index = int(start_index + ((post_time+pre_time) * SR))
            
            data = experimental_data[epoch_index,start_index:end_index]
            trials_data[jj,:data.shape[0],ii] = data
            trials_stimulus[jj,:data.shape[0],ii] = stimulus_data[epoch_index,start_index:end_index]
                
    means = np.mean(trials_data[:,0:int(pre_time*SR),:],axis=1)
    mean_adj_trials_ramp_data = trials_data.copy()
    
    stimuli = np.round(np.mean(np.max(trials_stimulus,axis=1) - np.min(trials_stimulus,axis=1),axis=1),1)
    
    plt.clf()
    plt.plot(mean_adj_trials_ramp_data[:,:,0].T)
    plt.show()
    for ii in range(mean_adj_trials_ramp_data.shape[0]):
        for jj in range(mean_adj_trials_ramp_data.shape[2]):
            mean_adj_trials_ramp_data[ii,:,jj] = mean_adj_trials_ramp_data[ii,:,jj] - means[ii,jj]
    plt.plot(mean_adj_trials_ramp_data[:,:,0].T)
    plt.clf()
    plt.plot(mean_adj_trials_ramp_data[:,:,0].T)
    plt.show()
    
    return mean_adj_trials_ramp_data, stimuli

mean_adj_trials_ramp_data, stimuli = get_trials_ramp_data(experimental_data,stimulus_data,SR,pre_time,
                    post_time,number_of_epochs,number_of_subpages,
                    start_first_epoch)

#%% Plot Processed Data

ylabel = 'Response Amplitude (mV)'
xlabel = 'time (s)'

time_vals = np.linspace(-pre_time,post_time,mean_adj_trials_ramp_data.shape[1])


plt.plot(time_vals,np.median(mean_adj_trials_ramp_data*1000,axis=2).T,color='black',linewidth=0.25)
plt.yticks(np.linspace(-1000,1000,num=5),fontsize=16,fontweight='bold')
plt.ylabel(ylabel,fontsize=16,fontweight='bold')
plt.xticks([0,0.5,1,1.5],fontsize=16,fontweight='bold')
plt.xlabel(xlabel,fontsize=16,fontweight='bold')
plt.plot([0,0],[-1,1],color='black',linestyle=':',linewidth=0.5)
plt.xlim([-0.5,1])
plt.ylim([-1000,1000])
plt.show()

plt.plot(time_vals,np.median(mean_adj_trials_ramp_data*1000,axis=2).T,color='black',linewidth=1)
plt.yticks(np.linspace(0,300,num=5),fontsize=16,fontweight='bold')
plt.ylabel(ylabel,fontsize=16,fontweight='bold')
plt.xticks([0,0.1,0.2],fontsize=16,fontweight='bold')
plt.xlabel(xlabel,fontsize=16,fontweight='bold')
plt.plot([0,0],[-1,1],color='black',linestyle=':',linewidth=0.5)
plt.xlim([-0.05,0.2])
plt.ylim([-100,400])
plt.show()

#%% Extracting L1, PR and L2
L_period = 0.2 # seconds
stim_off_times = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5])
xlabel = 'LED Intensity ON ramp (V/s)'
ylabel = 'Median response size (mV)'
fontsize = 16
fontweight = 'bold'

L1 = np.zeros([number_of_epochs,number_of_subpages])
L2 = L1.copy()
PR = L1.copy()
for ii,val in enumerate(stim_off_times+pre_time):
    L1_indexes = [int(pre_time*SR), int((pre_time+L_period)*SR)]
    # print(L1_indexes)
    PR_indexes = [int(SR*(val)-10),int(SR*(val))]
    print(np.array(PR_indexes))
    L2_indexes = [PR_indexes[1], PR_indexes[1]+int(L_period*SR)]
    
    L1[ii,:] = np.max(mean_adj_trials_ramp_data[ii,L1_indexes[0]:L1_indexes[1],:],axis=0)
    PR[ii,:] = np.median(mean_adj_trials_ramp_data[ii,PR_indexes[0]:PR_indexes[1],:],axis=0)
    L2[ii,:] = np.min(mean_adj_trials_ramp_data[ii,L2_indexes[0]:L2_indexes[1],:],axis=0) - PR[ii,:]


L1vPR = L1/PR
L2vPR = L2/PR

#%%

stimuli_vals = stimuli / stim_off_times
ylabel='Normalized Response'

def make_median_amplitude_ramp_plots(data,color):
    y = np.median(data,axis=1)
    max_y = np.max(y)
    y = y/max_y
    y_err = sc.stats.median_abs_deviation(data/max_y,axis=1)
    plt.plot(stimuli_vals,y,color=color,marker='o')
    plt.fill_between(stimuli_vals, y+y_err,
                               y-y_err,alpha=0.1,color=color)
    plt.xlim([0,30])
    plt.xlabel(xlabel,fontsize=fontsize,fontweight=fontweight)
    plt.ylabel(ylabel,fontsize=fontsize,fontweight=fontweight)
    
    # plt.xticks([3,4,5],fontsize=fontsize,fontweight=fontweight)
    # plt.yticks(fontsize=fontsize,fontweight=fontweight)
    # if np.mean(y) > 0:
    #     plt.ylim([0,np.round(np.max(y)*1.2,2)])
    # else:
    #     plt.ylim([np.round(np.min(y)*1.2,2),0])    
       
    # plt.show()

plt.clf()
make_median_amplitude_ramp_plots(L1,color='blue')
make_median_amplitude_ramp_plots(np.abs(L2),color='red')
make_median_amplitude_ramp_plots(np.abs(PR),color='black')
plt.legend(['L1','','L2','','PR'])
plt.show()

#%%
ylabel='Response Ratio'
plt.clf()
make_median_amplitude_ramp_plots(np.abs(L1vPR),color='blue')
make_median_amplitude_ramp_plots(L2vPR,color='red')

plt.show()