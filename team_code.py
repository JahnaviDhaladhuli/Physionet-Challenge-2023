# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:25:20 2023

@author: DELL
"""

from helper_code import *
import numpy as np, os, sys
from scipy.signal import hilbert
from scipy import signal
from scipy.spatial import distance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.cluster import mutual_info_score
import antropy as ant
import mne
import math
import pandas as pd
from typing import Union
from typing import Tuple
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract features.
        current_features = get_features(patient_metadata, recording_metadata, recording_data)
        features.append(current_features)

        # Extract labels.
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)

    # Train the models.
    features = imputer.transform(features)
    outcome_model = RandomForestClassifier(criterion='entropy',n_estimators=400,max_depth=None,max_features='sqrt').fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(criterion='mse',n_estimators=400,max_depth=None,max_features='sqrt').fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)
    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Extract features.
    features = get_features(patient_metadata, recording_metadata, recording_data)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)
class all_features:

    def __init__(self,array):
        self.array = array
        self.sum1 = 0
        self.expc = 0
        self.max_mod = 0
        self.differential1 = []
        self.differential2 = [] 
        self.F1 = 0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.F6 = 0
        self.F7 = 0
        self.F8 = 0
        self.F9 = 0
        self.F10 = 0
        self.F11 = 0
        self.F12 = 0
        self.F13 = 0
        self.F14 = 0
        self.F15 = 0
        self.F16 = 0
        self.F17 = 0
        self.F18 = 0
        self.F19 = 0
        self.F20 = 0
        self.F21 = []
        self.F24 = []
        self.F25 = []
        self.F26 = []
        self.F27 = []
        self.F28 = []
        self.F29 = []
    def variance(self,a):
        l=[]
        for i in range(len(a)):
            s=0
            for j in range(len(a[i])):
                s+=math.sqrt(abs(a[i][j]))
            l.append((s/len(a[i]))**2)
        return l
        
    def sum_array(self):
        self.sum1 = sum(list(self.array))
        return 

    def expectation(self):
        s=0
        for j in range(len(self.array)):
            s+=self.array[j]*(j+1)
        self.expc = s
        return     
    def maximum_mod_value(self):
        self.max_mod = abs(max(self.array,key=abs))
        return 
    
    def first_order_differential(self):
        for i in range(len(self.array)):
            s=[]
            for j in range(len(self.array[i])-1):
                s.append(self.array[i][j+1]-self.array[i][j])
            self.differential1.append(s)
        return  
    def second_order_differential(self):
        for i in range(len(self.array)):
            s=[]
            for j in range(len(self.differential1[i])-1):
                s.append(self.differential1[i][j+1]-self.differential1[i][j])
            self.differential2.append(s)
        return  
        
    def maximum_value(self):
        self.F1 = max(self.array)
        return self.F1

    def max_index(self):
        self.F2=list(self.array).index(max(list(self.array)))+1
        return self.F2

    def equivalent_width(self):
        self.F3 = self.expc/self.F1
        return self.F3

    def centroid(self):
        self.F4 = self.expc/self.sum1
        return self.F4
        
    def root_mean_square_width(self):
        s=0
        k=[]
        for j in range(len(self.array)):
            s+=self.array[j]*(j+1)*(j+1)
        self.F6 = math.sqrt(abs(s/self.sum1))
        return self.F6
        
    def mean_value(self):
        self.F7 = np.mean(self.array)
        return self.F7
    
    def standard_deviation(self):
        s = 0
        for j in range(len(self.array)):
            s+= (self.array[j] - self.F7)**2
        self.F8 = math.sqrt(s/len(self.array))
        return self.F8
    
    def skewness(self):
        s = 0
        for j in range(len(self.array)):
            s+= (self.array[j] - self.F7)**3
        self.F9 = s/((self.F8**3)*len(self.array))
        return self.F9
                           
    def kurtosis(self):
        s = 0
        for j in range(len(self.array)):
            s+= (self.array[j] - self.F7)**4
        self.F10 = s/((self.F8**4)*len(self.array))
        return self.F10
            
    def median(self):
        self.F11 = np.median(self.array)
        return self.F11
    
    def rms_value(self):
        s=0
        for j in range(len(self.array)):
            s+=self.array[j]**2
        self.F12 = math.sqrt(s/len(self.array))
        return self.F12
    
    def sqrt_amp(self):
        s=0
        for j in range(len(self.array)):
            s+=math.sqrt(abs(self.array[j]))
        self.F13 = (s/len(self.array))**2
        return self.F13
    
    def peaktopeak(self):
        self.F14 = self.F1 - min(self.array)
        return self.F14
    
    def var(self):
        self.F15 = self.F8**2
        return self.F15
        
    def crest_factor(self):
        self.F16 = self.max_mod/self.F12
        return self.F16
    
    def shape_factor(self):
        s=0
        for j in range(len(self.array)):
            s+=abs(self.array[j])
        self.F17 = (len(self.array)*self.F12)/s
        return self.F17
        
    def impulse_factor(self):
        s=0
        for j in range(len(self.array)):
            s+=abs(self.array[j])
        self.F18 = (len(self.array)*self.max_mod)/s
        return self.F18
        
    def margin_factor(self):
        self.F19 = self.max_mod/self.F15
        return self.F19
    
    def form_factor(self):
        self.F20 = self.F12/self.F7
        return  self.F20
    
    def clearance_factor(self):
        self.F21 = self.max_mod/self.F13
        return self.F21
    
    def peak_index(self):
        self.F24 = self.max_mod/self.F8
        return  self.F24
    
    def skewness_index(self):
        self.F25 = self.F9/(math.sqrt(self.F15))**3
        return self.F25
    
    def first_quartile(self):
        self.F26 = np.quantile(self.array, .25)
        return self.F26
    
    def third_quartile(self):
        self.F27 = np.quantile(self.array, .75) 
        return self.F27
    
    def waveform_length(self):
        s=0
        for j in range(len(self.array)-1):
            s+=abs(self.array[j+1]-self.array[j])
        self.F28 = s
        return self.F28
    
    def wilson_amplitude(self):
        s=0
        for j in range(len(self.array)-1):
            x=abs(self.array[j+1]-self.array[j])
            if x>=0.5:
                s+=1
            else:
                s+=0
        self.F29 = s
        return self.F29
def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    
    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]       
def fun(data,k):
    feature=all_features(data)
    feature.sum_array()
    feature.expectation()
    F1 = feature.maximum_value()
    F2 = feature.max_index()
    F3 = feature.equivalent_width()
    F4 = feature.centroid()
    F6 = feature.root_mean_square_width()
    F7 = feature.mean_value()
    F8 = feature.standard_deviation()
    F9 = feature.skewness()
    F10 = feature.kurtosis()
    F11 = feature.median()
    F12 = feature.rms_value()
    F13 = feature.sqrt_amp()
    F14 = feature.peaktopeak()
    F15 = feature.var()
    F16 = feature.crest_factor()             ##here max is 1 for all signals
    # feature.maximum_mod_value()  
    F17 = feature.shape_factor()
    F18 = feature.impulse_factor()
    F19 = feature.margin_factor()
    F20 = feature.form_factor()
    F21 = feature.clearance_factor()
    F24 = feature.peak_index()
    F25 = feature.skewness_index()
    F26 = feature.first_quartile()
    F27 = feature.third_quartile()
    F28 = feature.waveform_length()
    F29 = feature.wilson_amplitude()
    sp_entropy=ant.spectral_entropy(data, sf=100, method='welch', normalize=True)
    permutation_entropy=ant.perm_entropy(data, normalize=True)
    entropy_svd=ant.svd_entropy(data, normalize=True)
    # Hjorth mobility and complexity
    hjorth_mob=ant.hjorth_params(data)[0]
    hjorth_comp=ant.hjorth_params(data)[1]
    zcr=ant.num_zerocross(data)
    delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=100,  fmin=0.5,  fmax=8.0, verbose=False)
    theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=100,  fmin=4.0,  fmax=8.0, verbose=False)
    alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=100,  fmin=8.0, fmax=12.0, verbose=False)
    beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=100, fmin=12.0, fmax=30.0, verbose=False)
    delta_psd_mean = np.nanmean(delta_psd)
    theta_psd_mean = np.nanmean(theta_psd)
    alpha_psd_mean = np.nanmean(alpha_psd)
    beta_psd_mean  = np.nanmean(beta_psd)
    delta_psd_std = np.nanstd(delta_psd)
    theta_psd_std = np.nanstd(theta_psd)
    alpha_psd_std = np.nanstd(alpha_psd)
    beta_psd_std  = np.nanstd(beta_psd)
    delta_psd_max = np.nanmax(delta_psd)
    theta_psd_max = np.nanmax(theta_psd)
    alpha_psd_max = np.nanmax(alpha_psd)
    beta_psd_max  = np.nanmax(beta_psd)
    delta_psd_freq = np.nanargmax(delta_psd)
    theta_psd_freq = np.nanargmax(theta_psd)
    alpha_psd_freq = np.nanargmax(alpha_psd)
    beta_psd_freq  = np.nanargmax(beta_psd)
    IF_max= max((np.diff(np.unwrap(np.angle(hilbert(data))))/(2.0*np.pi) * 100))
    IF_min= min((np.diff(np.unwrap(np.angle(hilbert(data))))/(2.0*np.pi) * 100))
    hurst_exp = get_hurst_exponent(data)
    power=(1/len(data))*(np.sum(np.abs(data)**2))
    # Petrosian fractal dimension
    pfd=ant.petrosian_fd(data)
    # Katz fractal dimension
    kfd=ant.katz_fd(data)
    # Higuchi fractal dimension
    hfd=ant.higuchi_fd(data)
    # Detrended fluctuation analysis
    dfd=ant.detrended_fluctuation(data)
    TD_features = {'Max_val'+k:F1,'Max_index'+k:F2,'Equi_width'+k:F3,'Centroid'+k:F4,'RMS_width'+k:F6,'Mean'+k:F7,'SD'+k:F8,'Skewness'+k:F9,'Kurtosis'+k:F10,'Median'+k:F11,'SQRT'+k:F13,'p-p'+k:F14,'Crest_factor'+k:F16,'Shape_factor'+k:F17,'Impulse_factor'+k:F18,'Margin_factor'+k:F19,'Form_factor'+k:F20,'Clearance_factor'+k:F21,'Peak_index'+k:F24,'Skewness_index'+k:F25,'1st-Q'+k:F26,'3rd-Q'+k:F27,'Waveform_length'+k:F28,'Wilson_amp'+k:F29,'sp_entropy'+k:sp_entropy,'permutation_entropy'+k:permutation_entropy,'entropy_svd'+k:entropy_svd,'hjorth_mob'+k:hjorth_mob,'hjorth_comp'+k:hjorth_comp,'zcr'+k:zcr,'delta_psd_mean'+k:delta_psd_mean,'theta_psd_mean'+k:theta_psd_mean,'alpha_psd_mean'+k:alpha_psd_mean,'beta_psd_mean'+k:beta_psd_mean,'delta_psd_max'+k:delta_psd_max,'theta_psd_max'+k:theta_psd_max,'alpha_psd_max'+k:alpha_psd_max,'beta_psd_max'+k:beta_psd_max,'delta_psd_freq'+k:delta_psd_freq,'theta_psd_freq'+k:theta_psd_freq,'beta_psd_freq'+k:beta_psd_freq,'alpha_psd_freq'+k:alpha_psd_freq,'delta_psd_std'+k:delta_psd_std,'theta_psd_std'+k:theta_psd_std,'beta_psd_std'+k:beta_psd_std,'alpha_psd_std'+k:alpha_psd_std,'IF_max'+k:IF_max,'IF_min'+k:IF_min,'power'+k:power,'Hurst_exp'+k:hurst_exp,'pfd'+k:pfd,'kfd'+k:kfd,'hfd'+k:hfd,'dfd'+k:dfd}
    return TD_features
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)
    if len(available_signal_data) > 0:
        temp1=[]
        for i in range(len(available_signal_data)):
            f=list(fun(available_signal_data[i][0],'0').values())+list(fun(available_signal_data[i][0],'1').values())+list(fun(available_signal_data[i][0],'2').values())+list(fun(available_signal_data[i][0],'3').values())+list(fun(available_signal_data[i][0],'4').values())+list(fun(available_signal_data[i][0],'5').values())+list(fun(available_signal_data[i][0],'6').values())+list(fun(available_signal_data[i][0],'7').values())+list(fun(available_signal_data[i][0],'8').values())+list(fun(available_signal_data[i][0],'9').values())+list(fun(available_signal_data[i][0],'10').values())+list(fun(available_signal_data[i][0],'11').values())+list(fun(available_signal_data[i][0],'12').values())+list(fun(available_signal_data[i][0],'13').values())+list(fun(available_signal_data[i][0],'14').values())+list(fun(available_signal_data[i][0],'15').values())+list(fun(available_signal_data[i][0],'16').values())+list(fun(available_signal_data[i][0],'17').values())
            temp1.append(f)
        tt=pd.DataFrame(temp1).describe().loc[['mean','std']].T
        temp2=np.hstack((tt['mean'],tt['std']))
    else:
        pass
    recording_features = temp2
    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features