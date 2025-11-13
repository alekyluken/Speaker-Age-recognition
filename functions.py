

import librosa 
import numpy as np 
import pandas as pd
import scipy as sci
import matplotlib.pyplot as plt
import noisereduce as nr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import re
from sklearn.metrics import root_mean_squared_error
import maad.features as mad_features
import maad.sound as mad_sound
import parselmouth


#I will make a function to one hot encode only the first n ethnicity groups
def one_hot_ethnicity(dataset:pd.DataFrame, n, etn:pd.DataFrame):
    """function that takes in input the dataset, the number of ethnicity groups to one hot encode and a dataset of the sorted
    frequency of the ethnicity groups and returns the dataset with the one hot encoding and removes the original row ethnicity

    Args:
        dataset (pd.DataFrame): Dataset to be one hot encoded
        n (_type_): How many dimentions to one hot encode
        etn (pd.DataFrame): Dataset with the frequency of the ethnicity groups

    Returns:
        dataset (pd.DataFrame): Dataset with the one hot encoding and without the column ethnicity
    """
    
    dummies = pd.get_dummies(dataset['ethnicity'])
    # print(dummies[etn.index[:n].values])
    dataset = pd.concat([dataset, dummies[etn.index[:n].values]], axis=1)
    dataset.drop(columns =['ethnicity'], inplace=True)  
    # display(dataset)
    return dataset

def extract_float_from_brackets(s: str) -> float:
    
    pattern = r"\[(\d+\.\d+)\]" #pattern to find a floar number between brackets
    match = re.search(pattern, s)
    return float(match.group(1)) 
        
    
def make_tempo_float(dataset:pd.DataFrame)->pd.DataFrame:
    """
    Function that takes in input a dataset with col tempo as [_float_] and retuns the dataset with column tempo as float 
    """
    dataset['tempo'] = dataset['tempo'].apply(lambda x: extract_float_from_brackets(x))
    return dataset


def encode_gendedr(dataset:pd.DataFrame)->pd.DataFrame:
    '''
    takes in input a dataset with the column gendere male female and ebcodes it as True False
    '''
    #i will encode the male as true and the female as false
    dataset['gender'] = dataset['gender'].apply(lambda x: True if x =='male' else False)
    return dataset



def try_model(X,y):
    """takes as input the features and the target and returns the model trained on the data
    
        -the model is arandom forest regressor with 200 estimators
        -the data is split into train and test with 20% of the data as test
        -the printed metric is the root mean squared error 
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # clf = RandomForestRegressor(n_estimators=500, n_jobs=-1,min_samples_leaf=4,min_samples_split=4,max_features='log2')
    clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    
    clf.fit(x_train, y_train)
    ypred = clf.predict(x_test)
    print(root_mean_squared_error(y_test, ypred))
    return clf



def print_feature_importance(clf, dataset):
    '''
    Input: 
    - clf: the trained model
    - dataset: the dataset used to train the model
    Output:
    - prints the feature importance of the model
    '''
    
    importances = clf.feature_importances_
    feature_names = dataset.columns

    # Create a DataFrame for better visualization
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    print(feature_importances.to_string())

def modify_features_duration(dataset,dev_paths):
    duration = extract_audio_duration(dev_paths)
    dataset['duration'] = duration
    # dataset['energy'] = [dataset.iloc[i]['energy']/dataset.iloc[i]['duration'] for i in range(dataset.shape[0])]
    # dataset['words_per_sec'] = [dataset.iloc[i]['num_words']/dataset.iloc[i]['duration'] for i in range(dataset.shape[0])]
    # dataset['characters_per_sec'] = [dataset.iloc[i]['num_characters']/dataset.iloc[i]['duration'] for i in range(dataset.shape[0])]
    return dataset


def get_features_form_dataset(dataset:pd.DataFrame,dev_paths = None,is_train = True)->pd.DataFrame:
    # colums_to_drop = ['path', 'sampling_rate','Id','age', 'num_words', 'num_characters','ethnicity','tempo','energy','gender']
    colums_to_drop = ['path', 'sampling_rate','Id','age', 'num_words', 'num_characters','ethnicity','tempo','energy']
    
    if(not is_train):
        colums_to_drop.remove('age')
    dataset = encode_gendedr(dataset)
    dataset = make_tempo_float(dataset)
    if(dev_paths is not None):
        dataset = modify_features_duration(dataset,dev_paths)
    dataset.drop(columns = colums_to_drop, inplace=True)
    return dataset




#extract the mfcc features from the audio files by dooing the mean on the tempotral axis
def extract_mfcc(dev_paths, sampling_rate,n_mfcc=13, filtered = False)->pd.DataFrame:
    '''
    Input:
    - sampling_rate: the sampling rate of the audio files
    - n_mfcc: the number of mfcc features to extract
    - filtered: if the audio files are filtered or not
    - dev_paths: the paths to the audio files
    Output:
    - a dataframe with the mfcc features as cols and the audio files as rows
    '''
    
    mfccs = []
    for file in dev_paths:
        if filtered:
            file = 'filtered_dev/'+ file.split('/')[1]
        y, sr = librosa.load(file, sr=sampling_rate)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc, axis=1)
        mfccs.append(mfcc)
    mfccs = pd.DataFrame(mfccs)
    mfccs.columns = ['mfcc-'+str(x) for x in mfccs.columns.astype(str)]
    return mfccs



def compute_entropy(dev_paths,sampling_rate)->pd.DataFrame:
    '''
    INPUT: the paths and the sampling rate of the audio files
    OUTPUT: a dataframe with the entropy of the audio files
    '''
    enotrpies = []
    for file in dev_paths:
        audio = librosa.load(file, sr=sampling_rate)[0]
        entropy = mad_features.temporal_entropy(audio)
        enotrpies.append(entropy)
    enotrpies = pd.DataFrame(enotrpies,columns=['entropy'])
    return enotrpies   

def compute_peak_frequency(dev_paths,sampling_rate)->pd.DataFrame:
    '''
    INPUT: the paths and the sampling rate of the audio files
    OUTPUT: a dataframe with the peak frequency of the audio files, and the amplitude of the peak frequency
    '''
    freq = []
    amp = []
    for file in dev_paths:
        audio = librosa.load(file, sr=sampling_rate)[0]
        peak_freq, peak_amplitude = mad_features.peak_frequency(audio, fs = sampling_rate, nperseg=256, amp=True)
        freq.append(peak_freq)
        amp.append(peak_amplitude)
    df  = pd.DataFrame({
        'peak_freq': freq,
        'peak_amplitude': amp
    })
    return df   


def compute_temporal_median(dev_paths,sampling_rate)->pd.DataFrame:
    '''
    INPUT: the paths and the sampling rate of the audio files
    OUTPUT: a dataframe with the temporal median of the audio files
    '''
    
    median = []
    for file in dev_paths:
        audio = librosa.load(file, sr=sampling_rate)[0]
        temp_med= mad_features.temporal_median(audio)
        median.append(temp_med)
        
    df = pd.DataFrame(median,columns=['temporal_median'])
    return df   

def compute_all_temporal(dev_paths):
    df = pd.DataFrame(columns=['sm', 'sv', 'ss', 'sk', 'Time 5%', "Time 25%", "Time 50%", "Time 75%", "Time 95%  ", "zcr", "duration_50", "duration_90"])
    for file in dev_paths:
        audio = librosa.load(file, sr=22050)[0]
        serie = pd.Series(mad_features.all_temporal_features(audio, fs=22050, nperseg=256).values[0],
                     index=['sm', 'sv', 'ss', 'sk', 'Time 5%', "Time 25%", "Time 50%", "Time 75%", "Time 95%  ", 
                            "zcr", "duration_50", "duration_90"])
        df = pd.concat([df, serie.to_frame().T], ignore_index=True)
    return df


def compute_pitch_attributes(dev_paths,sampling_rate):
    ''''
    INPUT: the paths and the sampling rate of the audio files
    Ouput: a dataframe with the number of voiced frames, the mean absolute slope, the duration and the number of frames of the audio files
    '''
    df = pd.DataFrame(columns=['nVoicedFrames', 'meanAbsoluteSlope', 'nFrames'])
    for file in dev_paths:
        audio = librosa.load(file, sr=sampling_rate)[0]
        sound = parselmouth.Sound(audio)
        pitch = sound.to_pitch()
        serie = pd.Series([pitch.count_voiced_frames(), pitch.get_mean_absolute_slope(),  pitch.n_frames],
                        index=['nVoicedFrames', 'meanAbsoluteSlope', 'nFrames'])
        df = pd.concat([df, serie.to_frame().T], ignore_index=True)
    return df 

def compute_additional_features(dev_paths,sampling_rate):
    peak_freq = compute_peak_frequency(dev_paths, sampling_rate)
    temporal_median = compute_temporal_median(dev_paths, sampling_rate)
    pitch_vls = compute_pitch_attributes(dev_paths, sampling_rate)
    entropy = compute_entropy(dev_paths, sampling_rate)   
    return pd.concat([peak_freq, temporal_median, pitch_vls,entropy],axis=1)

def extract_skew_curtosis_audio(dev_paths):
    for file in dev_paths:
        audio = librosa.load(file, sr=22050)[0]
        serie = pd.Series(mad_features.skew_curtosis(audio, fs=22050).values[0],
                     index=['skewness', 'curtosis'])
        df = pd.concat([df, serie.to_frame().T], ignore_index=True)
        
def compute_spectral_energy(dev_paths):
    df = pd.DataFrame(columns=['spectralEnergy250-650', 'spectralEnergy1000-8000'])
    for file in dev_paths:
        audio = librosa.load(file, sr=22050)[0]
        S = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=22050)
        serie =  pd.Series([np.sum(np.abs(S[(freqs >= 250) & (freqs <= 650)])**2), np.sum(np.abs(S[(freqs >= 1000) & (freqs <= 8000)])**2)],
                            index=['spectral_energy_betwn250-650', 'spectral_energy_betwn1000-8000'])
        df = pd.concat([df, serie.to_frame().T], ignore_index=True)
    return df

def extract_audio_duration(dev_paths, filtered = False):
    '''
    Input:
    - dev_paths: the paths to the audio files
    - filtered: if the audio files are filtered or not
    Output:
    - a numpy array with the duration of the audio files
    
    '''
    duration = []

    for flie in dev_paths:
        if(filtered):
            file_filtered = 'filtered_dev/'+ flie.split('/')[-1]
        else:
            file_filtered = flie    
        duration.append(librosa.get_duration(path=file_filtered))
    return np.array(duration)
    
def get_fondamental_freq(dev_paths,sampling_rate):
    '''
    INPUT: the paths and the sampling rate of the audio files
    OUTPUT: a dataframe with the fondamental frequency of the audio files
    '''
    df = pd.DataFrame(columns=['fondamental_freq_mean', 'fondamental_freq_std'])
    for file in dev_paths:
        audio = librosa.load(file, sr=sampling_rate)[0]
        audio = pd.Series(mad_features.fundamental_frequency(audio, fs=sampling_rate))
        
    df = pd.DataFrame(freq,columns=['fondamental_freq'])
    return df

def get_mfccsSTD(dev_paths,sampling_rate):
    eval_mfccsSTD = []
    n_std = 13
    for file in dev_paths:
        y,sr = librosa.load(file,sr=sampling_rate)
        mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_std)
        mfcc = np.std(mfcc,axis=1)
        eval_mfccsSTD.append(mfcc)
    return pd.DataFrame(eval_mfccsSTD,columns=['mfcc_std_'+str(i) for i in range(n_std)])

