import os, pickle
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator #translation 

#note this will require internet connection
def translate_text(text, target_lang='en'):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

#change font to support Chinese characters
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 

#dataset params 
trial_num = 0 #out of 130
channels_for_brain_signals = 122 #exclude auxillary channels
subject_nums = [1, 2, 3, 5] #4 not included because it didn't have the same number of experiments 
experiments = [10, 11, 12, 13, 14, 15] #only using 6 experiments for demo

#returns pkl file for ONE subject, each subject has 130 trials; data[trial_num] <- dict with keys: 'text' and 'input_features'
def load_subject_experiment(subject_num, experiment_num):
    if subject_num not in subject_nums:
        raise ValueError(f"Subject number {subject_num} not in available subjects: {subject_nums}")
    
    if experiment_num not in experiments:
        raise ValueError(f"Experiment number {experiment_num} not in available experiments: {experiments}")
    
    data_dir = f"dataset_demos\chisco\subject_{subject_num}" #select subject
    file_name = f"sub-{subject_num:02d}_task-imagine_run-{experiment_num:03d}_eeg.pkl" #select experiment; contains 130 trials each
    file_path = os.path.join(data_dir, file_name)

    with open(file_path, 'rb') as f:
        data = pickle.load(f) #load file 
    
    return data

def get_sample(data, trial_num, channels=channels_for_brain_signals, language='en', translate=True):
    # if trial_num > 130:
    #     raise ValueError("Trial number exceeds available trials (130)")
    
    #get raw eeg from trial num
    sample_eeg = data[trial_num]['input_features'][0, :channels, :] #shape: (channels, timepoints)
    sample_eeg = sample_eeg * 1e6 #convert from volts to microvolts 

    #get chinese text label
    text_label_chinese = data[trial_num]['text']

    if translate:
        text_label_english = translate_text(text=text_label_chinese, target_lang=language)
    else:
        text_label_english = None

    return sample_eeg, text_label_chinese, text_label_english