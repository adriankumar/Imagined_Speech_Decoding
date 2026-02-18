from utilities.chisco_preprocessing.data_loading import load_subject_experiment
import json 
import torch
import numpy as np
import re

channels_for_brain_signals = 122 #exclude auxillary channels
subject_nums = [1, 2, 3, 5] #4 not included because it didn't have the same number of experiments 
experiments = [10, 11, 12, 13, 14, 15] #only using 6 experiments for demo
json_path = r"utilities\chisco_preprocessing\unique_sentences.json"

#load all data in a list, index access via: data[list_elem][trial_num][dict_key]
#note trial length for each data[list_elem] is around 130, so use len() to check
def load_all_data():
    all_data =  []
    for subject_num in subject_nums:
        for experiment_num in experiments:
            data = load_subject_experiment(subject_num, experiment_num)
            all_data.append(data) 
    
    return all_data

#organise dataset by unique sentences as labels
def organise_dataset(dataset):

    #load unqiue sentences json
    with open(json_path, 'r', encoding='utf-8') as f:
        unique_sentences = json.load(f)
    
    dataset_dictionary = {}

    for chinese, english in unique_sentences.items():
        dataset_dictionary[chinese] = {
            'english': english, #store english translation
            'eeg_samples': [] #store all subject experiment trials with this sentence as its label
        }
    
    #for every subject experiment
    for data in dataset:
        trials = len(data) #get number of trials this subject's experiment has

        for trial_num in range(trials):
            #get eeg data
            eeg_data = data[trial_num]['input_features'][0, :channels_for_brain_signals, :] #shape (channels, timepoints)
            text_label_chinese = data[trial_num]['text']

            #if key in dictionary, append eeg data
            if text_label_chinese in dataset_dictionary:
                dataset_dictionary[text_label_chinese]['eeg_samples'].append(eeg_data)

    return dataset_dictionary 

#conversion for single sample
def eeg_to_tensor(eeg_sample, device='cpu', batch_dim=False):
    #convert eeg sample (numpy array) to pytorch tensor
    eeg_tensor = torch.tensor(eeg_sample, dtype=torch.float32) #shape (channels, timepoints)

    if batch_dim:
        eeg_tensor = eeg_tensor.unsqueeze(0) #add batch dimension at dim 0, new shape (1, channels, timepoints)

    return eeg_tensor.to(device)

#segment for single sample in raw numpy form
def segment_eeg_numpy(eeg_sample, window_size=500):
    channels, timepoints = eeg_sample.shape

    num_of_windows = timepoints // window_size 
    remainder_timepoints = timepoints % window_size #this will always be less than the window size (unless it is 0) so it will need to be zero padded

    windows = [] #store segments

    #extract the full number of windows first
    for i in range(num_of_windows):
        start = i * window_size #accumulator index to start at every loop
        end = start + window_size #accumulator end index

        window = eeg_sample[:, start:end] #channels x segment_length/window_size
        windows.append(window)
    
    #handle remainder if present
    if remainder_timepoints > 0:
        start = num_of_windows * window_size 
        remainder_window = eeg_sample[:, start:] #get the remainder

        #zero pad
        pad_size = window_size - remainder_timepoints
        zero_padding = np.zeros((channels, pad_size))
        padded_window = np.concatenate([remainder_window, zero_padding], axis=1)

        windows.append(padded_window) #remainder segment with padding to be equal segments
    
    return windows

def segment_eeg_tensor(eeg_sample, window_size=300):
    channels, timepoints = eeg_sample.shape
    
    num_of_windows = timepoints // window_size 
    remainder_timepoints = timepoints % window_size #this will always be less than the window size (unless it is 0) so it will need to be zero padded
    
    windows = [] #store segments each of size channels x window_size
    
    #extract full windows
    for i in range(num_of_windows):
        start = i * window_size #accumulator index to start at every loop
        end = start + window_size #accumulator end index
        window = eeg_sample[:, start:end]  #[channels, window_size]
        windows.append(window)
    
    #handle remainder with zero padding
    if remainder_timepoints > 0:
        start = num_of_windows * window_size 
        remainder_window = eeg_sample[:, start:]  # [channels, remainder_timepoints]
        
        #zero pad 
        pad_size = window_size - remainder_timepoints
        zero_padding = torch.zeros(
            channels, pad_size, 
            dtype=eeg_sample.dtype, 
            device=eeg_sample.device  # Keep on same device!
        )
        padded_window = torch.cat([remainder_window, zero_padding], dim=1) 
        
        windows.append(padded_window) #remainder segment with padding to be equal segments
    
    return windows

#we have 895 unique sentences and 1854 words
def get_vocab_list(unique_sentences_path=json_path, sort=True):

    #load the unique sentences
    with open(unique_sentences_path, "r", encoding="utf-8") as f:
        unique_sentences = json.load(f)
    
    vocab = set() #store unique words as vocab list

    for english_sentence in unique_sentences.values():
        for word in english_sentence.split():
            vocab.add(word)
    
    if sort:
        vocab = sorted(vocab)
    
    return vocab #list of words

def clean_word(word):
    # Remove leading/trailing punctuation except apostrophes
    # Pattern: remove . , ? ! ; : from start/end of word only
    cleaned = re.sub(r'^[.,?!;:]+|[.,?!;:]+$', '', word)
    return cleaned

#returns sentence label mapping, and the keys for it
#and the vocabulary sroted alphabetically in a list with its dict decoder
def map_label_to_index(dataset, length):
    label_mapping = {}
    keys = list(dataset.keys())
    keys = keys[:length]
    vocab = set() #each model trained for num_sentences, will only have the vocab that makes up those number of sentences; but it will always be alphabetically sorted

    for i, chinese_label in enumerate(keys):
        english_label = dataset[chinese_label]['english'].lower() #extract english translated label;#make all lower case before adding to sentence label and vocab label

        #remove any grammar; can keep apsotrphe words, model doesnt need to understand how language grammar works, its okay
        # #if a word has an apsotrophe variant because we dont check exact word by word, just semantic proximity so even if the
        # model outputs 'baby-ish' sentences thats fine and expected 
        #, . ? -
        sentence = [clean_word(word) for word in english_label.split()]


        label_mapping[i] = sentence #index i maps to unique english sentence label as list of words

        for word in sentence:
            vocab.add(word) #set only adds unique elements so duplicates 
    
    vocab_list = sorted(vocab)
    vocab_list.insert(0, '<PAD>') #add 'space' as vocab
    vocab_dict = {word: i for i, word in enumerate(vocab_list)}
    
    return label_mapping, keys, vocab_list, vocab_dict

#returns everything in tensor form
def get_train_list(train_size, dataset):

    if train_size > len(dataset):
        raise ValueError(f"train size {train_size} must be smaller than {len(dataset)}")
    
    label_mapping, train_sentences, vocab_list, vocab_dict = map_label_to_index(dataset, train_size)

    train_list = [] #each element is an entire imagined speech segment shape 1 x channels x timepoints (not segmented)
    train_labels = []
    
    for i, chinese_label in enumerate(train_sentences):
        imagined_speech_recordings = dataset[chinese_label]['eeg_samples'] 

        #loop through list of recordings for the label
        for sample in imagined_speech_recordings:
            train_list.append(sample) #currently shape channels x timepoints, segment and batch dim=1 during training
            train_labels.append(i) #its label
    
    #shape train_size x channels x timepoints
    train_tensor = torch.stack([torch.tensor(sample, dtype=torch.float32) for sample in train_list])
    labels_tensor = torch.tensor(train_labels, dtype=torch.long) #shape train_size

    return train_tensor, labels_tensor, label_mapping, vocab_list, vocab_dict
        