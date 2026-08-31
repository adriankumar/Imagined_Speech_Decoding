import os, json, random 
import numpy as np 
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .helpers import (read_json, resolve_partial_label, get_labels_matrix, truncate_recording,
                      DECODER_CLASSES)

#only two arrangements trained on, single windows or sequence of windows; 
#anything between them is handled by the bptt truncation in the training loop 
WINDOWS = "windows" #one window per item, independent, for ffn
SEQUENCES = "sequences" #one whole recording per item, for rnns
ARRANGEMENTS = (WINDOWS, SEQUENCES)


def seq_len_for(arrangement):
    assert arrangement in ARRANGEMENTS, f"arrangement must be one of {ARRANGEMENTS}, got {arrangement}"
    return 1 if arrangement == WINDOWS else None

class Motor2aCache(Dataset):
    def __init__(self, cache_path, arrangement=WINDOWS, partial_window_threshold=0.7):
        self._dir = cache_path
        self._arrangement = arrangement
        self._threshold = partial_window_threshold
        self._meta = read_json(os.path.join(cache_path, "meta.json")) #meta data about cache

        self._run_meta = read_json(os.path.join(cache_path, "index.json"))["runs"] #the info for each feature window
        self._feat_windows = np.load(os.path.join(cache_path, "features.npy"), mmap_mode="r") #the actual cached data

        classes = np.load(os.path.join(cache_path, "classes.npy"))
        partial_windows = np.load(os.path.join(cache_path, "fractions.npy"))
        assert len(classes) == len(partial_windows) == self._feat_windows.shape[0], "label arrays are not row aligned with the feature array"
        self._classes = resolve_partial_label(classes, partial_windows, self._threshold)
        self._targets = get_labels_matrix()[self._classes]

        #labels for indexing the feat_windows at load
        self._seq_index, self._chunk_runs, self._seq_len = truncate_recording(self._run_meta,
                                                                              seq_len_for(arrangement))

    def __len__(self):
        return len(self._seq_index)
    
    def __getitem__(self, index):
        start_window_idx = self._seq_index[index]
        end_window_idx = start_window_idx + self._seq_len

        #returned shape is num_windows x nchns x F; num_windows is the new seq length
        #returned copy array independent of memory mapping from numpy as cache already stored as np.flaot32
        sequence = np.array(self._feat_windows[start_window_idx:end_window_idx], dtype=np.float32)
        labels = self._targets[start_window_idx:end_window_idx] #num_windows x model_class

        return sequence, labels, self._chunk_runs[index] #corresponding run index for diagnostics

    def trace_run_to_subject(self, run_idx):
        run = self._run_meta[int(run_idx)]
        return {k: run[k] for k in ("subject", "session", "run")}

    #the raw cached array, used by the centre and clip helpers without going through the loader
    @property
    def features(self):
        return self._feat_windows

    @property
    def arrangement(self):
        return self._arrangement

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def threshold(self):
        return self._threshold

    @property
    def decoder_classes(self):
        return DECODER_CLASSES

    @property
    def num_channels(self):
        return self._feat_windows.shape[-2]

    @property
    def num_features(self):
        return self._feat_windows.shape[-1]

    @property
    def feature_names(self):
        return self._meta["toggled_features"]

    @property
    def window_seconds(self):
        return self._meta["window_seconds"]

    @property
    def subjects(self):
        return sorted({run["subject"] for run in self._run_meta}, key=int)

    @property
    def runs(self):
        return self._run_meta

    @property
    def env_path(self):
        return self._meta["env_path"]

    @property
    def meta(self):
        return self._meta

    def describe(self):
        print(f"cache: {self._dir}")
        print(f"window: {self.window_seconds}s = {self._meta['window_size']} samples")
        print(f"features: {self._feat_windows.shape} {self._feat_windows.dtype}")
        print(f"chunks: {len(self)} of {self._seq_len} windows | {len(self._run_meta)} runs")
        print(f"subjects: {self.subjects}")
        print(f"channels: {self.num_channels} | features {self.feature_names}")
        print(f"decoder cls: {DECODER_CLASSES} | threshold {self._threshold}")
        print(f"positives: {dict(zip(DECODER_CLASSES, self._targets.sum(axis=0).astype(int)))}")
        print(f"rest rows: {int((self._classes == 0).sum())} of {len(self._classes)}")


#one cache per window size; each cache has diff values of features computed from
#window sizes 0.2, 0.5 and 0.7 in seconds
def build_sources(cache_paths, arrangement, threshold):
    return [Motor2aCache(cache_path=path, arrangement=arrangement,
                         partial_window_threshold=threshold) for path in cache_paths]


#for ffns & eelectrode vs coeffs experiemnt
def build_window_loader(cache_paths, batch_size=64, threshold=0.7, shuffle=True,
                        num_workers=0, drop_last=False):

    sources = build_sources(cache_paths, WINDOWS, threshold)

    loader = DataLoader(dataset=ConcatDataset(sources), batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=drop_last)

    return sources, loader


#windows per run varies with window size, so items from different sources cannot be stacked
#into one batch; sources are mixed at the batch level instead, each batch from one source
#third element of each batch is that source's window seconds
class SequenceLoader:
    def __init__(self, sources, batch_size=64, shuffle=True, num_workers=0, drop_last=False):
        self._sources = sources
        self._loaders = [DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers, drop_last=drop_last) for ds in sources]

    #every source contributes the same number of recordings regardless of window size
    def __len__(self):
        return sum(len(loader) for loader in self._loaders)

    def __iter__(self):
        iterators = [iter(loader) for loader in self._loaders]
        schedule = [i for i, loader in enumerate(self._loaders) for _ in range(len(loader))]
        random.shuffle(schedule)

        for i in schedule:
            windows, labels, _ = next(iterators[i]) #run index unused
            yield windows, labels, self._sources[i].window_seconds

    def describe(self):
        for ds in self._sources:
            ds.describe()
            print()

    @property
    def sources(self):
        return self._sources

#for rnns
def build_sequence_loader(cache_paths, batch_size=64, threshold=0.7, shuffle=True,
                          num_workers=0, drop_last=False):

    sources = build_sources(cache_paths, SEQUENCES, threshold)

    loader = SequenceLoader(sources, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, drop_last=drop_last)

    return sources, loader