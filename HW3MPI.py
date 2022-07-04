#!/usr/bin/env python
# coding: utf-8

# jupyter nbconvert --to script create_audio_features.ipynb

# /opt/anaconda3/bin/python -m venv "name" для создания своего Python енвайронмента

# In[ ]:


import os
import pandas as pd
import ast
import numpy as np

import librosa

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from collections import defaultdict
import pickle
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore")
# In[ ]:


# path to the small directory
SMALL_AUDIO_DIR = 'data/fma_small/'

# function to get the paths to all the songs in the small dataset
def audio_paths(AUDIO_DIR):
    AUDIO_PATHS = []
    # iterate through all the directories with songs in them
    for path in [os.path.join(SMALL_AUDIO_DIR, p) 
                 for p in os.listdir(SMALL_AUDIO_DIR) 
                 if not (p.endswith('checksums') or p.endswith('.txt') or p.endswith('.DS_Store'))]:
        # add all songs to the list
        AUDIO_PATHS = AUDIO_PATHS + [os.path.join(path, track).replace('\\', '/') for track in os.listdir(path)]
    
    return AUDIO_PATHS

# store all the small paths
SMALL_PATHS = audio_paths(SMALL_AUDIO_DIR)


# In[ ]:


# function to load metadata
# adapted from https://github.com/mdeff/fma/blob/master/utils.py
def metadata_load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    pd.CategoricalDtype(categories=SUBSETS, ordered=True))
        except ValueError:
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


# In[ ]:


# function to get genre information for each track ID
def track_genre_information(GENRE_PATH, TRACKS_PATH, FILE_PATHS, subset):
    """
    GENRE_PATH (str): path to the csv with the genre metadata
    TRACKS_PATH (str): path to the csv with the track metadata
    FILE_PATHS (list): list of paths to the mp3 files
    subset (str): the subset of the data desired
    """
    # get the genre information
    genres = pd.read_csv(GENRE_PATH)

    # load metadata on all the tracks
    tracks = metadata_load(TRACKS_PATH)

    # focus on the specific subset tracks
    subset_tracks = tracks[tracks['set', 'subset'] <= subset]

    # extract track ID and genre information for each track
    subset_tracks_genre = np.array([np.array(subset_tracks.index), 
                                  np.array(subset_tracks['track', 'genre_top'])]).T
    
    # extract track indices from the file paths
    track_indices = []
    for path in FILE_PATHS:
        track_indices.append(path.split('/')[-1].split('.')[0].lstrip('0'))

    # get the genre associated with each file path, thanks to the path ID
    track_indices = pd.DataFrame({'file_path':FILE_PATHS,'track_id':np.array(track_indices).astype(int)})
    tracks_genre_df = pd.DataFrame({'track_id': subset_tracks_genre[:,0], 'genre': subset_tracks_genre[:,1]})
    track_genre_data = track_indices.merge(tracks_genre_df, how='left')
    
    # label classes with numbers
    encoder = LabelEncoder()
    track_genre_data['genre_nb'] = encoder.fit_transform(track_genre_data.genre)
    
    return track_genre_data

world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
my_rank = world_comm.Get_rank()
# get genre information for all tracks from the small subset
GENRE_PATH = 'data/fma_metadata/genres.csv'
TRACKS_PATH = 'data/fma_metadata/tracks.csv'
subset = 'small'
small_tracks_genre = track_genre_information(GENRE_PATH, TRACKS_PATH, SMALL_PATHS, subset) #8000,4
# # split these paths and associated genres into training and test sets

SMALL_AUDIO_TRAIN, SMALL_AUDIO_TEST= train_test_split(SMALL_PATHS,test_size=0.2, random_state=42)


def compute_mfcc(file_path):
    x, sr = librosa.load(file_path, sr=None, mono=True)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    return mfccs

def compute_zcr(file_path):
    x, sr = librosa.load(file_path, sr=None, mono=True)
    zcr = librosa.feature.zero_crossing_rate(x)
    return zcr

def compute_chroma_stft(file_path, hop_length = 512):
    x, sr = librosa.load(file_path, sr=None, mono=True)
    stft = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    return stft

def compute_spectral_centroid(file_path):
    x, sr = librosa.load(file_path, sr=None, mono=True)
    centroid = librosa.feature.spectral_centroid(x, sr=sr)
    return centroid

def compute_spectral_rolloff(file_path):
    x, sr = librosa.load(file_path, sr=None, mono=True)
    rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)
    return rolloff


# # In[ ]:


CONVERTED_TRAIN_PATH = 'data/pickle/train/'

X_train = []
y_train = []

if not os.path.exists(CONVERTED_TRAIN_PATH):
    os.mkdir(CONVERTED_TRAIN_PATH)
mfcc = defaultdict(np.array)
zcr = defaultdict(np.array)
chroma_stft = defaultdict(np.array)
spectral_centroid = defaultdict(np.array)
spectral_rolloff = defaultdict(np.array)
y   = small_tracks_genre[small_tracks_genre.file_path.isin(SMALL_AUDIO_TRAIN)].genre.values
data =  SMALL_AUDIO_TRAIN[1000:1024]

N = len(data)
workloads = [ N // world_size for i in range(world_size) ]
for i in range( N % world_size ):
    workloads[i] += 1
my_start = 0
for i in range( my_rank ):
    my_start += workloads[i]
my_end = my_start + workloads[my_rank]
for i in range(workloads[my_rank]):
    small_path = data[my_start+i]
    X_train.append(np.concatenate((compute_spectral_centroid(small_path).flatten()[:2000], compute_spectral_rolloff(small_path).flatten()[:2000])))
    y_train.append(small_tracks_genre[small_tracks_genre.file_path == small_path].genre.values[0])
    try:
        mfcc[small_path] = compute_mfcc(small_path)
        zcr[small_path] = compute_zcr(small_path)
        chroma_stft[small_path] = compute_chroma_stft(small_path)
        spectral_centroid[small_path] = compute_spectral_centroid(small_path)
        spectral_rolloff[small_path] = compute_spectral_rolloff(small_path)
    except:
        print("{} - corrupt".format(small_path))

result_X_train = world_comm.gather(X_train, root=0)
result_y_train = world_comm.gather(y_train, root=0)
def myadd(d1, d2, dt):
    return {**d1, **d2}
def flatten(xss):
    return [x for xs in xss for x in xs]

counterSumOp = MPI.Op.Create(myadd, commute=True)

mfcc_all = world_comm.allreduce(mfcc, op=counterSumOp)
zcr_all = world_comm.allreduce(zcr, op=counterSumOp)
chroma_stft_all = world_comm.allreduce(chroma_stft, op=counterSumOp)
spectral_centroid_all = world_comm.allreduce(spectral_centroid, op=counterSumOp)
spectral_rolloff_all = world_comm.allreduce(spectral_rolloff, op=counterSumOp)

if my_rank == 0:
    X_train = flatten(result_X_train)
    y_train = flatten(result_y_train)
   
    pickle.dump( mfcc_all , open( CONVERTED_TRAIN_PATH + "mfcc.p", "wb" ) )
    pickle.dump( zcr_all, open( CONVERTED_TRAIN_PATH + "zcr.p", "wb" ) )
    pickle.dump( chroma_stft_all, open( CONVERTED_TRAIN_PATH + "chroma_stft.p", "wb" ) )
    pickle.dump(spectral_centroid_all, open( CONVERTED_TRAIN_PATH + "spectral_centroid.p", "wb" ) )
    pickle.dump( spectral_rolloff_all, open( CONVERTED_TRAIN_PATH + "spectral_rolloff.p", "wb" ) )


# In[ ]:

print ('Test')
CONVERTED_TEST_PATH = 'data/pickle/test/'

X_test = []
y_test = []

if not os.path.exists(CONVERTED_TEST_PATH):
    os.mkdir(CONVERTED_TEST_PATH)
mfcc_test = defaultdict(np.array)
zcr_test = defaultdict(np.array)
chroma_stft_test = defaultdict(np.array)
spectral_centroid_test = defaultdict(np.array)
spectral_rolloff_test = defaultdict(np.array)
y = small_tracks_genre[small_tracks_genre.file_path.isin(SMALL_AUDIO_TEST)].genre.values
data =  SMALL_AUDIO_TEST[700:750]

N = len(data)
workloads = [ N // world_size for i in range(world_size) ]
for i in range( N % world_size ):
    workloads[i] += 1
my_start = 0
for i in range( my_rank ):
    my_start += workloads[i]
my_end = my_start + workloads[my_rank]
for i in range(workloads[my_rank]):
    small_path = data[my_start+i]
    X_test.append(np.concatenate((compute_spectral_centroid(small_path).flatten()[:2000], compute_spectral_rolloff(small_path).flatten()[:2000])))
    y_test.append(small_tracks_genre[small_tracks_genre.file_path == small_path].genre.values[0])
    try:
        mfcc_test[small_path] = compute_mfcc(small_path)
        zcr_test[small_path] = compute_zcr(small_path)
        chroma_stft_test[small_path] = compute_chroma_stft(small_path)
        spectral_centroid_test[small_path] = compute_spectral_centroid(small_path)
        spectral_rolloff_test[small_path] = compute_spectral_rolloff(small_path)
    except:
        print("{} - corrupt".format(small_path))

result_X_test = world_comm.gather(X_test, root=0)
result_y_test = world_comm.gather(y_test, root=0)

mfcc_all_test = world_comm.allreduce(mfcc_test, op=counterSumOp)
zcr_all_test = world_comm.allreduce(zcr_test, op=counterSumOp)
chroma_stft_all_test = world_comm.allreduce(chroma_stft_test, op=counterSumOp)
spectral_centroid_all_test = world_comm.allreduce(spectral_centroid_test, op=counterSumOp)
spectral_rolloff_all_test = world_comm.allreduce(spectral_rolloff_test, op=counterSumOp)
if my_rank == 0:
    X_test = flatten(result_X_test)
    y_test = flatten(result_y_test)
    pickle.dump( mfcc_all_test , open( CONVERTED_TEST_PATH + "mfcc.p", "wb" ) )
    pickle.dump( zcr_all_test, open( CONVERTED_TEST_PATH+ "zcr.p", "wb" ) )
    pickle.dump( chroma_stft_all_test, open( CONVERTED_TEST_PATH + "chroma_stft.p", "wb" ) )
    pickle.dump(spectral_centroid_all_test, open( CONVERTED_TEST_PATH + "spectral_centroid.p", "wb" ) )
    pickle.dump( spectral_rolloff_all_test, open( CONVERTED_TEST_PATH + "spectral_rolloff.p", "wb" ) )

