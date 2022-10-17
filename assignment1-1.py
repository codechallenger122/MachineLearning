# ===============================================================
# 1. These are all the modules we'll be using later. 
# Make sure you can import them before proceeding further.

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline # notebook 상에서 도표, 그림등을 바로 확인하기 위함.

# ===============================================================
# 2. 데이터를 다운받아 tar.gz 형태로 저장.

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = './data' # Change me to store data elsewhere

if not os.path.exists(data_root):
    os.makedirs(data_root)

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
          'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696) # ./data/notMNIST_large.tar.gz 으로 저장.
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)    # ./data/notMNIST_small.tar.gz 으로 저장.

# ===============================================================
# 2. tar.gz 을 extract 하고, extract 된 폴더 경로를 저장.
num_classes = 10
np.random.seed(0)

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
    # You may override by setting force=True.
        print(f'{root} already present - Skipping extraction of {filename}.')
    else:
        print(f'Extracting data for {root}. This may take a while. Please wait.')
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(f'Expected {num_classes} folders, one per class. Found {len(data_folders)} instead.')
    print(data_folders)
    return data_folders
  
train_folders = maybe_extract(train_filename) # 
# ['./data/notMNIST_large/A', './data/notMNIST_large/B', './data/notMNIST_large/C', 
#  './data/notMNIST_large/D', './data/notMNIST_large/E', './data/notMNIST_large/F', 
#  './data/notMNIST_large/G', './data/notMNIST_large/H', './data/notMNIST_large/I', './data/notMNIST_large/J']
test_folders = maybe_extract(test_filename)
# ['./data/notMNIST_small/A', './data/notMNIST_small/B', './data/notMNIST_small/C', 
#  './data/notMNIST_small/D', './data/notMNIST_small/E', './data/notMNIST_small/F', 
#  './data/notMNIST_small/G', './data/notMNIST_small/H', './data/notMNIST_small/I', './data/notMNIST_small/J']



