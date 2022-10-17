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

%matplotlib inline # notebook 상에서 도표, 그림등을 바로 확인하기 위함.

# ===============================================================
# 2. 데이터를 다운받아 tar.gz 형태로 저장.

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = './data'   # 데이터 저장할 곳 위치

if not os.path.exists(data_root):
    os.makedirs(data_root)

def download_progress_hook(count, blockSize, totalSize):
    """
    A hook to report the progress of a download. 
    This is mostly intended for users with slow internet connections. 
    Reports every 5% change in download progress.
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
# 3. tar.gz 을 extract 하고, extract 된 폴더 경로를 저장.
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

# ===============================================================
# 4. notebook 상에서 파일을 visualize 해서 확인 가능.

from IPython.display import Image, display
display(Image(filename= os.path.abspath('data/notMNIST_large/A/ZXVyb2Z1cmVuY2UgYm9sZGl0YWxpYy50dGY=.png')))

# ===============================================================
# 5. 각 data 들을 pickle 형태로 저장한다.

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:            
            image_data = (plt.imread(image_file, 0).astype(float) - 
                        pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception(f'Unexpected image shape: {str(image_data.shape)}')
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception(f'Many fewer images than expected: {num_images} < {min_num_images}')

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
          # You may override by setting force=True.
          print(f'{set_filename} already present - Skipping pickling.')
        else:
            print(f'Pickling {set_filename}.')
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

train_datasets = maybe_pickle(train_folders, 45000) # picke 형태로 data 저장.
test_datasets = maybe_pickle(test_folders, 1800)

# ===============================================================
# 6. pickle 형태로 저장된 data 를 불러와서 visualize 한다.

# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray.
file = open(train_datasets[0], "rb")
images = pickle.load(file)
file.close()

fig, axes = plt.subplots(3, 3)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
for i, ax in enumerate(axes.flat):
        # Plot image.
    ax.imshow(images[i].reshape([28, 28]), cmap='binary')               
        
        # Remove ticks from the plot.
    ax.set_xticks([])
    ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
plt.show()

# ===============================================================
# 7. pickle 형태로 저장된 data 를 불러와서, 파일 정보 확인하기.

# Another check: we expect the data to be balanced across classes. Verify that if the number of samples across classes are balanced.
for data in train_datasets:
    file = open(data, "rb")
    images = pickle.load(file)
    print('file : ', file, 'number of samples : ', images.shape[0])
    file.close()
    
# ===============================================================
# 8. pickle 형태로 저장된 data 를 불러와서, training set, validation set, test set 만들기.
#    각각의 set 은 class 별로 같은 양의 data 를 불러와서 구성한다.

""" Generate train, test, validation sets
Merge and prune(제거하다/가지치다) the training data as needed. 

Depending on your computer setup, you might not be able to fit it all in memory, 
and you can tune train_size as needed. 

The labels will be stored into a separate array of integers 0 through 9.
Also create a validation dataset for hyperparameter tuning."""

def make_arrays(nb_rows, img_size):
    # dataset, lables 에 맞는 ndarray container 를 만들어 return.
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files) # 10개.
    valid_dataset, valid_labels = make_arrays(valid_size, image_size) # image_size = 28 
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes # 만약 20만 / 10 = 2만.. 즉, class 당 2만개의 data 를 가져올 것임.

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):       
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

"""
<output>

Training: (200000, 28, 28) (200000,)
Validation: (10000, 28, 28) (10000,)
Testing: (10000, 28, 28) (10000,)
"""

# ===============================================================
# 9. 생성된 training, validation(dev), test set 을 random 하게 섞는다!

#  Next, we'll randomize the data. 
#  It's important to have the labels well shuffled for the training and test distributions to match.
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# ===============================================================
# 10. 섞은 data 를 visual 하게 check 하기.
# Convince yourself that the data is still good after shuffling! 
# Display one of the images and see if it's not distorted.

images = train_dataset

fig, axes = plt.subplots(3, 3)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
for i, ax in enumerate(axes.flat):
        # Plot image.
    ax.imshow(images[i].reshape([28, 28]), cmap='binary')               
        
        # Remove ticks from the plot.
    ax.set_xticks([])
    ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
plt.show()

# ===============================================================
# 11. (9)번 step 에서 만든 random 하게 shuffle 된 training, validation, test set 을 
#      nonMNIST.pickle 파일로 저장한다.

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
    
# ===============================================================
# 12. 저장한정보 확인하기.      

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

# 13. 문제.
"""
Problem
Let's get an idea of what an off-the-shelf classifier can give you on this data. 
It's always good to check that there is something to learn, 
and that it's a problem that is not so trivial that a canned solution solves it.
Train a simple model on this data using  100, 500, 2500, 𝑎𝑛𝑑 10000 training samples. 

Hint: Use LogisticRegression model from sklearn.linear_model. 
You do not need to care about FutureWarning in sklearn.

Evaluation: Demonstration of training results from different sizes of dataset with test data.
