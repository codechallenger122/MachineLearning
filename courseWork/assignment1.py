""" # 1. import module """
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

# 쥬피터 노트북을 실행한 브라우저에서 바로 그림을 볼 수 있게, 
# 즉 브라우저 내부(inline) 에 바로 그려지게 하는 코드이다.
%matplotlib inline

""" # 2. nonMNIST 데이터 다운로드 """
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

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

""" #3. dataSet 을 extract 한다. from compressed .tar.gz 파일.
        give me a directories, labeled A through J. """

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
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

""" #4. take a peek at some of data : exercise 1. """
from IPython.display import Image, display

display(Image(filename= os.path.abspath('data/notMNIST_large/A/ZXVyb2Z1cmVuY2UgYm9sZGl0YWxpYy50dGY=.png')))  

""" #5. load datasets 
Now let's load the data in a more manageable format. 
Since, depending on your computer setup you might not be able to fit it all in memory, 
we'll load each class into a separate dataset, store them on disk and curate them independently. 
Later we'll merge them into a single dataset of manageable size.
We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values.
A few images might not be readable, we'll just skip them. """

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

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

""" #6. Verify that the data still looks good. : Exercise 2"""
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

""" #7. Another check : sample 의 개수가 balanced 되어있는지 체크 """
for data in train_datasets:
    file = open(data, "rb")
    images = pickle.load(file)
    print('file : ', file, 'number of samples : ', images.shape[0])
    file.close()
    
""" 
