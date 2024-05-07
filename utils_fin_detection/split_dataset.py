import os
import random
import shutil

def split_dataset(dataset_folder, train_ratio=0.8):
    # Create folders for train and test datasets
    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get list of filenames in the images folder
    filenames = os.listdir(os.path.join(dataset_folder, 'images'))
    random.shuffle(filenames)  # Shuffle filenames randomly

    # Calculate number of files for training and testing
    num_train = int(len(filenames) * train_ratio)
    train_filenames = filenames[:num_train]
    test_filenames = filenames[num_train:]

    # Copy images and labels to train and test folders
    train_img_folder = os.path.join(train_folder, 'images')
    os.makedirs(train_img_folder, exist_ok=True)
    train_label_folder = os.path.join(train_folder, 'labels')
    os.makedirs(train_label_folder, exist_ok=True)
    for filename in train_filenames:
        name, ext = os.path.splitext(filename)
        image_src = os.path.join(dataset_folder, 'images', filename)
        label_src = os.path.join(dataset_folder, 'labels', name + '.txt')
        shutil.copy(image_src, os.path.join(train_img_folder, filename))
        shutil.copy(label_src, os.path.join(train_label_folder, name + '.txt'))

    test_img_folder = os.path.join(test_folder, 'images')
    os.makedirs(test_img_folder, exist_ok=True)
    test_label_folder = os.path.join(test_folder, 'labels')
    os.makedirs(test_label_folder, exist_ok=True)
    for filename in test_filenames:
        name, ext = os.path.splitext(filename)
        image_src = os.path.join(dataset_folder, 'images', filename)
        label_src = os.path.join(dataset_folder, 'labels', name + '.txt')
        shutil.copy(image_src, os.path.join(test_img_folder, filename))
        shutil.copy(label_src, os.path.join(test_label_folder, name + '.txt'))

# Usage
dataset_folder = 'all_dataset'
split_dataset(dataset_folder)
