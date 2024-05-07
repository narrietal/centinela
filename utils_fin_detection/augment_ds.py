import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imgaug
import imgaug.augmenters as iaa
from tqdm import tqdm
import numpy as np
import shutil
import random

class DataAugmentator():

    def __init__(self,image_folder, label_folder, output_folder) -> None:
        # Create output folder if it doesn't exist
        self.output_folder = output_folder
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.images_out_folder = os.path.join(output_folder,'images')
        self.labels_out_folder = os.path.join(output_folder,'labels')
        if not os.path.exists(self.images_out_folder):
            os.makedirs(self.images_out_folder)
        if not os.path.exists(self.labels_out_folder):
            os.makedirs(self.labels_out_folder)

    def augment_dataset(self, n_augmented_samples=5):
        # List all image files in the image folder
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg') or f.endswith('.png')]

        for i, image_file in tqdm(enumerate(image_files)):

            # Read the image
            image_path = os.path.join(self.image_folder, image_file)
            image = cv2.imread(image_path)

            # Save original image and label
            cv2.imwrite(os.path.join(self.images_out_folder, image_file), image) #save original
            label_file = image_file.split('.jpg')[0] + '.txt'
            shutil.copy(os.path.join(self.label_folder, label_file), os.path.join(self.labels_out_folder, label_file)) #save original label

            # Augmentations
            for i in range(n_augmented_samples):
                self.img_augment_chain(image, image_file, i)

    def img_augment_chain(self, img, image_file, augm_idx):

        # Chain of standard image augmentations
        img = iaa.Sometimes(0.35, iaa.Multiply((0.5, 1.5))).augment_image(img) #Changes brightness
        img = iaa.Sometimes(0.25, iaa.ChannelShuffle(p=0.35)).augment_image(img) #Inverts channels 0 and 1 with a prob of .35
        img = iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))).augment_image(img) #Gaussian noise with mean 0, std 0.15*255
        img = iaa.Sometimes(0.25, iaa.Dropout(p=(0, 0.10))).augment_image(img) #Drops pixels
        img = iaa.Sometimes(0.35, iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)).augment_image(img.astype(np.uint8)).astype(np.float32) # Hue and saturation (only works with uint8)
        img = iaa.Sometimes(0.25, iaa.LinearContrast((0.4, 1.6))).augment_image(img) #Contrast

        # flip image 50% of the time
        current_bbox = []
        if random.random() < 0.5:
            img, bbox = self.flip_image(img, image_file)
            current_bbox = bbox

        # crop image 50% of the time
        if random.random() < 0.5:
            img, bbox = self.zoom_in_image(img, image_file, current_bbox)
            current_bbox = bbox

        # Saving image
        augm_img_path = os.path.join(self.images_out_folder, image_file.split('.jpg')[0]+f'_AUGMENTED_{augm_idx}.jpg')
        cv2.imwrite(augm_img_path, img)

        label_file = image_file.split('.jpg')[0] + '.txt'
        augmented_label_path = os.path.join(self.labels_out_folder, label_file.split('.txt')[0]+f'_AUGMENTED_{augm_idx}.txt')
        # Save update box coordinates if changed, otherwise copy coordinates from label

        if len(current_bbox) > 0:
            with open(augmented_label_path, 'w') as f:
                for line in current_bbox:
                    f.write(line)
        else:
            # Copying label file as it has not been modified     
            label_path = os.path.join(self.label_folder, label_file)
            shutil.copy(label_path, augmented_label_path)

    def zoom_in_image(self, image, image_path, bbox, threshold=10):
        # Read image
        height, width = image.shape[:2]

        # Read from bbox argument if previous augmentation has modify it or read from origina label
        if len(bbox) > 0:
            lines = bbox
        else:
            label_file = image_path.split('.jpg')[0] + '.txt'
            label_path = os.path.join(self.label_folder, label_file)
            # Read label file
            with open(label_path, 'r') as f:
                lines = f.readlines()

        # Augm. only for single fin images
        if len(lines) < 2:

            parts = lines[0].strip().split(' ')
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:])

            x_center = float(x_center) * width
            y_center = float(y_center) * height
            box_width= float(box_width) * width
            box_height = float(box_height) * height

            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)
            
            # Calculate distances to the edges of the image
            left_dist = x_min
            right_dist = width - x_max
            bottom_dist = y_min
            top_dist = height-y_max

            # Check if the box is small
            min_dist = min(top_dist, bottom_dist, left_dist, right_dist)

            if min_dist > threshold:
                
                # Calculate extra margin
                margin = random.randint(1, min_dist-1)

                # Calculate the coordinates of the bounding box with margin
                x_min_crop = max(0, x_min - margin)
                y_min_crop = max(0, y_min - margin)
                x_max_crop = min(width, x_max + margin)
                y_max_crop = min(height, y_max + margin)

                # Calculate new coordinates for the cropped bounding box
                new_x_center = (x_center - x_min_crop) / (x_max_crop - x_min_crop)
                new_y_center = (y_center - y_min_crop) / (y_max_crop - y_min_crop)
                new_box_width = box_width / (x_max_crop - x_min_crop)
                new_box_height = box_height / (y_max_crop - y_min_crop)

                #augmented_image_path = os.path.join(self.images_out_folder, image_path.split('.jpg')[0]+f'_AUGMENTED_{augm_idx}_RANDOM_ZOOM_IN_{margin}.jpg')
                cropped_image = image[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
                cropped_image_resized = cv2.resize(cropped_image, (640, 640), interpolation = cv2.INTER_CUBIC)

                return cropped_image_resized, [f'{class_id} {new_x_center} {new_y_center} {new_box_width} {new_box_height}\n']

        return image, bbox

    def flip_image(self, image, image_file):
        # Flip the image horizontally
        flipped_image = cv2.flip(image, 1)

        # Read corresponding label file
        label_file = image_file.split('.jpg')[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_file)

        # Flip bounding box coordinates and write to new label file
        with open(label_path, 'r') as f:
            lines = f.readlines()

        flipped_lines = []
        for line in lines:
            values = line.strip().split()
            # Flip x-coordinate
            x_center = 1.0 - float(values[1])
            # No change to y-coordinate
            y_center = float(values[2])
            # No change to width and height
            width = float(values[3])
            height = float(values[4])
            flipped_line = f'{values[0]} {x_center} {y_center} {width} {height}\n'
            flipped_lines.append(flipped_line)

        return flipped_image, flipped_lines
    
    # Function to visualize bounding boxes on images and save plots
    def visualize_bounding_boxes(self, num_images=30):

        output_path = os.path.join(self.output_folder, 'plotted_examples')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # List all image files in the image folder
        image_files = [f for f in os.listdir(self.images_out_folder) if f.endswith('.jpg') or f.endswith('.png')]

        for i, image_file in enumerate(image_files[:num_images]):
            # Read the image
            image_path = os.path.join(self.images_out_folder, image_file)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read corresponding label file
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(self.labels_out_folder, label_file)

            # Read bounding box coordinates
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                values = line.strip().split()
                x_center = float(values[1]) * image.shape[1]
                y_center = float(values[2]) * image.shape[0]
                width = float(values[3]) * image.shape[1]
                height = float(values[4]) * image.shape[0]
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Save the plot with bounding boxes
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.savefig(os.path.join(output_path, f'{image_file}_with_boxes.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

# Example usage
image_folder = 'train/images'
label_folder = 'train/labels'
output_folder = 'augmented_train'

data_augmentator = DataAugmentator(image_folder, label_folder, output_folder)
data_augmentator.augment_dataset(n_augmented_samples=5)
data_augmentator.visualize_bounding_boxes(num_images=30)
