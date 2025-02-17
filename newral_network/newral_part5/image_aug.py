import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  ##This somehow avoids some errors 
import matplotlib
matplotlib.use ("TkAgg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from matplotlib import pyplot as plt 
import numpy as np
import random 
 



src = 'C:\Real junkies\sem 8\Datasets\cats_and_dogs\PetImages'
dog_folder = 'C:\Real junkies\sem 8\Datasets\cats_and_dogs\PetImages\Dog'

image_generator = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')

fig, ax = plt.subplots(2,3, figsize = (20,10))
all_images = []

_,_, dog_images = next(os.walk(dog_folder))
random_img = random.sample (dog_images, 1)[0]
print (random_img)
random_img = plt.imread(os.path.join(dog_folder, random_img ))
all_images.append(random_img)

random_img = random_img.reshape((1,) + random_img.shape)
sample_augmented_images = image_generator.flow(random_img)


for _ in range (5):
    augmented_imgs = next(sample_augmented_images)
    for img in augmented_imgs:
        all_images.append(img.astype('uint8'))

for idx, img in enumerate(all_images):
    ax[int(idx/3), idx%3].imshow(img)
    ax[int(idx/3), idx%3].axis('off')
    if idx == 0:
        ax[int(idx/3), idx%3].set_title('Original Image')
    else:
        ax[int(idx/3), idx%3].set_title('Augmented Image {}'.format(idx))

plt.show()