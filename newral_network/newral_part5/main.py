import os 
import random 
import warnings 
warnings.filterwarnings("ignore")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  ##This somehow avoids some errors 

##utils code here
from utils_image import train_test_split

src = 'C:\Real junkies\sem 8\Datasets\cats_and_dogs\PetImages'

if not os.path.isdir(os.path.join(src, "train")):
    train_test_split(src)

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D # type: ignore
from tensorflow.keras.layers import Dropout, Flatten, Dense # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

#Hyperparameters

FILTER_SIZE = 3 
NUM_FILTERS = 32
INPUT_SIZE = 32
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000//BATCH_SIZE # // to get interger values 
EPOCHS = 10

model = Sequential()
model.add (Conv2D(NUM_FILTERS,(FILTER_SIZE, FILTER_SIZE), input_shape = (INPUT_SIZE, INPUT_SIZE, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator = ImageDataGenerator(rescale = 1./255)

training_set = training_data_generator.flow_from_directory (os.path.join(src, "Train"), target_size = (INPUT_SIZE, INPUT_SIZE), batch_size = BATCH_SIZE, class_mode = 'binary')

test_set = testing_data_generator.flow_from_directory (os.path.join(src, "Test"), target_size = (INPUT_SIZE, INPUT_SIZE), batch_size = BATCH_SIZE, class_mode = 'binary')

model.fit(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose = 1)

score = model.evaluate(test_set, steps = 100)

for idx, metric in enumerate (model.metrics_names):
    print("{}: {}".format (metric, score[idx]))


### This code in normal is overfitting 