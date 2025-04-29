import os 
import random 
import warnings
warnings.filterwarnings("ignore")
from utils_image import train_test_split

src = 'C:\Real junkies\sem 8\Datasets\cats_and_dogs\PetImages'

from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore ## its "keras.application's'"
from keras.models import Model # type: ignore
from keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

INPUT_SIZE = 128
BATCH_SIZE = 16
STEPS_PER_EPOCH = 200
EPOCHS = 3

vgg16 = VGG16(include_top = False, weights = 'imagenet', input_shape = (INPUT_SIZE, INPUT_SIZE, 3))


for layer in vgg16.layers:
    layer.trainable = False # Freezing the pretrained layers?

input_ = vgg16.input
output_ = vgg16(input_)
last_layer = Flatten (name = 'flatten')(output_)
last_layer = Dense(1, activation = 'sigmoid')(last_layer)
model = Model(inputs = input_, outputs  = last_layer)

model.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator = ImageDataGenerator(rescale = 1./255)


training_set = training_data_generator.flow_from_directory(os.path.join(src, "Train"), target_size = (INPUT_SIZE, INPUT_SIZE), batch_size = BATCH_SIZE, class_mode = 'binary')
test_set = testing_data_generator.flow_from_directory(os.path.join(src, "Test"), target_size = (INPUT_SIZE, INPUT_SIZE), batch_size = BATCH_SIZE, class_mode = 'binary')

model.fit(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose = 1) #fit_generator = fit
score = model.evaluate(test_set, steps = 100) #evaluate_generator = evaluate 

for idx, metric in enumerate (model.metrics_names):
    print("{}: {}".format(metric, score[idx]))

