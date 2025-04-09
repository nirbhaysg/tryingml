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

# TensorFlow & Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Hyperparameters
FILTER_SIZE = 3  
NUM_FILTERS = 16  # Reduced from 32 to 16
INPUT_SIZE = 32
pad = 3
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000 // BATCH_SIZE  
EPOCHS = 20  # Increased for better training

# 🏆 Model Definition with Improvements
model = Sequential()

# Input layer
model.add(ZeroPadding2D(padding=pad, input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))

# Convolutional Block 1
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(MAXPOOL_SIZE, MAXPOOL_SIZE), strides=2))

# Convolutional Block 2
model.add(Conv2D(NUM_FILTERS * 2, (FILTER_SIZE, FILTER_SIZE), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(MAXPOOL_SIZE, MAXPOOL_SIZE), strides=2))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))  # L2 Regularization added
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))  # Binary Classification Output

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔥 Data Augmentation to Prevent Overfitting
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

testing_data_generator = ImageDataGenerator(rescale=1./255)  # No augmentation for test set

# Load Train & Test Data
training_set = training_data_generator.flow_from_directory(
    os.path.join(src, "Train"), target_size=(INPUT_SIZE, INPUT_SIZE),
    batch_size=BATCH_SIZE, class_mode='binary'
)

test_set = testing_data_generator.flow_from_directory(
    os.path.join(src, "Test"), target_size=(INPUT_SIZE, INPUT_SIZE),
    batch_size=BATCH_SIZE, class_mode='binary'
)

# 📉 Callbacks: Learning Rate Reduction & Early Stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# 🚀 Train the Model
model.fit(
    training_set, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
    validation_data=test_set, callbacks=[lr_scheduler, early_stopping], verbose=1
)

# Evaluate Model
score = model.evaluate(test_set, steps=100)

for idx, metric in enumerate(model.metrics_names):
    print("{}: {:.4f}".format(metric, score[idx]))
