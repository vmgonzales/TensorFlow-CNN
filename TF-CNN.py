# -*- coding: utf-8 -*-
'''
TensorFlow CNN for image recognition
=====

Big Idea:   This code was adapted from a Google Developers Codelab.

            The original doesn't seem to be abailable as of 10 May 2022, but
            it was here:
            https://developers.google.com/profile/pathways/beta/tensorflow



Topics Included
==============
* 

Created around 1 May 2022

@author: vmgon
'''


#%%



#%%
import os
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile


print(len(os.listdir('./data/PetImages/Cat/')))
print(len(os.listdir('./data/PetImages/Dog/'))) 
# Expected Output:
# 12501
# 12501

#%%

try:
    os.mkdir('./data/cats-v-dogs')
    os.mkdir('./data/cats-v-dogs/training')
    os.mkdir('./data/cats-v-dogs/testing')
    os.mkdir('./data/cats-v-dogs/training/cats')
    os.mkdir('./data/cats-v-dogs/training/dogs')
    os.mkdir('./data/cats-v-dogs/testing/cats')
    os.mkdir('./data/cats-v-dogs/testing/dogs')
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
 
    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]
 
    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)
 
    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)
 
#%% Training / test splits
 
CAT_SOURCE_DIR = "./data/PetImages/Cat/"
TRAINING_CATS_DIR = "./data/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "./data/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "./data/PetImages/Dog/"
TRAINING_DOGS_DIR = "./data/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "./data/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
# Expected output
# 666.jpg is zero length, so ignoring
# 11702.jpg is zero length, so ignoring


# Checking train/test splits
print(len(os.listdir('./data/cats-v-dogs/training/cats/')))
print(len(os.listdir('./data/cats-v-dogs/training/dogs/')))
print(len(os.listdir('./data/cats-v-dogs/testing/cats/')))
print(len(os.listdir('./data/cats-v-dogs/testing/dogs/')))
# Expected output:
# 11250
# 11250
# 1250
# 1250


#%% Define the model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
]) 
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])



# Prepare for training the model
TRAINING_DIR = "./data/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
 
VALIDATION_DIR = "./data/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))

# Expected Output:
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes.


#%% Setup checkpoints to save the trained model! I'm not sure this is all correct.
checkpoint_path = "./checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


#%% Training
# Note that this may take some time.
history = model.fit_generator(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks = [cp_callback])


#%% Save weights manually. This seems to work!
model.save_weights('./checkpoints/my_checkpoint')




#%% Explore the results
# Hmmm... graphs are not overlaying properly, and there are a few
# commands towards the top that are not working.

%matplotlib inline

#matplotlib <- import("matplotlib", convert = TRUE)
$matplotlib$use("Agg")
import matplotlib
#matplotlib.use('Agg', force=True)

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
 
epochs=range(len(acc)) # Get number of epochs
 
#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.show()
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()
plt.show()


import seaborn as sns
tips = sns.load_dataset("tips")

df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")
plt.show()


#plt.plot(epochs, acc, 'r', "Training Accuracy")
sns.relplot(x="epochs", y="acc", data=tips);

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.show()
