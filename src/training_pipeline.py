import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt


from keras.applications import InceptionV3
from keras.applications import VGG16

from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Dropout
from keras.layers import MaxPooling2D, GlobalAveragePooling2D


from keras.layers import Input, Average
from plotting_model import plot


def train_image_Classification():

    # Data Augmentaion:
    TRAINING_DIR = "D:\A_Category\Assesment\dataset_256X256\train"
    training_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = "D:\A_Category\Assesment\dataset_256X256\test"
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(256,256),
        class_mode='categorical',
        batch_size=40
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(256,256),
        class_mode='categorical',
        batch_size=40
    )


    # Building Model 1:
    base_model_1 = InceptionV3(
        input_shape = (256,256,3),
        weights = 'imagenet',
        include_top = False
    )

    for layer in base_model_1.layers:
        layer.trainable = False

    x = base_model_1.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(4, activation = 'softmax')(x)
    model1 = Model(inputs= base_model_1.inputs, outputs = predictions)

    # Compile Model_1
    model1.compile(optimizer ='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])


    # Defining checkpoint save the best model during training
    model_filepath = '/model/model_1.h5'
    checkpoint_1 = ModelCheckpoint(
        filepath = model_filepath,
        monitor = 'val_accuracy',
        mode = 'max',
        save_best_only = True,
        verbose = 1
    )

    # Train Model_1
    history1 = model1.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = 5, callbacks = [checkpoint_1]
    )


    # Building Model_2

    base_model_2 = VGG16(
        input_shape = (256,256,3),
        weights = 'imagenet',
        include_top = False
    )

    for layer in base_model_2.layers:
        layer.trainable = False
    x = base_model_2.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(4, activation = 'softmax')(x)
    model2 = Model(inputs= base_model_2.inputs, outputs = predictions)


    # Compile the model
    model2.compile(optimizer ='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])


    # Define the checkpoint to save the best model during training
    model_filepath = '/model/model_2.h5'
    checkpoint_2 = ModelCheckpoint(
        filepath = model_filepath,
        monitor = 'val_accuracy',
        mode = 'max',
        save_best_only = True,
        verbose = 1
    )



    # Train the model_2
    history2 = model2.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = 5, callbacks = [checkpoint_2]
    )

    plot(history1)
    plot(history2)

    # Build ensemble model
    model_1 = load_model('/model/model_1.h5')
    model_1 = Model(inputs = model_1.inputs,
                    outputs = model_1.outputs,
                    name = 'name_of_model_1')


    model_2 = load_model('//model/model_2.h5')
    model_2 = Model(inputs = model_2.inputs,
                    outputs = model_2.outputs,
                    name = 'name_of_model_2')

    models = [model_1, model_2]
    model_input = Input(shape = (256,256,3))
    model_outputs = [model(model_input) for model in models]
    ensemble_output = Average()(model_outputs)
    ensemble_model = Model(inputs = model_input,
                    outputs = ensemble_output,
                    name = 'ensemble')

    # Compile the ensemble_model
    ensemble_model.compile(optimizer ='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

    # Define the checkpoint to save the best model during training
    model_filepath = '/model/ensemble_model.h5'
    checkpoint = ModelCheckpoint(
        filepath = model_filepath,
        monitor = 'val_accuracy',
        mode = 'max',
        save_best_only = True,
        verbose = 1
    )

    # Train the model
    history = ensemble_model.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = 5, callbacks = [checkpoint]
    )

    plot(history)

    return ensemble_model