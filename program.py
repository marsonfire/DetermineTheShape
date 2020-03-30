from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfile
from tkinter.filedialog import askdirectory
from tkinter import ttk
from tkinter import StringVar
from tkinter import Label
from tkinter import Button
from tkinter import mainloop

root = Tk()
guessedShape = StringVar()
actualShape = StringVar()
result = StringVar()
amountCorrectVal = StringVar()
amountOfTestsVal = StringVar()
accuracy = StringVar()

amountCorrect = 0
amountOfTests = 0

train_data_dir = 'train'
validation_data_dir = 'validation'
test_data_dir = 'test/testimages/'

img_width, img_height = 56, 56
nb_train_samples = 300
nb_validation_samples = 60
epochs = 20
batch_size = 30

# path to folder with images
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path + "/"
FILETYPES = [("jpg files", "*.jpg")]
TEMPDIR = "tempDir/"


def startGUI():
    # make sure images format isn't a total mess
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # load the model, if it doesn't exist, well then lets recreate the model
    recreatedModel = False
    try:
        model = load_model('savedModel.h5')
    except:
        # this builds up the model that we use and train on
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3))
        model.add(Activation('sigmoid'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        recreatedModel = True

    # datagens created so we can load our images correctly
    # this is the what we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.5,
        horizontal_flip=True)

    # this is what we will use for validating when training
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # loading in our training images in the correct format
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # loading in our validation images in the correct format
    validation_generator = valid_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # only retrain everything if we've recreated the model
    if(recreatedModel):
        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

        model.save('savedModel.h5')

    # create the GUI
    root.title("Shape Learner")
    root.geometry("250x200")

    retrainModelButton = Button(root, text="Retrain Model", command=lambda: retrainModel(
        train_generator, validation_generator))
    retrainModelButton.pack()

    openFileButton = Button(root, text="Open File",
                            command=lambda: openFiles(train_generator))
    openFileButton.pack()

    openFolderButton = Button(root, text="Open Folder",
                              command=lambda: openFolder(train_generator))
    openFolderButton.pack()

    verifyShapeLabel = Label(root, textvariable=actualShape)
    guessShapeLabel = Label(root, textvariable=guessedShape)
    resultLabel = Label(root, textvariable=result)
    amountCorrectLabel = Label(root, textvariable=amountCorrectVal)
    amountOfTestsLabel = Label(root, textvariable=amountOfTestsVal)
    accuracyLabel = Label(root, textvariable=accuracy)

    verifyShapeLabel.pack()
    guessShapeLabel.pack()
    resultLabel.pack()
    amountCorrectLabel.pack()
    amountOfTestsLabel.pack()
    accuracyLabel.pack()

    mainloop()


def openFiles(train_generator):
    model = load_model('savedModel.h5')

    # open the UI for choosing and picking a picture
    fileName = askopenfile(filetypes=FILETYPES)

    # move the file into a temporary folder since we need to read from a directory
    lastIndex = fileName.name.rfind('/') + 1
    fileName = fileName.name[lastIndex:]
    shutil.move(dir_path + test_data_dir + fileName,
                dir_path + test_data_dir + TEMPDIR + fileName)
    actualShapeVerified = verifyShape(fileName)
    actualShape.set("Actual Shape: " + actualShapeVerified)

    # read in and rescale our selected image
    test_datagen = ImageDataGenerator(rescale=1./255)

    # read in our selected image in the correct format
    test_generator = test_datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42)

    # use the model passed in to predict what the shape is
    pred = model.predict_generator(test_generator, steps=(
        test_generator.n//test_generator.batch_size), verbose=1)
    # whichever image class has the highest weight, that's what we'll use as the prediction
    predicted_class_indices = np.argmax(pred)

    # it's a dictionary, but the keys and values are flip flopped so we want to change them so it's easier to access later
    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    # now it holds the predicted shape!
    predictedShape = labels[predicted_class_indices]

    # checking if the file we read in is correct or not
    if(predictedShape == actualShapeVerified):
        global amountCorrect
        result.set("CORRECT")
        amountCorrect = amountCorrect + 1
        amountCorrectVal.set("Amount Correct: " + str(amountCorrect))
        incrementTests()
    else:
        result.set("WRONG")
        incrementTests()

    guessedShape.set("Guessed Shape: " + predictedShape)

    accuracy.set("Accuracy: " + str(round(amountCorrect/amountOfTests, 4)))

    shutil.move(dir_path + test_data_dir + TEMPDIR +
                fileName, dir_path + test_data_dir + fileName)


def openFolder(train_generator):
    model = load_model('savedModel.h5')

    # open the UI for choosing and picking a picture
    folderName = askdirectory()

    test_datagen = ImageDataGenerator(rescale=1./255)

    # read in our selected image in the correct format
    test_generator = test_datagen.flow_from_directory(
        directory=folderName,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42)

    # use the model passed in to predict what the shapes are
    pred = model.predict_generator(test_generator, steps=(
        test_generator.n//test_generator.batch_size), verbose=1)

    # it's a dictionary, but the keys and values are flip flopped so we want to change them so it's easier to access later
    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    global amountCorrect
    index = 0
    for val in pred:
        predicted_class_indices = np.argmax(val)
        if(index < 10):
            if(predicted_class_indices == 0):
                amountCorrect = amountCorrect + 1
        elif(index < 20):
            if(predicted_class_indices == 1):
                amountCorrect = amountCorrect + 1
        elif(index < 30):
            if(predicted_class_indices == 2):
                amountCorrect = amountCorrect + 1
        index = index + 1

    amountCorrectVal.set("Amount Correct: " + str(amountCorrect))
    global amountOfTests
    amountOfTests = amountOfTests + index
    amountOfTestsVal.set("Amount Tested: " + str(amountOfTests))
    accuracy.set("Accuracy: " + str(round(amountCorrect/amountOfTests, 4)))


def verifyShape(fileOpened):
    # read the file we've been given so we can know what shape was opened
    lastIndex = fileOpened.rfind('/')
    fileName = fileOpened[lastIndex + 1:]
    shape = ''
    if("circ" in fileName):
        shape = "circle"
    elif ("rect" in fileName):
        shape = "rectangle"
    elif ("tri" in fileName):
        shape = "triangle"
    return shape


def incrementTests():
    global amountOfTests
    amountOfTests = amountOfTests + 1
    amountOfTestsVal.set("Amount Tested: " + str(amountOfTests))


def retrainModel(train_generator, validation_generator):
    global amountCorrect
    global amountOfTests
    amountCorrect = 0
    amountOfTests = 0 
    amountCorrectVal.set("Amount Correct: " + str(amountCorrect))
    amountOfTestsVal.set("Amount Tested: " + str(amountOfTests))
    accuracy.set("Accuracy: 0")

    # make sure images format isn't a total mess
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # this builds up the model that we use and train on
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save('savedModel.h5')

    return model

# open up our GUI so we can upload an image
startGUI()
