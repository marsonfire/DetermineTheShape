from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import os
import shutil
from tkinter import Tk
from tkinter.filedialog import askopenfile
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

img_width, img_height = 28, 28
nb_train_samples = 300
nb_validation_samples = 30
epochs = 30
batch_size = 30

# path to folder with images
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path + "/"
print(dir_path + "/")
FILETYPES = [("jpg files", "*.jpg")]
TEMPDIR = "tempDir/"


def startGUI():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

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

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = valid_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    root.title("Shape Learner")
    root.geometry("250x150")
    openFileButton = Button(root, text="Open File", command=lambda: openFiles(model, train_generator))
    openFileButton.pack()

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


def openFiles(model, train_generator):
    # open the UI for choosing and picking a picture
    fileName = askopenfile(filetypes=FILETYPES)
    lastIndex = fileName.name.rfind('/') + 1
    fileName = fileName.name[lastIndex:]
    shutil.move(dir_path + test_data_dir + fileName, dir_path + test_data_dir + TEMPDIR + fileName)
    actualShapeVerified = verifyShape(fileName)
    actualShape.set("Actual Shape: " + actualShapeVerified)

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42)

    pred = model.predict_generator(test_generator, steps=(test_generator.n//test_generator.batch_size), verbose=1)
    predicted_class_indices = np.argmax(pred)

    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    predictedShape = labels[predicted_class_indices]

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

    shutil.move(dir_path + test_data_dir + TEMPDIR + fileName, dir_path + test_data_dir + fileName)


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


# open up our GUI so we can upload an image
startGUI()
