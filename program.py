from tkinter import Tk
from tkinter.filedialog import *

root = Tk()
fileName = ''
guessedShape = StringVar()
actualShape = StringVar()

guessShapeLabel = Label(root, textvariable=guessedShape)
verifyShapeLabel = Label(root, textvariable=actualShape)


def startGUI():
    root.title("Shape Learner")
    root.geometry("250x150")

    batchSizeLabel = Label(root, text="Batch Size")
    epochsLabel = Label(root, text="Epochs")

    batchSize = Entry(root)
    batchSize.insert(END, "128")
    epochs = Entry(root)
    epochs.insert(END, "15")

    batchSizeLabel.pack()
    batchSize.pack()
    epochsLabel.pack()
    epochs.pack()

    openFileButton = Button(root, text="Open File", command=openFiles)
    openFileButton.pack()
    verifyShapeLabel.pack()
    guessShapeLabel.pack()

    mainloop()

# open the UI for choosing and picking a picture
def openFiles():
    fileTypes = [("jpg files", "*.jpg")]
    global fileName
    fileName = askopenfilename(filetypes = fileTypes)
    actualShape.set("Actual Shape: " + verifyShape(fileName))

# read the file we've been given so we can know what shape was opened
def verifyShape(fileOpened):
    lastIndex = fileOpened.rfind('/')
    fileName = fileOpened[lastIndex + 1:]
    shape = ''
    if("circ" in fileName):
        shape = "Circle"
    elif ("rect" in fileName):
        shape = "Rectangle"
    elif ("tri" in fileName):
        shape = "Triangle"
    return shape
    
    
# start the program off by learning and training 
# learn()

# open up our GUI so we can upload an image
startGUI()