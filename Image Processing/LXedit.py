from tkinter import *
from tkinter import filedialog, NW
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np


class App:
    def __init__(self, window, window_title, window_size="1280x800"):
        self.window = window
        self.window.title(window_title)
        self.window.geometry(window_size)
        self.window.option_add('*tearOff', FALSE)

        # Init panel for display image
        self.rightPanel = None  # panelA -> original image
        self.leftPanel = None   # panelB -> edge map
        self.histogramFigure = None
        self.histogramCanvas = None

        self.menu = Menu(self.window)
        self.fileMenu = Menu(self.menu)
        self.editMenu = Menu(self.menu)
        self.basicMenu = Menu(self.menu)
        self.brightMenu = Menu(self.menu)
        self.colorMenu = Menu(self.menu)
        self.RGBMenu = Menu(self.menu)
        self.GrayMenu = Menu(self.menu)
        self.sizeMenu = Menu(self.menu)
        self.filterMenu = Menu(self.menu)
        self.samplingMenu = Menu(self.menu)
        self.equalizeMenu = Menu(self.menu)
        self.quantizationMenu = Menu(self.menu)

        self.menu.add_cascade(label="File", menu=self.fileMenu)
        self.menu.add_cascade(label="Edit", menu=self.editMenu)

        self.fileMenu.add_cascade(label="Open File", command=self.open_Image)
        self.fileMenu.add_cascade(
            label="Reset Image", command=self.resetEditing)
        self.fileMenu.add_cascade(label="Quit", command=self.window.destroy)

        self.editMenu.add_cascade(label="Basic Op.", menu=self.basicMenu)
        self.editMenu.add_cascade(label="Brightness Op.", menu=self.brightMenu)
        self.editMenu.add_cascade(
            label="Equalize Hist.", menu=self.equalizeMenu)
        self.editMenu.add_cascade(
            label="Quantization Op.", command=self.quantization)
        self.editMenu.add_cascade(label="Sampling Op.", menu=self.samplingMenu)
        self.editMenu.add_cascade(
            label="Grayscale Op.", command=self.grayscaleImg)
        self.editMenu.add_cascade(label="Klise Op.", command=self.negativeImg)
        self.editMenu.add_cascade(label="Filter Op.", menu=self.filterMenu)

        self.basicMenu.add_cascade(label="Color", menu=self.colorMenu)
        self.basicMenu.add_cascade(label="Size", menu=self.sizeMenu)

        self.brightMenu.add_cascade(
            label="Increase Brightness", command=lambda: self.increase_brightness())
        self.brightMenu.add_cascade(
            label="Decrease Brightness", command=lambda: self.decrease_brightness())

        self.equalizeMenu.add_cascade(
            label="RGB", command=lambda: self.equalizeHistogramRGB())
        self.equalizeMenu.add_cascade(
            label="Grayscale", command=lambda: self.equalizeHistogramGray())

        self.quantizationMenu.add_cascade(
            label="Quantization RGB", command=lambda: self.quantization)
        # self.quantizationMenu.add_cascade(
        #     label="Quantization Gray", command=lambda: self.quantizationGray)

        self.samplingMenu.add_cascade(
            label="RGB", command=lambda:self.downSampling())
        self.samplingMenu.add_cascade(
            label="Gray", command=lambda:self.downSamplingGray())

        self.filterMenu.add_cascade(
            label="Low-Pass", command=self.lowPassFilter)
        self.filterMenu.add_cascade(
            label="High-Pass", command=self.highPassFilter)
        self.filterMenu.add_cascade(
            label="Band-Pass", command=self.bandPassFilter)

        self.colorMenu.add_cascade(label="RGB", menu=self.RGBMenu)
        self.colorMenu.add_cascade(label="Gray", menu=self.GrayMenu)

        self.sizeMenu.add_cascade(label="Resize +", command=self.increaseSize)
        self.sizeMenu.add_cascade(label="Resize -", command=self.decreaseSize)

        self.RGBMenu.add_cascade(label="Add Color", command=lambda: self.addColor())
        self.RGBMenu.add_cascade(label="Subs Color", command=lambda: self.subColor())

        self.GrayMenu.add_cascade(label="Add Gray", command=lambda: self.addColorGray())
        self.GrayMenu.add_cascade(label="Subs Gray", command=lambda: self.subColorGray())


        self.window['menu'] = self.menu

        self.window.mainloop()

    def openFile(self):
        fileDir = filedialog.askopenfilename(title="Open an Image", filetypes=[(
            'Image files', '*.jpg *.jpeg *.jfif *.png *.bmp *.tiff *.svg *.gif')])
        return fileDir

    def open_Image(self, size=[625, 625]):
        fileDir = self.openFile()                                       # Open the file
        # Save the file directory
        self.currentFileDir = fileDir
        self.img = cv2.cvtColor(cv2.imread(
            fileDir), cv2.COLOR_BGR2RGB)  # Read the image
        self.imgEdit = self.img                                         # Save the image
        # Convert the image to PIL format
        self.photo = Image.fromarray(self.img)
        # Resize the image
        self.photo.thumbnail(size, Image.ANTIALIAS)
        # Convert the image to Tk format
        self.photo = ImageTk.PhotoImage(image=self.photo)

        # If the left panel and right panel are not None
        if(self.leftPanel != None and self.rightPanel != None):
            # Display the image on the left panel
            self.leftPanel.configure(image=self.photo)
            # Save the image on the left panel
            self.leftPanel.image = self.photo
            # Display the image on the right panel
            self.rightPanel.configure(image=self.photo)
            # Save the image on the right panel
            self.rightPanel.image = self.photo
            # Remove the histogram panel
            self.histogramCanvas.get_tk_widget().grid_forget()
            # Set the subplot to 221
            subplot = 221
            self.histogramFigure = plt.Figure(
                figsize=(4, 4), dpi=100)   # Create a figure
            # For each color
            for i, color in enumerate(['r', 'g', 'b']):
                # Plot the histogram of the image
                self.histogramFigure.add_subplot(subplot).plot(cv2.calcHist(
                    [self.imgEdit], [i], None, [256], [0, 256]), color=color)
                subplot = subplot + 1
            self.histogramCanvas = FigureCanvasTkAgg(
                self.histogramFigure, self.window)  # Create a canvas
            self.histogramCanvas.get_tk_widget().grid(
                row=1, column=0, sticky=NW)       # Display the histogram panel
        else:
            # Display the image on the left panel
            self.leftPanel = Label(self.window, image=self.photo)
            # Save the image on the left panel
            self.leftPanel.image = self.photo
            # Display the image on the left panel
            self.leftPanel.grid(row=0, column=0, sticky=NW)
            # Display the image on the right panel
            self.rightPanel = Label(self.window, image=self.photo)
            # Save the image on the right panel
            self.rightPanel.image = self.photo
            # Display the image on the right panel
            self.rightPanel.grid(row=0, column=1, sticky=NW)
            self.histogramFigure = plt.figure(
                figsize=(4, 4), dpi=100)   # Create a figure
            self.histogramCanvas = FigureCanvasTkAgg(
                self.histogramFigure, self.window)  # Create a canvas
            # Set the subplot to 221
            subplot = 221
            self.histogramFigure = plt.figure(
                figsize=(4, 4), dpi=100)   # Create a figure
            # For each color
            for i, color in enumerate(['r', 'g', 'b']):
                # Plot the histogram of the image
                self.histogramFigure.add_subplot(subplot).plot(cv2.calcHist(
                    [self.imgEdit], [i], None, [256], [0, 256]), color=color)
                subplot = subplot + 1
            self.histogramCanvas = FigureCanvasTkAgg(
                self.histogramFigure, self.window)  # Create a canvas
            self.histogramCanvas.get_tk_widget().grid(
                row=1, column=0, sticky=NW)       # Display the histogram panel


    def downSampling(self):        
        realWidth = self.imgEdit.shape[1]
        realHeight = self.imgEdit.shape[0]
        realDimensions = (realWidth, realHeight)
        width = int(self.imgEdit.shape[1]*15/100)
        height = int(self.imgEdit.shape[0]*15/100)
        dimension = (width, height)
        imgDownSampling = cv2.resize(self.imgEdit, dimension, interpolation=cv2.INTER_AREA)
        self.imgEdit = cv2.resize(imgDownSampling, realDimensions, interpolation=cv2.INTER_AREA)
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogram(self.imgEdit)

    # Sampling gray of the image
    def downSamplingGray(self):
        # Get the width of the image
        realWidth = self.imgEdit.shape[1]
        # Get the height of the image
        realHeight = self.imgEdit.shape[0]
        # Get the dimensions of the image
        realDimensions = (realWidth, realHeight)
        # Get the width of the new image
        width = int(self.imgEdit.shape[1]*15/100)
        # Get the height of the new image
        height = int(self.imgEdit.shape[0]*15/100)
        # Get the dimensions of the new image
        dimension = (width, height)
        # Resize the image
        imgDownSampling = cv2.resize(self.imgEdit, dimension, interpolation=cv2.INTER_AREA)
        self.imgEdit = cv2.resize(imgDownSampling, realDimensions, interpolation=cv2.INTER_AREA)
        # Display the histogram of the new image
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramGray(self.imgEdit)

    def quantization(self):
        height, width = self.imgEdit.shape[0], self.imgEdit.shape[1]
        new_img = np.zeros((height, width, 3), np.uint8)

        # Image quantization operation, The quantification level is 2
        for i in range(height):
            for j in range(width):
                for k in range(3):  # Correspondence BGR Three channel
                    if self.imgEdit[i, j][k] < 128:
                        gray = 0
                    else:
                        gray = 129
                    new_img[i, j][k] = np.uint8(gray)
        self.imgEdit = new_img
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramRGB(self.imgEdit)


    # Increase and Decrease size image
    def increaseSize(self):
        # Get the width of the image
        realWidth = self.imgEdit.shape[1]
        # Get the height of the image
        realHeight = self.imgEdit.shape[0]
        # Get the dimensions of the image
        realDimensions = (realWidth, realHeight)
        # Get the width of the new image
        width = int(self.imgEdit.shape[1]*1.5)
        # Get the height of the new image
        height = int(self.imgEdit.shape[0]*1.5)
        # Get the dimensions of the new image
        dimension = (width, height)
        # Resize the image
        imgIncreaseSize = cv2.resize(
            self.imgEdit, dimension, interpolation=cv2.INTER_AREA)
        self.imgEdit = cv2.resize(
            imgIncreaseSize, realDimensions, interpolation=cv2.INTER_AREA)
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogram(self.imgEdit)

    def decreaseSize(self):
        # Get the width of the image
        realWidth = self.imgEdit.shape[1]
        # Get the height of the image
        realHeight = self.imgEdit.shape[0]
        # Get the dimensions of the image
        realDimensions = (realWidth, realHeight)
        # Get the width of the new image
        width = int(self.imgEdit.shape[1]*0.5)
        # Get the height of the new image
        height = int(self.imgEdit.shape[0]*0.5)
        # Get the dimensions of the new image
        dimension = (width, height)
        # Resize the image
        imgDecreaseSize = cv2.resize(
            self.imgEdit, dimension, interpolation=cv2.INTER_AREA)
        self.imgEdit = cv2.resize(
            imgDecreaseSize, realDimensions, interpolation=cv2.INTER_AREA)
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogram(self.imgEdit)


    # Coloring image operation
    def addColor(self):
        # Get the height and width of the image
        height, width = self.imgEdit.shape[0], self.imgEdit.shape[1]
        # Create a new image
        new_img = np.zeros((height, width, 3), np.uint8)

        # Image add color operation
        for i in range(height):                         # For each row
            for j in range(width):                      # For each column
                for k in range(3):                      # Correspondence BGR Three channel
                    new_img[i, j][k] = np.uint8(
                        self.imgEdit[i, j][k] + 50)     # Add 50 to the pixel
        self.imgEdit = new_img                          # Set the new image to the image
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramRGB(self.imgEdit)

    def subColor(self):
        # Get the height and width of the image
        height, width = self.imgEdit.shape[0], self.imgEdit.shape[1]
        # Create a new image
        new_img = np.zeros((height, width, 3), np.uint8)

        # Image sub color operation
        for i in range(height):                         # For each row
            for j in range(width):                      # For each column
                for k in range(3):                      # Correspondence BGR Three channel
                    new_img[i, j][k] = np.uint8(
                        self.imgEdit[i, j][k] - 50)     # Sub 50 to the pixel
        self.imgEdit = new_img                          # Set the new image to the image
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramRGB(self.imgEdit)


    # Add color grayscale image operation
    def addColorGray(self):
        # Get the height and width of the image
        height, width = self.imgEdit.shape[0], self.imgEdit.shape[1]
        # Create a new image
        new_img = np.zeros((height, width, 3), np.uint8)

        # Image add color operation
        for i in range(height):                         # For each row
            for j in range(width):                      # For each column
                new_img[i, j] = np.uint8(
                    self.imgEdit[i, j] + 50)            # Add 50 to the pixel
        self.imgEdit = new_img                          # Set the new image to the image
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramGray(self.imgEdit)

    # Sub color grayscale image operation
    def subColorGray(self):
        # Get the height and width of the image
        height, width = self.imgEdit.shape[0], self.imgEdit.shape[1]
        # Create a new image
        new_img = np.zeros((height, width, 3), np.uint8)

        # Image sub color operation
        for i in range(height):                         # For each row
            for j in range(width):                      # For each column
                new_img[i, j] = np.uint8(
                    self.imgEdit[i, j] - 50)           # Sub 50 to the pixel
        self.imgEdit = new_img                          # Set the new image to the image
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramGray(self.imgEdit)

    # RGB increase and decrease brightness
    def increase_brightness(self, value=30):
        # Convert the image to HSV
        hsv = cv2.cvtColor(self.imgEdit, cv2.COLOR_RGB2HSV)
        # Split the HSV image
        h, s, v = cv2.split(hsv)

        lim = 255 - value                        # Get the limit value
        # If the pixel is greater than the limit value, set the pixel to 255
        v[v > lim] = 255
        # If the pixel is less than or equal to the limit value, increase the pixel by the value
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))         # Merge the HSV image
        # Convert the HSV image to RGB
        self.imgEdit = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramRGB(self.imgEdit)

    def decrease_brightness(self, value=30):
        hsv = cv2.cvtColor(self.imgEdit, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        lim = 0 + value                        # Get the limit value
        # If the pixel is greater than the limit value, decrease the pixel by the value
        v[v > lim] -= value
        # If the pixel is less than or equal to the limit value, set the pixel to 0
        v[v <= lim] = 0
        # v[v < lim] = 0
        # v[v >= lim] -= value

        final_hsv = cv2.merge((h, s, v))       # Merge the HSV image
        # Convert the HSV image to RGB
        self.imgEdit = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        # Display the histogram of the image
        self.displayHistogramRGB(self.imgEdit)

    # Gray increase and decrease brightness
    def increase_gray_brightness(self, value=30):
        # Convert the image to gray
        gray = cv2.cvtColor(self.imgEdit, cv2.COLOR_RGB2GRAY)
        # If the pixel is greater than the limit value, set the pixel to 255
        gray[gray > value] = 255
        # If the pixel is less than or equal to the limit value, increase the pixel by the value
        gray[gray <= value] += value
        # Convert the gray image to RGB
        self.imgEdit = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # Display the histogram of the image
        self.displayHistogramRGB(self.imgEdit)

    def decrease_gray_brightness(self, value=30):
        # Convert the image to gray
        gray = cv2.cvtColor(self.imgEdit, cv2.COLOR_RGB2GRAY)
        # If the pixel is greater than the limit value, decrease the pixel by the value
        gray[gray > value] -= value
        # If the pixel is less than or equal to the limit value, set the pixel to 0
        gray[gray <= value] = 0
        # Convert the gray image to RGB
        self.imgEdit = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # Display the histogram of the image
        self.displayHistogramRGB(self.imgEdit)


    def resetEditing(self):
        # Read the image from the current file directory
        self.imgEdit = cv2.cvtColor(cv2.imread(
            self.currentFileDir), cv2.COLOR_BGR2RGB)
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramRGB(self.imgEdit)


    def negativeImg(self):
        self.imgEdit = cv2.bitwise_not(self.imgEdit)    # Invert the image
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramRGB(self.imgEdit)


    def grayscaleImg(self):
        # Convert the image to gray
        self.imgEdit = cv2.cvtColor(self.imgEdit, cv2.COLOR_RGB2GRAY)
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramGray(self.imgEdit)


    def lowPassFilter(self):
        matrixKernel = np.ones((5, 5), np.float32) / \
            25   # Create a matrix kernel
        self.imgEdit = cv2.filter2D(
            self.imgEdit, -1, matrixKernel)   # Apply the filter
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramRGB(self.imgEdit)


    def highPassFilter(self):
        # Create a matrix kernel
        matrixKernel = np.array([[0.0, -1.0, 0.0],
                                 [-1.0, 4.0, -1.0],
                                 [0.0, -1.0, 0.0]])
        matrixKernel = matrixKernel / \
            (np.sum(matrixKernel) if np.sum(matrixKernel)
             != 0 else 1)    # Normalize the kernel

        self.imgEdit = cv2.filter2D(
            self.imgEdit, -1, matrixKernel)   # Apply the filter

        # Display the image on the right panel
        self.displayOnRightPanel(self.imgEdit)
        # Display the histogram of the image
        self.displayHistogramRGB(self.imgEdit)


    def bandPassFilter(self):
        # Create a matrix kernel
        matrixKernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        matrixKernel = matrixKernel / \
            (np.sum(matrixKernel) if np.sum(matrixKernel)
             != 0 else 1)    # Normalize the kernel

        self.imgEdit = cv2.filter2D(
            self.imgEdit, -1, matrixKernel)   # Apply the filter

        # print ("bandPassFilter")
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramRGB(self.imgEdit)


    def displayHistogram(self, image):
        self.histogramCanvas.get_tk_widget().grid_forget()
        subplot = 221

        self.histogramFigure = plt.Figure(figsize=(4,4), dpi=100)

        for i, color in enumerate(['r', 'g', 'b']):
            self.histogramFigure.add_subplot(subplot).plot(cv2.calcHist([image],[i],None,[256],[0,256]), color = color)
            subplot = subplot + 1
        
        self.histogramCanvas = FigureCanvasTkAgg(self.histogramFigure, self.window)
        self.histogramCanvas.get_tk_widget().grid(row=1, column=0, sticky=NW)


    # RGB and Gray Equalize Histograms
    def equalizeHistogramRGB(self):
        # Split the image into three channels
        channel = cv2.split(self.imgEdit)
        eq_channel = []                     # Create a new list
        for chn, color in zip(channel, ['R', 'G', 'B']):    # For each channel
            # Apply equalizeHist()
            eq_channel.append(cv2.equalizeHist(chn))
        # Merge the channels
        self.imgEdit = cv2.merge(eq_channel)
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramRGB(self.imgEdit)

    def equalizeHistogramGray(self):
        # Apply equalizeHist()
        self.imgEdit = cv2.equalizeHist(self.imgEdit)
        # Display the image and the histogram
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogramGray(self.imgEdit)


    # Display RGB and gray histograms
    def displayHistogramRGB(self, image):
        # Remove the histogram canvas from the grid
        self.histogramCanvas.get_tk_widget().grid_forget()
        # Set the subplot to the first row, first column
        subplot = 221

        self.histogramFigure = plt.Figure(
            figsize=(4, 4), dpi=100)   # Create a new figure

        # For each channel
        for i, color in enumerate(['r', 'g', 'b']):
            # Plot the histogram
            self.histogramFigure.add_subplot(subplot).plot(
                cv2.calcHist([image], [i], None, [256], [0, 256]), color=color)
            subplot = subplot + 1   # Increase the subplot number

        self.histogramCanvas = FigureCanvasTkAgg(
            self.histogramFigure, self.window)     # Create a new canvas
        self.histogramCanvas.get_tk_widget().grid(
            row=1, column=0, sticky=NW)           # Add the canvas to the grid


    def displayHistogramGray(self, image):
        # Remove the histogram canvas from the grid
        self.histogramCanvas.get_tk_widget().grid_forget()
        # Set the subplot to the first row, first column
        subplot = 221

        # Create a new figure
        self.histogramFigure = plt.Figure(figsize=(4, 4), dpi=100)

        # Plot the histogram
        self.histogramFigure.add_subplot(subplot).plot(
            cv2.calcHist([image], [0], None, [256], [0, 256]), color='gray')
        subplot = subplot + 1   # Increase the subplot number

        # Create a new canvas and add canvar to the grid
        self.histogramCanvas = FigureCanvasTkAgg(
            self.histogramFigure, self.window)
        self.histogramCanvas.get_tk_widget().grid(row=1, column=0, sticky=NW)


    def displayOnRightPanel(self, image):
        # Convert the image to a PIL image
        self.photo = Image.fromarray(image)
        size = [625, 625]                       # Set the size of the image
        self.photo.thumbnail(size, Image.ANTIALIAS)         # Resize the image
        # Convert the image to a Tkinter image
        self.photo = ImageTk.PhotoImage(image=self.photo)
        # Display the image on the right panel
        self.rightPanel.configure(image=self.photo)
        # Keep a reference to the image
        self.rightPanel.image = self.photo

    # # Quantization of the image
    # def quantizationGray(self):
    #     # Get the height and width of the image
    #     height, width = self.imgEdit.shape[0], self.imgEdit.shape[1]
    #     # Create a new image
    #     new_img = np.zeros((height, width, 3), np.uint8)

    #     # Image quantization operation, The quantification level is 2
    #     for i in range(height):                         # For each row
    #         for j in range(width):                      # For each column
    #             for k in range(3):                      # Correspondence BGR Three channel
    #                 if self.imgEdit[i, j][k] < 128:     # If the pixel is less than 128
    #                     gray = 0                        # Set the pixel to 0
    #                 else:                               # If the pixel is greater than 128
    #                     gray = 129
    #                 # Set the pixel to the new image
    #                 new_img[i, j][k] = np.uint8(gray)
    #     self.imgEdit = new_img                          # Set the new image to the image
    #     self.displayOnRightPanel(self.imgEdit)
    #     self.displayHistogramGray(self.imgEdit)

App(Tk(), "LXEdit - Image Process")
