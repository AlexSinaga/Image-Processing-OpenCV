from tkinter import *
from tkinter import filedialog, NW
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np

class App:
    def __init__(self, window, window_title, window_size="1920x1080"):
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
        self.brightMenu = Menu(self.menu)
        self.filterMenu = Menu(self.menu)

        self.menu.add_cascade(label="File", menu=self.fileMenu)
        self.menu.add_cascade(label="Edit", menu=self.editMenu)

        self.fileMenu.add_cascade(label="Open File", command=self.open_Image)
        self.fileMenu.add_cascade(label="Reset Image", command=self.resetEditing)
        self.fileMenu.add_cascade(label="Quit", command=self.window.destroy)

        self.editMenu.add_cascade(label="Sampling Op.", command=self.downSampling)
        self.editMenu.add_cascade(label="Quantization Op.", command=self.quantization)
        self.editMenu.add_cascade(label="Brightness Op.", menu=self.brightMenu)
        self.editMenu.add_cascade(label="Klise Op.", command=self.negativeImg)
        self.editMenu.add_cascade(label="Filter Op.", menu=self.filterMenu)
        self.editMenu.add_cascade(label="Equalize Hist.", command=self.equalizeHistogram)
        # self.editMenu.add_cascade(label="Show Histogram", command=self.displayHistogram)

        self.brightMenu.add_cascade(label="Increase Brightness", command=lambda:self.increase_brightness())
        self.brightMenu.add_cascade(label="Decrease Brightness", command=lambda:self.decrease_brightness())

        self.filterMenu.add_cascade(label="Low-Pass", command=self.lowPassFilter)
        self.filterMenu.add_cascade(label="High-Pass", command=self.highPassFilter)
        self.filterMenu.add_cascade(label="Band-Pass", command=self.bandPassFilter)


        self.window['menu'] = self.menu

        self.window.mainloop()

    def openFile(self):
        fileDir = filedialog.askopenfilename(title="Open an Image", filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.svg *.gif')])
        return fileDir

    def open_Image(self, size=[625, 625]):
        fileDir = self.openFile()
        self.currentFileDir = fileDir
        self.img = cv2.cvtColor(cv2.imread(fileDir), cv2.COLOR_BGR2RGB)
        self.imgEdit = self.img
        self.photo = Image.fromarray(self.img)
        self.photo.thumbnail(size, Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image=self.photo)

        if(self.leftPanel != None and self.rightPanel != None):
            self.leftPanel.configure(image=self.photo)
            self.leftPanel.image = self.photo
            self.rightPanel.configure(image=self.photo)
            self.rightPanel.image = self.photo
            self.histogramCanvas.get_tk_widget().grid_forget()
            subplot=221
            self.histogramFigure = plt.Figure(figsize=(4,4), dpi=100)
            for i, color in enumerate(['r', 'g', 'b']):
                self.histogramFigure.add_subplot(subplot).plot(cv2.calcHist([self.imgEdit],[i],None,[256],[0,256]), color=color)
                subplot = subplot + 1
            self.histogramCanvas = FigureCanvasTkAgg(self.histogramFigure, self.window)
            self.histogramCanvas.get_tk_widget().grid(row=1, column=0, sticky=NW)
        else:
            self.leftPanel = Label(self.window, image=self.photo)
            self.leftPanel.image = self.photo
            self.leftPanel.grid(row=0, column=0, sticky=NW)
            self.rightPanel = Label(self.window, image=self.photo)
            self.rightPanel.image = self.photo
            self.rightPanel.grid(row=0, column=1, sticky=NW)
            self.histogramFigure = plt.figure(figsize=(4,4), dpi=100)
            self.histogramCanvas = FigureCanvasTkAgg(self.histogramFigure, self.window)
            subplot=221
            self.histogramFigure = plt.figure(figsize=(4,4), dpi=100)
            for i, color in enumerate(['r', 'g', 'b']):
                self.histogramFigure.add_subplot(subplot).plot(cv2.calcHist([self.imgEdit], [i], None, [256], [0,256]), color= color)
                subplot = subplot + 1
            self.histogramCanvas = FigureCanvasTkAgg(self.histogramFigure, self.window)
            self.histogramCanvas.get_tk_widget().grid(row=1, column=0, sticky=NW)

    def downSampling(self):        
        realWidth = self.imgEdit.shape[1]
        realHeight = self.imgEdit.shape[0]
        realDimensions = (realWidth, realHeight)
        width = int(self.imgEdit.shape[1]*15/100)
        height = int(self.imgEdit.shape[0]*15/100)
        dimension = (width, height)
        imgDownSampling = cv2.resize(self.imgEdit, dimension, interpolation=cv2.INTER_AREA)
        self.imgEdit = cv2.resize(imgDownSampling, realDimensions, interpolation=cv2.INTER_AREA)
        self.displayHistogram(self.imgEdit)

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
        self.displayHistogram(self.imgEdit)

    def increase_brightness(self, value=30):
        hsv = cv2.cvtColor(self.imgEdit, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        self.imgEdit = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogram(self.imgEdit)

    def decrease_brightness(self, value=30):
        hsv = cv2.cvtColor(self.imgEdit, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        lim = 0 + value
        v[v > lim] -= value
        v[v <= lim] = 0
        # v[v < lim] = 0
        # v[v >= lim] -= value

        final_hsv = cv2.merge((h, s, v))
        self.imgEdit = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        self.displayHistogram(self.imgEdit)

    def resetEditing(self):
        self.imgEdit = cv2.cvtColor(cv2.imread(
            self.currentFileDir), cv2.COLOR_BGR2RGB)
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogram(self.imgEdit)

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w*r), height)

        else:
            r = width / float(w)
            dim = (width, int(h*r))

        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def negativeImg(self):
        self.imgEdit = cv2.bitwise_not(self.imgEdit)
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogram(self.imgEdit)

    def lowPassFilter(self):
        matrixKernel = np.ones((5,5),np.float32)/25
        self.imgEdit = cv2.filter2D(self.imgEdit,-1,matrixKernel)

        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogram(self.imgEdit)

    def highPassFilter(self):
        matrixKernel = np.array([[0.0, -1.0, 0.0], 
                        [-1.0, 4.0, -1.0],
                        [0.0, -1.0, 0.0]])
        matrixKernel = matrixKernel/(np.sum(matrixKernel) if np.sum(matrixKernel)!=0 else 1)

        self.imgEdit = cv2.filter2D(self.imgEdit,-1,matrixKernel)

        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogram(self.imgEdit)

    def bandPassFilter(self):
        matrixKernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        matrixKernel = matrixKernel/(np.sum(matrixKernel) if np.sum(matrixKernel)!=0 else 1)

        self.imgEdit = cv2.filter2D(self.imgEdit,-1,matrixKernel)
        
        # print ("bandPassFilter")
        self.displayOnRightPanel(self.imgEdit)
        self.displayHistogram(self.imgEdit)

    def equalizeHistogram(self):
        channel = cv2.split(self.editedImg)
        eq_channel = []
        for chn, color in zip(channel, ['R', 'G', 'B']):
            eq_channel.append(cv2.equalizeHist(chn))
        self.editedImg = cv2.merge(eq_channel)
        self.displayOnRightPanel(self.editedImg)
        self.setHistogram(self.editedImg)

    def displayHistogram(self, image):
        self.histogramCanvas.get_tk_widget().grid_forget()
        subplot = 221

        self.histogramFigure = plt.Figure(figsize=(4,4), dpi=100)

        for i, color in enumerate(['r', 'g', 'b']):
            self.histogramFigure.add_subplot(subplot).plot(cv2.calcHist([image],[i],None,[256],[0,256]), color = color)
            subplot = subplot + 1
        
        self.histogramCanvas = FigureCanvasTkAgg(self.histogramFigure, self.window)
        self.histogramCanvas.get_tk_widget().grid(row=1, column=0, sticky=NW)

    def displayOnRightPanel(self, image):
        self.photo = Image.fromarray(image)
        size = [625, 625]
        self.photo.thumbnail(size, Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image=self.photo)
        self.rightPanel.configure(image=self.photo)
        self.rightPanel.image = self.photo

App(Tk(), "LXEdit - Image Process", "1280x800")
