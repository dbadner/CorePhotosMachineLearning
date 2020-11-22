from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
import Functions as fn
import cv2
import os
import shutil
import ctypes
import re


class UIForm:
    #Result: int #0 for accept, 1 skip
    #OutputPhotoFileName: str #output photo file name
    InputDir: str  # whiteboard image input directory (output directory of detectron)
    OutputDir: str  # final image output directory
    CFOutputList: list #list of tuples output from the OCR classification script that is returned from this function and fed
    SkipML: bool #true if machine learning components have been skipped. false if machine learning has been performed
    #into the user form
    #tuple parameters: [image filename, image filepath, whiteboard output image filepath, annotated output image filepath,
    #annotated whiteboard output image filepath, classified depthFrom, classified depthTo, classified wet/dry,
    #depthFrom probability as %, depthTo probability as %
    # whiteboard output image filepath, # annotated output image filepath)
    OutputListInd: int
    imgTK: object
    imgTK2: object

    #shows User Interface form
    #ImagePath: str#tk.StringVar
    def __init__(self, inputdir, outputdir, cfoutputList, skipML):
        #initialize variables related to image input/output
        self.InputDir = inputdir
        self.OutputDir = outputdir
        self.CFOutputList = cfoutputList
        self.OutputListInd = 0 #initialize index of the WBOutputList to start at (will iterate through images interactively)
        self.SkipML = skipML

        #initialize variables that save previous depths
        self.PrevDepthFrom = ""
        self.PrevDepthTo = ""

        self.window = Tk()
        self.window.title("Machine Learning Core Photo Renaming App")
        self.window.iconphoto(False, tk.PhotoImage(file='input/icon.png'))
        self.window.geometry('1100x600')
        self.window.bind('<Return>', self.enter_run)
        #window.geometry('1200x800')
        # window.configure(background="gray")
        #declare stringvars corresponding to entries

        self.InputPhotoName = tk.StringVar()
        self.BHIDstr = tk.StringVar()
        self.PrefixTextstr = tk.StringVar()
        self.DepthFromstr = tk.StringVar()
        self.DepthTostr = tk.StringVar()
        self.Unitsstr = tk.StringVar()
        self.WetDrystr = tk.StringVar()
        self.SuffixTextstr = tk.StringVar()
        self.FileNamestr = tk.StringVar()

        #form sizing variables
        ydim = (40, 35, 290)
        xdim = (10, 100, 500)

        defW = 20
        defS = "_"

        self.stringVarList = [("Borehole ID:", self.BHIDstr, defW, defS),
                              ("Prefix text:", self.PrefixTextstr, defW, defS),
                              ("Depth from:", self.DepthFromstr, defW, "-"),
                              ("Depth to:", self.DepthTostr, defW, defS),
                              ("Units:", self.Unitsstr, defW, defS), ("Dry / wet:", self.WetDrystr, defW, defS),
                              ("Suffix text:", self.SuffixTextstr, defW, defS)]
        #, ("FILE NAME:", self.FileNamestr, 50)]

        # = [tk.StringVar() for i in range(8)]
       #for i,j in stringVarList:
        #    j = tk.StringVar()

        #label showing
        self.photoNameLabel = ttk.Label(self.window, text="Input photo file name: ", textvariable=self.InputPhotoName)
        self.photoNameLabel.place(x=xdim[0], y=10)

        self.entryList = [] #list of entry objects
        for n, (txt, obj, W, S) in enumerate(self.stringVarList):
            L = ttk.Label(self.window, text=txt)
            L.place(x=xdim[0], y= ydim[0] + (n*ydim[1]))
            E = ttk.Entry(width=W, textvariable=obj)
            E.place(x=xdim[1], y= ydim[0] + (n*ydim[1]))
            self.entryList.append(E)

        btm = ydim[0] + (len(self.stringVarList) * ydim[1])

        L = ttk.Label(self.window, text="FILE NAME:")
        L.place(x=xdim[0], y=btm)
        self.fileNameEntry = ttk.Entry(width=50, textvariable=self.FileNamestr)
        self.fileNameEntry.place(x=xdim[1], y=btm)

        L = ttk.Label(self.window, text="*Note: Do not include file extension in file name above.")
        L.place(x=xdim[1],y = 25 + btm)

        #default units to 'm'
        self.Unitsstr.set("m")



        #for i in range(5):
            #ttk.Entry(width=20).grid(row=i + 1, column=2)
            # ent.pack(side = RIGHT, expand = YES, fill = X)
        #add buttons
        B3 = ttk.Button(text="Use Previous Depths", command=self.ApplyPrevDepths)
        B3.place(x = 235, y = 108)
        B4 = ttk.Button(text="Use Previous Depth To as Depth From", command=self.ApplyPrevDepthTo)
        B4.place(x = 235, y = 108 + ydim[1])

        # add radio buttons for DRY vs WET
        self.dw = IntVar()
        r = ttk.Radiobutton(text = "DRY", variable = self.dw, value=1, command=self.RadioChangeDryWet)
        r.place(x = 245, y = 215)
        r2 = ttk.Radiobutton(text = "WET", variable = self.dw, value=2, command=self.RadioChangeDryWet)
        r2.place(x = 300, y = 215)


        B = ttk.Button(text="Accept", command=self.accept_button) #.grid(row=9, column=0)
        B.place(x = xdim[0], y = btm + 70)
        B2 = ttk.Button(text="Skip", command=self.skip_button) #.grid(row=9, column=1)
        B2.place(x = xdim[1], y = btm + 70)



        #show images:
        #set dimensions of core photo image depending on whether 2 images need to be shown
        cvw = 800
        cvh = 250
        if self.SkipML:
            cvw *= 2
            cvh *= 2
        L = ttk.Label(self.window, text="Whiteboard found:")
        L.place(x=xdim[2], y=ydim[2])
        self.canvas2 = Canvas(self.window, height=cvh, width=cvw)  # dims[0], width = dims[1])
        self.canvas2.place(x=xdim[2], y=ydim[2] + 20)
        L = ttk.Label(self.window, text="Core photograph:")
        L.place(x=xdim[2], y=10)
        self.canvas1 = Canvas(self.window, height=cvh, width=cvw)  # dims[0], width = dims[1])
        self.canvas1.place(x=xdim[2], y=10 + 20)


        #store in tuple for reference later
        #self.canvasesTup = (self.canvas1, self.canvas2)

        #open image in CV2 format
        #image = cv2.imread(r'input/RC635_166.06-172.04_m_wet.JPG')
        # place annotated detectron image into canvas1
        #imgTK = self.imageIntoCanvas(self.WBOutputList[self.OutputListInd[3]])
        #self.canvas1.create_image(0, 0, anchor=NW, image=imgTK)

        #add trace to entry fields
        for txt, obj, W, S in self.stringVarList:
            obj.trace_add('write', self.update_FName)

        self.OCR(self.CFOutputList[0])  # call OCR method to read in first whiteboard image

        self.window.mainloop()

    def OCR(self, cfobj):

        #read in appropriate images
        #first, read in the annotated core photo as CV2, then resize and convert

        #image = cv2.imread(self.WBOutputList[self.OutputListInd][3])
        image = cv2.imread(cfobj.ImgWBAnnoFilePath)
        self.imgTK = self.imageIntoCanvas(image)
        self.canvas1.create_image(0, 0, anchor=NW, image=self.imgTK)

        if not self.SkipML:
            # next, read in whiteboard image
            image2 = cv2.imread(cfobj.ImgAnnoFilePath)
            self.imgTK2 = self.imageIntoCanvas(image2)
            self.canvas2.create_image(0, 0, anchor=NW, image=self.imgTK2)

            # set fields on form
            self.InputPhotoName.set("Input photo file name: " + cfobj.ImgFileName)  # set the label at the top of the form
            self.DepthFromstr.set(self.AddLeadingZeros(cfobj.DepthFrom))
            self.DepthTostr.set(self.AddLeadingZeros(cfobj.DepthTo))
            self.WetDrystr.set(cfobj.WetDry)
            if cfobj.WetDry == "DRY": #set radio button for DRY vs WET
                self.dw.set(1)
            elif cfobj.WetDry == "WET":
                self.dw.set(2)

    def RadioChangeDryWet(self):
        #function executes when radio buttons for dry / wet are selected
        if self.dw.get() == 1: #dry
            self.WetDrystr.set("DRY")
        elif self.dw.get() == 2: #wet
            self.WetDrystr.set("WET")

    def ApplyPrevDepths(self):
        #function applies previous depths from and to when corresponding button is clicked
        self.entryList[2].delete(0, 'end')
        self.entryList[3].delete(0, 'end')
        self.entryList[2].insert(0, self.PrevDepthFrom)
        self.entryList[3].insert(0, self.PrevDepthTo)

    def ApplyPrevDepthTo(self):
        #function applies previous depths to to current depth from when corresponding button is clicked
        self.entryList[2].delete(0, 'end')
        self.entryList[2].insert(0, self.PrevDepthTo)

    def AddLeadingZeros(self, strNum: str):
        #function adds leading zeros to string of numbers if < 4 x 0s
        l = len(strNum)
        strOut = strNum
        left = re.search(r"(.*)\.", strNum)
        if left is not None:
            l = len(left.group(0)[:-1]) #find length of the characters to the left of the "."
            if l < 4:
                strOut = "0"*(4-l) + strNum
        return strOut


    def imageIntoCanvas(self, image):
        #function takes image in CV2 format, converts to PIL image, and places in canvas identified by canvind
        # image = Image.open(r'input/RC635_166.06-172.04_m_wet.JPG')
        #wb = hw.WBImage("")
        cvw = 800
        cvh = 250
        if self.SkipML:
            cvw *= 2
            cvh *= 2
        image = fn.ResizeImage(image, cvw, cvh) #resize image, pass max x and y dimensions
        dims = image.shape
        impil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # make PIL image
        photo = ImageTk.PhotoImage(impil)
        # img = PhotoImage(photo)
        #self.canvasesTup[canvind].create_image(0, 0, anchor=NW, image=photo)
        return photo

    def update_FName(self, var, indx, mode):
        self.fileNameEntry.delete(0,'end')
        fname = ""
        for txt, obj, W, Suf in self.stringVarList:
            strtemp = obj.get()
            if strtemp != "":
                fname += strtemp + Suf
        if len(fname) > 0:#remove final underscore
            fname = fname[0:len(fname)-1]
        self.fileNameEntry.insert(0,fname)
        #self.FileNamestr.set(self.BHIDstr.get()+"TEST")


    def accept_button(self):

        #save current image, and then move onto the next image
        #self.Result = 0
        #self.OutputPhotoFileName = ""
        #save the image as a copy in the output directory
        temp, suffix = os.path.splitext(self.CFOutputList[self.OutputListInd].ImgFileName)
        new_name = self.OutputDir + "/" + self.FileNamestr.get() + suffix
        if os.path.exists(new_name):
            result = ctypes.windll.user32.MessageBoxW(0,"Warning: File "+ self.FileNamestr.get() + suffix + " already exists" +
                    " in output directory. Press OK to overwrite or Cancel to rename.", "Warning", 1)
            if result == 2: return #Cancel
        shutil.copy(self.CFOutputList[self.OutputListInd].ImgFilePath, new_name)
        #save depth from and depth to into class variables
        self.PrevDepthFrom = self.DepthFromstr.get()
        self.PrevDepthTo = self.DepthTostr.get()

        self.reset_fields()
        self.next_photo()

    def enter_run(self, obj):
        # obj is the object returned when the enter key is pressed to enter this function. not used

        self.accept_button()

    def skip_button(self):
        # do not save current image, and then move onto the next image
        #self.Result = 1
        #self.OutputPhotoFileName = ""
        self.reset_fields()
        self.next_photo()
            #self.window.destroy()

    def reset_fields(self):
        #called to clean fields in between photo iterations
        self.DepthFromstr.set("")
        self.DepthTostr.set("")
        self.WetDrystr.set("")

    def next_photo(self):
        self.OutputListInd += 1
        if self.OutputListInd < len(self.CFOutputList):
            self.OCR(self.CFOutputList[self.OutputListInd]) #process the next image
        else: # all photos in directory complete, exit
            ctypes.windll.user32.MessageBoxW(0,"All photos in specified folder location complete. " +
                                             "Press OK to exit the program. ","Notice", 0)
            self.window.destroy()

# https://www.tutorialspoint.com/simple-registration-form-using-python-tkinter
