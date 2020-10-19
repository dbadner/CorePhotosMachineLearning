from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import Functions as fn
import cv2


class UIForm:
    Result: int #0 for accept, 1 skip
    OutputPhotoFileName: str
    #shows User Interface form
    #ImagePath: str#tk.StringVar
    def __init__(self):
        self.window = Tk()
        self.window.title("Machine Learning Core Photo Renaming App")
        self.window.iconphoto(False, tk.PhotoImage(file='input\\icon.png'))
        self.window.geometry('1000x600')
        #window.geometry('1200x800')
        # window.configure(background="gray")
        #declare stringvars corresponding to entries

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
                              ("Units:", self.Unitsstr, defW, defS), ("Wet / dry:", self.WetDrystr, defW, defS),
                              ("Suffix text:", self.SuffixTextstr, defW, defS)]
        #, ("FILE NAME:", self.FileNamestr, 50)]

        # = [tk.StringVar() for i in range(8)]
       #for i,j in stringVarList:
        #    j = tk.StringVar()

        #label showing
        self.photoNameLabel = ttk.Label(self.window, text="Input photo file name: ")
        self.photoNameLabel.place(x=xdim[0], y=10)

        for n, (txt, obj, W, S) in enumerate(self.stringVarList):
            L = ttk.Label(self.window, text=txt)
            L.place(x=xdim[0], y= ydim[0] + (n*ydim[1]))
            E = ttk.Entry(width=W, textvariable=obj)
            E.place(x=xdim[1], y= ydim[0] + (n*ydim[1]))



        btm = ydim[0] + (len(self.stringVarList) * ydim[1])

        L = ttk.Label(self.window, text="FILE NAME:")
        L.place(x=xdim[0], y=btm)
        self.fileNameEntry = ttk.Entry(width=50, textvariable=self.FileNamestr)
        self.fileNameEntry.place(x=xdim[1], y=btm)

        L = ttk.Label(self.window, text="*Note: Do not include file extension in file name above.")
        L.place(x=xdim[1],y = 25 + btm)

        #for i in range(5):
            #ttk.Entry(width=20).grid(row=i + 1, column=2)
            # ent.pack(side = RIGHT, expand = YES, fill = X)

        B = ttk.Button(text="Accept", command=self.accept_button) #.grid(row=9, column=0)
        B.place(x = xdim[0], y = btm + 70)
        B2 = ttk.Button(text="Skip", command=self.skip_button) #.grid(row=9, column=1)
        B2.place(x = xdim[1], y = btm + 70)

        #show images:
        L = ttk.Label(self.window, text="Core photograph:")
        L.place(x=xdim[2], y=10)
        self.canvas1 = Canvas(self.window, height=250)  # dims[0], width = dims[1])
        self.canvas1.place(x=xdim[2], y=10 + 20)
        L = ttk.Label(self.window, text="Whiteboard found:")
        L.place(x=xdim[2], y=ydim[2])
        self.canvas2 = Canvas(self.window, height=250)  # dims[0], width = dims[1])
        self.canvas2.place(x=xdim[2], y=ydim[2] + 20)


        #store in tuple for reference later
        #self.canvasesTup = (self.canvas1, self.canvas2)

        #open image in CV2 format
        image = cv2.imread(r'input\\RC635_166.06-172.04_m_wet.JPG')
        imgTK = self.imageIntoCanvas(image) #place image in canvas
        self.canvas1.create_image(0, 0, anchor=NW, image=imgTK)



        #add trace to entry fields
        for txt, obj, W, S in self.stringVarList:
            obj.trace_add('write', self.update_FName)

        self.window.mainloop()


    def imageIntoCanvas(self, image):
        #function takes image in CV2 format, converts to PIL image, and places in canvas identified by canvind
        # image = Image.open(r'input\\RC635_166.06-172.04_m_wet.JPG')
        #wb = hw.WBImage("")
        image = fn.ResizeImage(image, 600, 250) #resize image, pass max x and y dimensions
        dims = image.shape
        impil = Image.fromarray(image)  # make PIL image
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


    def test_function(self):
        xxx = 1

    def browse_button(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.setPath(filename)

    def setPath(self, word):
        self.ImagePath.set(word)

    def accept_button(self):
        self.Result = 0
        self.OutputPhotoFileName = ""

    def skip_button(self):
        self.Result = 1
        self.OutputPhotoFileName = ""
            #self.window.destroy()


objUI = UIForm()
# https://www.tutorialspoint.com/simple-registration-form-using-python-tkinter

"""
        self.BHIDstr = tk.StringVar()
        self.PrefixTextstr = tk.StringVar()
        self.DepthFromstr = tk.StringVar()
        self.DepthTostr = tk.StringVar()
        self.Unitsstr = tk.StringVar()
        self.WetDrystr = tk.StringVar()
        self.SuffixTextstr = tk.StringVar()
        self.FileNamestr = tk.StringVar()

        #labels
        ttk.Label(self.window, text="Borehole ID:").grid(row=1, column=0)
        ttk.Label(self.window, text="Prefix text:").grid(row=2, column=0)
        ttk.Label(self.window, text="Depth from:").grid(row=3, column=0)
        ttk.Label(self.window, text="Depth to:").grid(row=4, column=0)
        ttk.Label(self.window, text="Units:").grid(row=5, column=0)
        ttk.Label(self.window, text="Wet / dry:").grid(row=6, column=0)
        ttk.Label(self.window, text="Suffix text:").grid(row=7, column=0)
        ttk.Label(self.window, text="FILE NAME:").grid(row=8, column=0)

        #textbox entries
        ttk.Entry(width=20, textvariable=self.BHIDstr).grid(row=1, column=1)
        ttk.Entry(width=20, textvariable=self.PrefixTextstr).grid(row=2, column=1)
        ttk.Entry(width=20, textvariable=self.DepthFromstr).grid(row=3, column=1)
        ttk.Entry(width=20, textvariable=self.DepthTostr).grid(row=4, column=1)
        ttk.Entry(width=20, textvariable=self.Unitsstr).grid(row=5, column=1)
        ttk.Entry(width=20, textvariable=self.WetDrystr).grid(row=6, column=1)
        ttk.Entry(width=20, textvariable=self.SuffixTextstr).grid(row=7, column=1)
        self.fileNameEntry = ttk.Entry(width=40, textvariable=self.FileNamestr).grid(row=8, column=1)
        """