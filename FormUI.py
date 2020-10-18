from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk


class UIForm:
    Result: int #0 for accept, 1 skip
    OutputPhotoFileName: str
    #shows User Interface form
    #ImagePath: str#tk.StringVar
    def __init__(self):
        self.window = Tk()
        self.window.title("Machine Learning Core Photo Renaming App")
        self.window.iconphoto(False, tk.PhotoImage(file='input\\icon.png'))
        self.window.geometry('1200x800')
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

        #for i in range(5):
            #ttk.Entry(width=20).grid(row=i + 1, column=2)
            # ent.pack(side = RIGHT, expand = YES, fill = X)

        ttk.Button(text="Accept", command=self.accept_button).grid(row=9, column=0)
        ttk.Button(text="Skip", command=self.skip_button).grid(row=9, column=1)

        canvas = Canvas(self.window, width = 300, height = 300)
        #canvas.pack()
        img = PhotoImage(file=r'input\\icon.png')
        canvas.create_image(0,0, anchor=NW, image = img)

        self.BHIDstr.trace('w',self.update_FName)
        self.window.mainloop()

    def update_FName(self):
        self.fileNameEntry.intert(0,self.BHIDstr.get()+"TEST")
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
xxx=1
# https://www.tutorialspoint.com/simple-registration-form-using-python-tkinter