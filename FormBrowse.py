from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk
import ctypes
import os

class frmBrowse:
    #shows User Interface form
    #ImagePathStr: str

    def __init__(self):
        self.window = Tk()
        self.window.title("Machine Learning Core Photo Renaming App")
        self.window.iconphoto(False, tk.PhotoImage(file='input\\icon.png'))
        self.window.bind('<Return>', self.enter_run)
        self.window.geometry('1000x200')
        # window.configure(background="gray")
        self.ImagePath = tk.StringVar()
        self.skipML = False
        self.ImagePathStr = ""
        self.cpuMode = True #default to True
        self.chkValue = tk.BooleanVar()
        self.device = tk.IntVar()

        L1 = ttk.Label(self.window, text="Photograph folder location:")#.grid(row=0, column=0)
        L1.place(x=5, y=2)
        B = ttk.Button(text="Browse", command=self.browse_button)#.grid(row=0, column=1)
        B.place(x=160, y=0)
        E = ttk.Entry(width=120, textvariable=self.ImagePath)#.grid(row=0, column=2)
        E.place(x=245, y=2)
        C = ttk.Checkbutton(text="Skip machine learning", variable=self.chkValue)#.grid(row=1, column=0)
        C.place(x=5, y=40)
        L2 = ttk.Label(self.window, text="Process graphics using:")#.grid(row=2, column=0)
        L2.place(x=5, y=80)
        r = ttk.Radiobutton(text="CPU", variable=self.device, value=1)#.grid(row=2, column=1)
        r.place(x=140, y=80)
        r2 = ttk.Radiobutton(text="GPU", variable=self.device, value=0)#.grid(row=2, column=2)
        r2.place(x=190, y=80)
        bt = ttk.Button(text="Run", command=self.run_button)#.grid(row=3, column=0)
        bt.place(x=5, y=120)

        #initialize values
        self.device.set(1)
        self.chkValue.set(False)

        self.window.mainloop()

    def browse_button(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.setPath(filename)

    def setPath(self, word):
        self.ImagePath.set(word)

    def run_button(self):

        if not os.path.exists(self.ImagePath.get()):
            errStr = "Error: Directory does not exist. Please select an existing directory containing your photographs."
            ctypes.windll.user32.MessageBoxW(0, errStr, "Error", 0)
        else:
            self.ImagePathStr = self.ImagePath.get()
            self.skipML = self.chkValue.get()
            self.cpuMode = bool(self.device.get())
            self.window.destroy()

    def enter_run(self, obj):
        # obj is the object returned when the enter key is pressed to enter this function. not used
        self.run_button()

    #def run(self):
#imagepath = tk.StringVar()
#objUI = UIForm()

#objUI.BuildForm()

#def BuildForm():
# https://www.tutorialspoint.com/simple-registration-form-using-python-tkinter