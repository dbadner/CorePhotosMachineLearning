from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk

"""
class UIForm:
    #shows User Interface form
    ImagePath: str

    def __init__(self, imagepath):
        self.ImagePath = imagepath

    def BuildForm(self):
"""

#https://www.tutorialspoint.com/simple-registration-form-using-python-tkinter


def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    filename = filedialog.askdirectory()
    setPath(filename)


def setPath(word):
    path.set(word)



#objUI = UIForm("")
#objUI.BuildForm()

def BuildForm():
    window = Tk()
    window.title("Machine Learning Core Photo Renaming App")
    window.geometry('1200x800')
    #window.configure(background="gray")
    
    path = tk.StringVar()

    ttk.Label(window, text="Photograph folder location:").grid(row = 0, column = 0)
    ttk.Button(text="Browse", command=browse_button).grid(row=0,column = 1)
    ttk.Entry(width=120, textvariable=path).grid(row=0,column=2)
    ttk.Label(window, text="Borehole ID:").grid(row=1,column=0)
    ttk.Label(window, text="Depth from:").grid(row=2,column=0)
    ttk.Label(window, text="Depth to:").grid(row=3,column=0)
    ttk.Label(window, text="Units:").grid(row=4,column=0)
    ttk.Label(window, text="Wet / dry:").grid(row=5,column=0)

    for i in range(5):
        ttk.Entry(width=50).grid(row=i+1,column=2)
        #ent.pack(side = RIGHT, expand = YES, fill = X)

    window.mainloop()