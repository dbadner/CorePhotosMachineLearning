from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk


class UIForm:
    #shows User Interface form
    #ImagePath: str#tk.StringVar

    def __init__(self):
        window = Tk()
        window.title("Machine Learning Core Photo Renaming App")
        window.geometry('1200x800')
        # window.configure(background="gray")
        self.ImagePath = tk.StringVar()

        ttk.Label(window, text="Photograph folder location:").grid(row=0, column=0)
        ttk.Button(text="Browse", command=self.browse_button).grid(row=0, column=1)
        ttk.Entry(width=120, textvariable=self.ImagePath).grid(row=0, column=2)
        ttk.Label(window, text="Borehole ID:").grid(row=1, column=0)
        ttk.Label(window, text="Depth from:").grid(row=2, column=0)
        ttk.Label(window, text="Depth to:").grid(row=3, column=0)
        ttk.Label(window, text="Units:").grid(row=4, column=0)
        ttk.Label(window, text="Wet / dry:").grid(row=5, column=0)

        for i in range(5):
            ttk.Entry(width=50).grid(row=i + 1, column=2)
            # ent.pack(side = RIGHT, expand = YES, fill = X)

        window.mainloop()

    def browse_button(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.setPath(filename)

    def setPath(self, word):
        self.ImagePath.set(word)


# https://www.tutorialspoint.com/simple-registration-form-using-python-tkinter