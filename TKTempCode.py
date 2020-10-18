#https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application

#from tkinter import *
from tkinter import ttk, Button
from tkinter import filedialog
import tkinter as tk

class BrowseWindow:

    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.lblDir = tk.Label(self, text="Photograph folder location:")#.grid(row=0, column=0)
        #self.txtDir = tk.Label(self, text="Photograph folder location:")#.grid(row=0, column=1)
        self.btnBrowse = tk.Button(self.frame, text = 'Browse', width = 15, command = self.Browse)#.grid(row=0, column=2)
        self.lblDir.pack(side=tk.LEFT)
        self.btnBrowse.pack(side=tk.LEFT)

        self.frame.pack()
    def Browse(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow)

class Demo2:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.quitButton = tk.Button(self.frame, text = 'Quit', width = 25, command = self.close_windows)
        self.quitButton.pack()
        self.frame.pack()
    def close_windows(self):
        self.master.destroy()

def main():
    root = tk.Tk()
    app = BrowseWindow(root)
    root.mainloop()

if __name__ == '__main__':
    main()

"""
class BrowseForm(tk.Frame):
    Path: tk.StringVar()

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.title("Machine Learning Core Photo Renaming App")
        self.geometry('1200x800')
        # window.configure(background="gray")

        self.Path = tk.StringVar()

        ttk.Label(self, text="Photograph folder location:").grid(row=0, column=0)
        #ttk.Button(text="Browse", command=browse_button).grid(row=0, column=1)

    #@staticmethod
    def browse_button(self):
        # Allow user to select a directory and store it in global var
        # called folder_path
        filename = filedialog.askdirectory()
        self.setPath(filename)

    def setPath(self, word):
        self.Path.set(word)

if __name__ == "__main__":
    root = tk.Tk()
    BrowseForm(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
"""