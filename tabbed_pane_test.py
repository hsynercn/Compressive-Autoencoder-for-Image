from tkinter import *
from tkinter import ttk
master = Tk()

n = ttk.Notebook(master)
f1 = Frame(n)   # first page, which would get widgets gridded into it
f2 = Frame(n)   # second page
n.add(f1, text='One')
n.add(f2, text='Two')
n.pack(expand=1, fill="both")
mainloop()