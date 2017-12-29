from tkinter import *
from tkinter import ttk

from tkinter.filedialog import askopenfilename

class MyUserInterface:


    loaded_filename = ""
    Model_file_ext = ".ckpt"

    def __init__(self):
        self.master = Tk()
        self.architecture = [784, 784]

        self.notebook = ttk.Notebook(self.master)
        self.frame1 = Frame(self.notebook)  # first page, which would get widgets gridded into it
        self.frame2 = Frame(self.notebook)  # second page
        self.notebook.add(self.frame1, text='Training')
        self.notebook.add(self.frame2, text='Compression')
        self.notebook.pack(expand=1, fill="both")

        self.training_label_arc = Label(self.frame1, text="Network Architecture:" + " ".join(str(self.architecture)))
        self.training_label_arc.grid(row=0)
        self.training_entry = Entry(self.frame1, width=10)
        self.training_entry.grid(row=1, column=1)
        self.training_button_addlayer = Button(self.frame1, text="Add Layer", command=self.training_addlayer_callback)
        self.training_button_addlayer.grid(row=1, column=2)
        self.training_button_train = Button(self.frame1, text="Train Net", command=self.training_train_callback)
        self.training_button_train.grid(row=2, column=1)
        self.training_button_clear = Button(self.frame1, text="Clear Net", command=self.training_clear_callback)
        self.training_button_clear.grid(row=2, column=2)


        self.model_file_label = Label(self.frame2, text="Model File:")
        self.model_file_label.grid(row=0, column=0)
        self.model_file_button = Button(self.frame2, text="Select File", command=self.modelfile_select_callback)
        self.model_file_button.grid(row=0, column=1)
        self.model_file_entry = Entry(self.frame2, width=120)
        self.model_file_entry.grid(row=0, column=2)



    def modelfile_select_callback(self):
        file_ext = MyUserInterface.Model_file_ext
        loaded_filename = askopenfilename(filetypes=[('TensorFlow model files', '*' + file_ext + '*')])
        loaded_filename = loaded_filename.split(file_ext)
        loaded_filename = loaded_filename[0] + file_ext
        self.model_file_entry.delete(0, END)
        self.model_file_entry.insert(0, loaded_filename)
        return

    def training_addlayer_callback(self):
        new_layer = int(self.training_entry.get())
        if(len(self.architecture) == 2):
            self.architecture.insert(len(self.architecture)//2, new_layer)
        else:
            old_middle = self.architecture[len(self.architecture) // 2]
            self.architecture.insert(len(self.architecture) // 2, old_middle)
            self.architecture.insert(len(self.architecture) // 2, new_layer)
        self.training_label_arc.config(text="Network Architecture:" + " ".join(str(self.architecture)))
        return

    def training_train_callback(self):
        return

    def training_clear_callback(self):
        self.architecture = [784, 784]
        self.training_label_arc.config(text="Network Architecture:" + " ".join(str(self.architecture)))
        return

    def mainloop(self):
        self.master.mainloop()


if __name__ == "__main__":
    ui = MyUserInterface()
    ui.mainloop()

