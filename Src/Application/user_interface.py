from tkinter import *
from tkinter import ttk
import utility as util
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
class MyUserInterface:


    loaded_filename = ""
    Model_file_ext = ".ckpt"

    def __init__(self):
        self.master = Tk()
        self.master.geometry('1000x500')
        self.architecture = [784, 784]

        self.notebook = ttk.Notebook(self.master)
        self.frame1 = Frame(self.notebook)
        self.frame2 = Frame(self.notebook)
        self.notebook.add(self.frame1, text='Training')
        self.notebook.add(self.frame2, text='Compression')
        self.notebook.pack(expand=1, fill="both")

        self.training_label_arc = Label(self.frame1, text="Network Architecture:" + " ".join(str(self.architecture)))
        self.training_label_arc.place(relx=0.02, rely=0.02)
        self.training_entry = Entry(self.frame1, width=10)
        self.training_entry.place(relx=0.1, rely=0.105)
        self.training_button_addlayer = Button(self.frame1, text="Add Layer", command=self.training_addlayer_callback)
        self.training_button_addlayer.place(relx=0.02, rely=0.1)
        self.training_button_train = Button(self.frame1, text="Train Net", command=self.training_train_callback)
        self.training_button_train.place(relx=0.02, rely=0.2)
        self.training_button_clear = Button(self.frame1, text="Clear Net", command=self.training_clear_callback)
        self.training_button_clear.place(relx=0.1, rely=0.2)

        self.selected_image_number = 0
        canvas = FigureCanvasTkAgg(self.get_figure(self.selected_image_number), master=self.frame2)
        self.plot_widget = canvas.get_tk_widget()
        self.plot_widget.place(relx=0.5, rely=0.5)

        self.model_file_label = Label(self.frame2, text="Model File:")
        self.model_file_label.place(relx=0.02, rely=0.02)
        self.model_file_button = Button(self.frame2, text="Select File", command=self.modelfile_select_callback)
        self.model_file_button.place(relx=0.1, rely=0.02)
        self.model_file_entry = Entry(self.frame2, width=120)
        self.model_file_entry.place(relx=0.2, rely=0.02)
        self.model_file_compress_button = Button(self.frame2, text="Compress Data", command=self.model_file_compress)
        self.model_file_compress_button.place(relx=0.02, rely=0.15)
        self.model_file_decompress = Button(self.frame2, text="Decompress Data", command=self.modelfile_select_callback)
        self.model_file_decompress.place(relx=0.15, rely=0.15)
        self.model_file_select_image_button = Button(self.frame2, text="Select Image", command=self.got_to_image)
        self.model_file_select_image_button.place(relx=0.35, rely=0.35)
        self.model_file_image_number = Entry(self.frame2, width=10)
        self.model_file_image_number.place(relx=0.45, rely=0.45)

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
        save_file = asksaveasfilename(defaultextension=".ckpt")
        util.train_network_mnist(self.architecture, save_file)
        return

    def training_clear_callback(self):
        self.architecture = [784, 784]
        self.training_label_arc.config(text="Network Architecture:" + " ".join(str(self.architecture)))
        return

    def model_file_compress(self):
        autoencoder_architecture = [784, 128, 64, 128, 784]
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        image_data = mnist.train.images[self.selected_image_number]
        save_file = asksaveasfilename()
        util.export_compressed_data(image_data, self.model_file_entry, autoencoder_architecture, save_file)
        return

    def model_file_decompress(self):
        return

    def get_figure(self, selected_image_number):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        original_image = mnist.train.images[selected_image_number]
        canvas_orig = np.empty((28, 28))
        canvas_orig = original_image.reshape([28, 28])
        figure = plt.figure(figsize=(2, 2))
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        plt.imshow(canvas_orig, cmap="gray")
        return figure

    def got_to_image(self):
        self.selected_image_number = int(self.model_file_image_number.get())
        canvas = FigureCanvasTkAgg(self.get_figure(self.selected_image_number), master=self.frame2)
        self.plot_widget = canvas.get_tk_widget()
        self.plot_widget.place(relx=0.5, rely=0.5)
        return

    def mainloop(self):
        self.master.mainloop()

if __name__ == "__main__":
    ui = MyUserInterface()
    ui.mainloop()

