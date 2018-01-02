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
        self.master.title("Autoencoder Demo")
        self.master.geometry('1000x500')
        self.architecture = [784, 784]

        self.notebook = ttk.Notebook(self.master)
        self.frame1 = Frame(self.notebook)
        self.frame2 = Frame(self.notebook)
        self.notebook.add(self.frame1, text='Training')
        self.notebook.add(self.frame2, text='Compression')
        self.notebook.pack(expand=1, fill="both")

        self.canvas_decomp_out = np.zeros((28, 28))
        self.canvas_input = np.empty((28, 28))

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
        self.canvas_selector = FigureCanvasTkAgg(self.get_figure(self.selected_image_number), master=self.frame2)
        self.plot_widget_selector = self.canvas_selector.get_tk_widget()
        self.plot_widget_selector.place(relx=0.02, rely=0.4)

        self.canvas_ext_res = FigureCanvasTkAgg(self.get_empty_figure(), master=self.frame2)
        self.plot_widget_ext_res = self.canvas_ext_res.get_tk_widget()
        self.plot_widget_ext_res.place(relx=0.4, rely=0.4)

        self.model_file_label = Label(self.frame2, text="Model File:")
        self.model_file_label.place(relx=0.02, rely=0.02)
        self.model_file_button = Button(self.frame2, text="Select File", command=self.modelfile_select_callback)
        self.model_file_button.place(relx=0.1, rely=0.02)
        self.model_file_entry = Entry(self.frame2, width=120)
        self.model_file_entry.place(relx=0.2, rely=0.02)
        self.model_file_compress_button = Button(self.frame2, text="Compress Data", command=self.model_file_compress)
        self.model_file_compress_button.place(relx=0.02, rely=0.15)
        self.model_file_decompress = Button(self.frame2, text="Decompress Data", command=self.model_file_decompress)
        self.model_file_decompress.place(relx=0.15, rely=0.15)
        self.model_file_select_image_button = Button(self.frame2, text="Select Image", command=self.got_to_image)
        self.model_file_select_image_button.place(relx=0.02, rely=0.25)
        self.model_file_image_number = Entry(self.frame2, width=10)
        self.model_file_image_number.place(relx=0.11, rely=0.26)

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
        autoencoder_architecture = self.architecture#[784, 128, 64, 128, 784]
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        image_data = mnist.train.images[self.selected_image_number]
        image_data = image_data.reshape(1, autoencoder_architecture[0])
        save_file = asksaveasfilename(defaultextension=".test")
        util.export_compressed_data(image_data, self.model_file_entry.get(), autoencoder_architecture, save_file)
        return

    def model_file_decompress(self):
        autoencoder_architecture = self.architecture
        load_compressed_file = askopenfilename(filetypes=[('Compression results', '.test')])
        load_data = util.import_compressed_data(self.model_file_entry.get(), autoencoder_architecture, load_compressed_file)

        self.canvas_decomp_out = np.zeros((28, 28))
        self.canvas_decomp_out = load_data.reshape([28, 28])
        figure = plt.figure(figsize=(2, 2), frameon=False)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        plt.imshow(self.canvas_decomp_out, cmap="gray")
        self.canvas_ext_res.figure = figure
        self.canvas_ext_res.show()

        s = np.sum((self.canvas_decomp_out - self.canvas_input) ** 2) / 784
        print(s)

        return

    def get_figure(self, selected_image_number):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        original_image = mnist.train.images[selected_image_number]
        self.canvas_input = np.empty((28, 28))
        self.canvas_input = original_image.reshape([28, 28])
        figure = plt.figure(figsize=(2, 2), frameon=False)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        plt.imshow(self.canvas_input, cmap="gray")
        return figure

    def get_empty_figure(self):
        self.canvas_decomp_out = np.zeros((28, 28))
        figure = plt.figure(figsize=(2, 2), frameon=False)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        plt.imshow(self.canvas_decomp_out, cmap="gray")
        return figure

    def got_to_image(self):
        self.selected_image_number = int(self.model_file_image_number.get())
        self.canvas_selector.figure = self.get_figure(self.selected_image_number)
        self.canvas_selector.show()
        return

    def mainloop(self):
        self.master.mainloop()

if __name__ == "__main__":
    ui = MyUserInterface()
    ui.mainloop()

