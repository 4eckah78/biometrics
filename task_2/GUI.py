from tkinter import *
from tkinter import filedialog
from face_recognition import *
from test import *

image = []
data = []
train = []
train_size = 5
methods = [histogram, dft, dct, gradient, scale]
params = [23, 15, 8, 4, 0.2]
show_methods = [histogram, show_dft, show_dct, gradient, scale]


def browseFiles():
    global image
    image_path = filedialog.askopenfilename(initialdir="C:/",
                                            title="Select a File",
                                            filetypes=[("Image files", "*.jpg *.png *.pgm")])
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image = image / 255
    label_database.configure(text="File selected")


def load():
    global data, train
    data = load_faces_from("./orl_faces/s")
    label_database.configure(text="Database is uploaded")
    x_train, _, y_train, _ = split_data_not_random(data, train_size, DRAW=True)
    train = mesh_data([x_train, y_train])


def classify():
    classify_image(train, image, methods, params, train_size=train_size)


def show():
    methods_to_image_with_params(show_methods, image, params)


def cross_valid():
    parameters, best_size, score = cross_validation(data, etalons_range=[1, 6], SHOW=True)
    label_database.configure(text=f"best size: {best_size}, best_params: {parameters}, scrore: {score}")


def voting():
    vote_classifier(data, SHOW=True)


if __name__ == "__main__":
    window = Tk()
    window.title('File Explorer')
    window.geometry("500x500")
    window.config(background="white")

    label_database = Label(window, text="", width=100, height=4, fg="blue")

    btn1 = Button(window, text="Load ORL database (112x92)", command=load)

    button_explore = Button(window, text="Select Image", command=browseFiles)

    btn2 = Button(window, text="Classify image", command=classify)

    btn3 = Button(window, text="Show methods", command=show)

    btn4 = Button(window, text="Cross validation", command=cross_valid)

    btn5 = Button(window, text="Voting", command=voting)

    button_exit = Button(window,
                         text="Exit",
                         command=exit)

    label_database.grid(column=1, row=1)
    btn1.grid(column=1, row=2)
    button_explore.grid(column=1, row=3)
    btn2.grid(column=1, row=4)
    btn3.grid(column=1, row=5)
    btn4.grid(column=1, row=6)
    btn5.grid(column=1, row=7)
    button_exit.grid(column=1, row=8)

    window.mainloop()
