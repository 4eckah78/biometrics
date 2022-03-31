from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from painting_author import *

image = None
data = []
train = []
test = []
train_size = 8
seed = 52
methods = get_methods()
classes = {0: "Шишкин", 1: "Айвазовский",
           2: "Пикассо", 3: "Суриков"}
plt.rcParams["figure.figsize"] = [8, 5]


def browseFiles():
    global image
    image_path = filedialog.askopenfilename(initialdir="C:/Paints/",
                                            title="Select a File",
                                            filetypes=[("Image files", "*.jpg *.png *.pgm")])
    image = cv2.imread(image_path)
    label_database.configure(text="✓ File selected")


def load():
    global data, train, test
    data = load_paintings_from("./Paints/s", 4, 16)
    label_database.configure(text="✓ Database is uploaded")
    x_train, x_test, y_train, y_test = split_data_random(data, 16, train_size, seed=seed)
    train = [x_train, y_train]
    test = [x_test, y_test]


def classify():
    if image is None:
        label_database.configure(text="You need to upload an image first!")
    else:
        answer = voting(train, [[image], [0]])

        plt.subplot(1, 2, 1, title="Query Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis("off")
        plt.subplot(1, 2, 2, title="Result"), plt.axis("off")
        plt.text(0.3,
                 0.5,
                 classes[answer[0]],
                 transform=plt.gca().transAxes, fontdict={'size': 22})
        plt.show()


def show():
    if image is None:
        label_database.configure(text="You need to upload an image first!")
    else:
        draw_methods(image)
        plt.rcParams["figure.figsize"] = [8, 5]


def test():
    for test_img in test[0]:
        answer = voting(train, [[test_img], [0]])

        plt.subplot(1, 2, 1, title="Query Image")
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)), plt.axis("off")
        plt.subplot(1, 2, 2, title="Result"), plt.axis("off")
        plt.text(0.3,
                 0.5,
                 classes[answer[0]],
                 transform=plt.gca().transAxes, fontdict={'size': 22})
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def test_on_train():
    for train_img in train[0]:
        answer = voting(train, [[train_img], [0]])

        plt.subplot(1, 2, 1, title="Query Image")
        plt.imshow(cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)), plt.axis("off")
        plt.subplot(1, 2, 2, title="Result"), plt.axis("off")
        plt.text(0.3,
                 0.5,
                 classes[answer[0]],
                 transform=plt.gca().transAxes, fontdict={'size': 22})
        plt.show(block=False)
        plt.pause(2)
        plt.close()


if __name__ == "__main__":
    window = Tk()
    window.title('Paintings')
    window.geometry("640x480")
    window.config(background="white")

    bg = PhotoImage(file="./2.png")

    label1 = Label(window, image=bg)
    label1.place(x=150, y=10)
    label_database = Label(window)

    btn1 = Button(window, text="Load database", command=load)

    button_explore = Button(window, text="Select Image", command=browseFiles)

    btn2 = Button(window, text="Classify image", command=classify)

    btn3 = Button(window, text="Show methods", command=show)

    btn4 = Button(window, text="Test", command=test)

    btn5 = Button(window, text="Test on train", command=test_on_train)

    label_database.grid(column=3, row=1)

    btn1.grid(column=1, row=1, pady=5, padx=10)
    button_explore.grid(column=1, row=2, pady=5)
    btn2.grid(column=1, row=4, pady=5)
    btn3.grid(column=1, row=3, pady=5)
    btn4.grid(column=1, row=5, pady=5)
    btn5.grid(column=1, row=6, pady=5)

    window.mainloop()
