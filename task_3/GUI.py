from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from painting_author import *

image = None
data = []
train = []
test = []
featured_train = {}
train_size, seed, classes = get_size_and_seed()
methods = get_methods()
plt.rcParams["figure.figsize"] = [8, 5]


def browseFiles():
    if len(data) == 0:
        label_database.configure(text="You need to upload a database first!")
    else:
        global image
        image_path = filedialog.askopenfilename(initialdir="C:/Paints/",
                                                title="Select a File",
                                                filetypes=[("Image files", "*.jpg *.png *.pgm")])
        image = cv2.imread(image_path)
        label_database.configure(text="✓ File selected")


def load():
    label_database.configure(text="Database is uploading...")
    global data, train, test, featured_train
    data = load_paintings_from("./Paints/s", len(classes), 16)
    x_train, x_test, y_train, y_test = split_data_random(data, 16, train_size, seed=seed)
    train = [x_train, y_train]
    test = [x_test, y_test]
    start = time.time()
    featured_train = {method.__name__: create_feature(x_train, method) for method in get_methods()}
    label_database.configure(text="✓ Database is uploaded")
    print(f"Database loaded in {int(time.time() - start)} seconds")


def classify():
    if image is None:
        label_database.configure(text="You need to upload an image first!")
    else:
        answer = voting(train, [[image], [0]], SHOW=True, use_database=featured_train)
        label_database.configure(text=classes[answer[0]])

        plt.subplot(3, 3, 8, title="Result"), plt.axis("off")
        plt.text(0.3,
                 0.5,
                 classes[answer[0]],
                 transform=plt.gca().transAxes, fontdict={'size': 12})
        plt.show()


def show():
    if image is None:
        label_database.configure(text="You need to upload an image first!")
    else:
        draw_methods(image)
        plt.rcParams["figure.figsize"] = [8, 5]


def test():
    if len(data) == 0:
        label_database.configure(text="You need to upload a database first!")
    else:
        for test_img in test[0]:
            answer = voting(train, [[test_img], [0]], SHOW=True, use_database=featured_train)
            label_database.configure(text=classes[answer[0]])
            # plt.subplot(1, 2, 1, title="Query Image")
            # plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)), plt.axis("off")
            plt.subplot(3, 3, 8, title="Result"), plt.axis("off")
            plt.text(0.3,
                     0.5,
                     classes[answer[0]],
                     transform=plt.gca().transAxes, fontdict={'size': 12})
            plt.show()
            plt.show(block=False)
            plt.pause(2)
            plt.close()


def test_on_train():
    if len(data) == 0:
        label_database.configure(text="You need to upload a database first!")
    else:
        for train_img in train[0]:
            answer = voting(train, [[train_img], [0]], SHOW=True, use_database=featured_train)
            label_database.configure(text=classes[answer[0]])
            plt.subplot(3, 3, 8, title="Result"), plt.axis("off")
            plt.text(0.3,
                     0.5,
                     classes[answer[0]],
                     transform=plt.gca().transAxes, fontdict={'size': 12})
            plt.show()
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
