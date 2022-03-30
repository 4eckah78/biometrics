import numpy as np
import random as rnd
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog


def HOG(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd, hog_image


def color_hist(img):
    color = ('b', 'g', 'r')
    hists = []
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [64], [0, 256])
        hists.append(hist)
    return hists


def gabor(img):
    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sunusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values that each pixel in the gabor kernel can hold
    filters = []
    ksize = 51
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel(ksize=(ksize, ksize), sigma=4.0, theta=theta, lambd=10.0,
                                  gamma=0.5, psi=0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum


def load_paintings_from(data_folder, classes, images_in_class, type=".jpg"):
    print("Accessing to database...")
    data_paintings = []
    data_target = []
    for i in range(1, classes + 1):
        for j in range(1, images_in_class + 1):
            image = cv2.imread(f"{data_folder}{i}/{j}{type}")
            if image is not None:
                data_paintings.append(image)
                data_target.append(i - 1)
    print(
        f"Database is uploaded: {len(data_paintings)} paintings, {classes} classes, {images_in_class} images in each class")
    print("=" * 50)
    return [data_paintings, data_target]


def split_data_not_random(data, images_per_class=10, images_per_class_in_train=5):
    amount_of_images = len(data[0])

    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, amount_of_images, images_per_class):
        x_train.extend(data[0][i: i + images_per_class_in_train])
        y_train.extend(data[1][i: i + images_per_class_in_train])

        x_test.extend(data[0][i + images_per_class_in_train: i + images_per_class])
        y_test.extend(data[1][i + images_per_class_in_train: i + images_per_class])

    return x_train, x_test, y_train, y_test


def split_data_random(data, images_per_class=10, images_per_person_in_train=5):
    amount_of_images = len(data[0])

    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, amount_of_images, images_per_class):
        indexes = list(range(i, i + images_per_class))
        train_indexes = rnd.sample(indexes, images_per_person_in_train)
        x_train.extend([data[0][index] for index in train_indexes])
        y_train.extend([data[1][index] for index in train_indexes])

        test_indexes = set(indexes) - set(train_indexes)
        x_test.extend([data[0][index] for index in test_indexes])
        y_test.extend([data[1][index] for index in test_indexes])

    return x_train, x_test, y_train, y_test


def create_feature(images, method):
    return [method(image)[0] if method == HOG else method(image) for image in images]


def distance(el1, el2):
    return np.linalg.norm(np.array(el1) - np.array(el2))


def classifier(train, test, method):
    if method not in [HOG, color_hist, gabor]:
        return []
    featured_train = create_feature(train[0], method)
    featured_test = create_feature(test[0], method)
    answers = []
    for test_element in featured_test:
        min_el = [100000, -1]
        for i in range(len(featured_train)):
            dist = distance(test_element, featured_train[i])
            if dist < min_el[0]:
                min_el = [dist, i]
        if min_el[1] < 0:
            answers.append(0)
        else:
            answers.append(train[1][min_el[1]])
    return answers


def voting(train, test):
    methods = [HOG, color_hist, gabor]
    res = {}
    for method in methods:
        res[method.__name__] = classifier(train, test, method)
    voted_answers = []
    for i in range(len(test[0])):
        answers_to_image_1 = {}
        for method in res:
            answer = res[method][i]
            if answer in answers_to_image_1:
                answers_to_image_1[answer] += 1
            else:
                answers_to_image_1[answer] = 1
        best_size = sorted(answers_to_image_1.items(), key=lambda item: item[1], reverse=True)[0]
        voted_answers.append(best_size[0])

    return voted_answers


def test_voting(train, test):
    res = voting(train, test)
    sum = 0
    for i in range(len(test[0])):
        if test[1][i] == res[i]:
            sum += 1
    return sum / len(test[0])


def draw_methods(image):
    plt.rcParams["figure.figsize"] = (20, 3)
    plt.subplot(1, 4, 1, title="Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis("off")

    fd, hog_image = HOG(image)
    plt.subplot(1, 4, 2, title="HOG")
    plt.imshow(hog_image, cmap="gray"), plt.axis("off")

    hists = color_hist(image)
    plt.subplot(1, 4, 3, title="Histogram")
    for hist, col in zip(hists, ('b', 'g', 'r')):
        plt.plot(range(len(hist)), hist, col)

    accum = gabor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 4, 4, title="Gabor")
    plt.imshow(accum), plt.axis("off")
    plt.show()


if __name__ == "__main__":
    data = load_paintings_from("./Paints/s", 4, 16)
    # for image in data[0]:
    #     draw_methods(image)
    size = 10
    rnd.seed(6)  # 0.45833
    print(f"{size} PAINTS IN TRAIN, {16 - size} PAINTS IN TEST")
    x_train, x_test, y_train, y_test = split_data_random(data, 16, size)
    train = [x_train, y_train]
    test = [x_test, y_test]
    # voting(train, [[test[0][0]], [test[1][0]]])
    classf = test_voting(train, test)
    print(f"score = {classf}")
    print("*" * 10)
