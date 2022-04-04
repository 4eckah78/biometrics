import numpy as np
import random as rnd
import cv2
import matplotlib.pyplot as plt
from skimage.filters import sobel as sob
from skimage.filters import hessian, laplace, sato
import time
from skimage.measure import moments
from skimage.feature import hog
from mahotas import dog, euler, gaussian_filter, label, otsu
from mahotas.features import lbp, haralick, roundness


# bl = cv2.medianBlur(img, ksize=15)
# fltr = cv2.bilateralFilter(img, 7, 100, 100)

def Moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    order = 3
    res = moments(gray, order=order)
    return res, res


def Euler(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    T_otsu = otsu(gray)
    gray = gray > T_otsu
    eulr = np.abs(euler(gray))
    return eulr, eulr


def Haralick(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = gaussian_filter(img, 24)
    gaussian = (gaussian > gaussian.mean())
    labelled, n = label(gaussian)
    edges = haralick(labelled)
    return edges, edges


def LBP(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = lbp(img, 200, 10)
    # edges[edges > 0.05] = 1
    # edges[edges <= 0.05] = 0
    return edges, edges


def Hessian(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = hessian(img)
    # edges[edges > 0.05] = 1
    # edges[edges <= 0.05] = 0
    return edges, edges


def sobel(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = sob(img)
    # edges[edges > 0.05] = 1
    # edges[edges <= 0.05] = 0
    return edges, edges


def Sato(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = gray.shape
    # new_size = (int(w * 0.5), int(h * 0.5))
    # gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    edges = sato(gray)
    return edges, edges


def Laplace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = gray.shape
    # new_size = (int(w * 0.5), int(h * 0.5))
    # gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    edges = laplace(gray)
    # edges[edges > 0.05] = 1
    # edges[edges <= 0.05] = 0
    return edges, edges


def DOG(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = gray.shape
    # new_size = (int(w * 0.5), int(h * 0.5))
    # gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    edges = dog(gray)
    return edges.astype(int), edges


def hough(img):
    res = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_size = (int(w * 0.5), int(h * 0.5))
    gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    circles_img = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
                                   param1=200, param2=50, minRadius=10, maxRadius=0)
    feature = 0
    if circles_img is not None:
        feature = circles_img.shape[1]
        circles_img = np.uint16(np.around(circles_img))
        for i in circles_img[0, :]:
            cv2.circle(res, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(res, (i[0], i[1]), 2, (0, 0, 255), 3)
    return feature, res


def harris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = gray.shape
    # new_size = (int(w * 0.5), int(h * 0.5))
    # gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    res = np.copy(img)
    features = res[dst > 0.05 * dst.max()]
    res[dst > 0.05 * dst.max()] = [0, 0, 255]
    return features.shape[0], res


def HOG(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd, hog_image


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
    return accum, accum


def fast(img):
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(50)
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    return len(kp), img2


def canny(img):
    edges = cv2.Canny(img, 220, 230)
    amount_of_edges = len(edges[edges == 255])
    return amount_of_edges, edges


def color_hist(img):
    color = ('b', 'g', 'r')
    hists = []
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [64], [0, 256])
        hists.append(hist)
    return hists, hists


def random(img):
    np.random.seed(0)
    h, w, _ = img.shape
    features = []
    centers = []
    img2 = np.copy(img)
    for _ in range(400):
        x0 = int(np.random.rand() * w)
        y0 = int(np.random.rand() * h)
        centers.append((x0, y0))
        features.append(np.mean(img[y0, x0]))
        img2 = cv2.circle(img2, (x0, y0), 1, (0, 0, 255), 2)

    return features, img2


def get_methods():
    return [color_hist, canny, random, harris, LBP, Haralick]


# [sobel, canny, random, hough, harris, DOG, Laplace, Sato, HOG, gabor, color_hist, fast, Hessian, LBP,
#            Haralick, Euler, Moments]


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
            else:
                print(f"Error reading image s{i}/{j}{type}")
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


def split_data_random(data, images_per_class=10, images_per_person_in_train=5, seed=None, SHOW=False):
    amount_of_images = len(data[0])
    if seed:
        rnd.seed(seed)
    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, amount_of_images, images_per_class):
        indexes = list(range(i, i + images_per_class))
        train_indexes = rnd.sample(indexes, images_per_person_in_train)
        if SHOW:
            print(f"train_indexes: {[(ind + 1) % 17 for ind in train_indexes]}")
        x_train.extend([data[0][index] for index in train_indexes])
        y_train.extend([data[1][index] for index in train_indexes])

        test_indexes = set(indexes) - set(train_indexes)
        x_test.extend([data[0][index] for index in test_indexes])
        y_test.extend([data[1][index] for index in test_indexes])

    return x_train, x_test, y_train, y_test


def split_data_cross(data, images_per_class=10, images_per_person_in_train=5, train_indxs=[]):
    amount_of_images = len(data[0])

    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, amount_of_images, images_per_class):
        indexes = list(range(i, i + images_per_class))
        train_indexes = [indexes[i] for i in train_indxs]
        x_train.extend([data[0][index] for index in train_indexes])
        y_train.extend([data[1][index] for index in train_indexes])

        test_indexes = set(indexes) - set(train_indexes)
        x_test.extend([data[0][index] for index in test_indexes])
        y_test.extend([data[1][index] for index in test_indexes])

    return x_train, x_test, y_train, y_test


def create_feature(images, method):
    return [method(image)[0] for image in images]


def distance(el1, el2):
    return np.linalg.norm(np.array(el1) - np.array(el2))


def classifier(train, test, method, use_database=None):
    if use_database:
        featured_train = use_database[method.__name__]
    else:
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


def voting(train, test, SHOW=False, use_database=None):
    methods = get_methods()
    res = {}
    cols = len(methods) // 3 + 1
    index = 1
    start = time.time()
    for method in methods:
        res[method.__name__] = classifier(train, test, method, use_database=use_database)
        # print(f"method {method.__name__} ended in {int(time.time() - start)} seconds")
    voted_answers = []
    for i in range(len(test[0])):
        answers_to_image_1 = {}

        if SHOW:
            plt.subplot(3, cols, index)
            index += 1
            plt.imshow(cv2.cvtColor(test[0][i], cv2.COLOR_BGR2RGB)), plt.axis("off"), plt.title("Query Image")

        for method in res:
            answer = res[method][i]
            if answer in answers_to_image_1:
                answers_to_image_1[answer] += 1
            else:
                answers_to_image_1[answer] = 1
            if method == "color_hist" and answers_to_image_1[answer]:
                answers_to_image_1[answer] += 0.5

            if SHOW:
                plt.subplot(3, cols, index)
                index += 1
                for train_image, true_answer in zip(train[0], train[1]):
                    if true_answer == answer:
                        plt.imshow(cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)), plt.axis("off"), plt.title(method)
                        break

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
    plt.rcParams["figure.figsize"] = (20, 20)
    index = 1
    cols = len(get_methods()) // 2 + 1
    plt.subplot(2, cols, index, title="original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis("off")
    for method in get_methods():
        index += 1
        plt.subplot(2, cols, index, title=method.__name__)
        to_draw = method(image)[1]
        if method == color_hist:
            for hist, col in zip(to_draw, ('b', 'g', 'r')):
                plt.plot(range(len(hist)), hist, col)
        elif method in (LBP, ):
            plt.plot(range(len(to_draw)), to_draw)
        elif method == Haralick:
            plt.imshow(to_draw, cmap="gray")
        elif method in (Euler,):
            plt.text(0.35,
                     0.5,
                     to_draw,
                     transform=plt.gca().transAxes, fontdict={'size': 20}), plt.axis("off")
        elif len(to_draw.shape) == 2:
            plt.imshow(to_draw, cmap="gray"), plt.axis("off")
        else:
            plt.imshow(cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)), plt.axis("off")
    plt.show()


def test_methods(train, test):
    res = {}
    for method in get_methods():
        answers = classifier(train, test, method)
        correct_answers = 0
        for i in range(len(answers)):
            if answers[i] == test[1][i]:
                correct_answers += 1
        res[method.__name__] = correct_answers / len(answers)
    return res


def get_size_and_seed():
    size = 8
    seed = 87
    classes = {0: "И. И. Шишкин", 1: "И. К. Айвазовский",
               2: "П. Пикассо", 3: "В. И. Суриков"}

    # 10 PAINTS IN TRAIN, 6  PAINTS  IN TEST, seed = 87
    # methods: {'color_hist': 0.5833333333333334, 'canny': 0.4583333333333333, 'random': 0.4166666666666667,
    #           'harris': 0.5, 'LBP': 0.5416666666666666, 'Haralick': 0.375}
    # voting --> 0.7083333333333334
    return size, seed, classes


if __name__ == "__main__":
    size, seed, classes = get_size_and_seed()
    data = load_paintings_from("./Paints/s", len(classes), 16)
    # for image in data[0]:
    #     draw_methods(image)

    stats = {}
    for seed in range(1, 30):
        print(f"seed={seed}")
        # for size in range(1, 16):
        x_train, x_test, y_train, y_test = split_data_random(data, 16, size, seed=seed)
        train = [x_train, y_train]
        test = [x_test, y_test]
        res = test_methods(train, test)
        classf = test_voting(train, test)
        # for method in get_methods():
        #     if method.__name__ in stats:
        #         stats[method.__name__].append(res[method.__name__])
        #     else:
        #         stats[method.__name__] = [res[method.__name__]]
        # if "voting" in stats:
        #     stats["voting"].append(classf)
        # else:
        #     stats["voting"] = [classf]
        if classf >= 0.6:
            print(f"{size} PAINTS IN TRAIN, {16 - size} PAINTS IN TEST, seed={seed}")
            print("methods: ", res)
            print(f"voting --> {classf}")
            print("*" * 10)
    # for method, stat in stats.items():
    #     plt.plot(range(len(stat)), stat, label=method)
    # plt.title(f"train size={size}"), plt.legend(loc='best'), plt.xlabel("partition"), plt.ylabel("score")
    # plt.show()

    # count = 0
    # summ = 0
    # result = []
    # for test_image, true_answer in zip(x_test, y_test):
    #
    #     res = voting(train, [[test_image], [true_answer]])
    #     # res = classifier(train, test, color_hist)
    #     if true_answer == res[0]:
    #         summ += 1
    #     else:
    #         print(f"return {classes[res[0]]} but true is {classes[true_answer]}")
    #         # res = {}
    #         # for method in get_methods():
    #         #     answers = classifier(train, [[test_image], [true_answer]], method)
    #         #     print(f"method {method.__name__} found {classes[answers[0]]}")
    #         # print(test_methods(train, [[test_image], [true_answer]]))
    #         # plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)), plt.axis("off")
    #         # plt.show()
    #     count += 1
    #     result.append(summ / count)
    #     print(f"{count} images --> {summ / count}")
    # plt.plot(range(1, len(result) + 1),  result), plt.xlabel("amount of test images"), plt.ylabel("score"), plt.title("Voting")
    # plt.show()

    # count = 1
    # plt.rcParams["figure.figsize"] = (10, 6)
    # index = 1
    # for train_image, ans in zip(x_train, y_train):
    #     plt.subplot(2, 5, index)
    #     plt.imshow(cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)), plt.axis("off")
    #     index += 1
    #     if index > 10:
    #         plt.savefig(f"./Results/{count}.jpg")
    #         count += 1
    #         index = 1
