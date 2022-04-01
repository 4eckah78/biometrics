import numpy as np
import random as rnd
import cv2
import matplotlib.pyplot as plt


def canny(img):
    edges = cv2.Canny(img, 200, 220)
    amount_of_edges = len(edges[edges == 255])
    return amount_of_edges, edges


# methods:  {'canny': 0.40625, 'color_hist': 0.625, 'random': 0.4375}
# voting --> 0.59375
def color_hist(img):
    color = ('b', 'g', 'r')
    hists = []
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [64], [0, 256])
        hists.append(hist)
    return hists


def random(img):
    # bl = cv2.medianBlur(img, ksize=15)
    # fltr = cv2.bilateralFilter(img, 7, 100, 100)
    np.random.seed(0)
    h, w, _ = img.shape
    features = []
    centers = []
    for _ in range(400):
        x0 = int(np.random.rand() * w)
        y0 = int(np.random.rand() * h)
        centers.append((x0, y0))
        features.append(img[y0, x0])
    return features, centers


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
    return [method(image) if method == color_hist else method(image)[0] for image in images]


def distance(el1, el2):
    return np.linalg.norm(np.array(el1) - np.array(el2))


def classifier(train, test, method):
    if method not in get_methods():
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


def get_methods():
    return [canny, color_hist, random]


def voting(train, test):
    methods = get_methods()
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
            if method == "color_hist" and answers_to_image_1[answer]:
                answers_to_image_1[answer] += 1

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

    edges = canny(image)[1]
    plt.subplot(1, 4, 2, title="Canny")
    plt.imshow(edges, cmap="gray"), plt.axis("off")

    hists = color_hist(image)
    plt.subplot(1, 4, 3, title="Histogram")
    for hist, col in zip(hists, ('b', 'g', 'r')):
        plt.plot(range(len(hist)), hist, col)

    centers = random(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[1]
    plt.subplot(1, 4, 4, title="Random")
    rand_image = np.copy(image)
    for x0, y0 in centers:
        rand_image = cv2.circle(rand_image, (x0, y0), 1, (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(rand_image, cv2.COLOR_BGR2RGB)), plt.axis("off")
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


if __name__ == "__main__":
    data = load_paintings_from("./Paints/s", 4, 16)
    # for image in data[0]:
    #     draw_methods(image)

    size = 8
    # for size in range(5, 11):
    #     for train_indxs in combinations(range(16), size):
    #         x_train, x_test, y_train, y_test = split_data_cross(data, 16, size, train_indxs=train_indxs)
    #         train = [x_train, y_train]
    #         test = [x_test, y_test]
    #         # voting(train, [[test[0][0]], [test[1][0]]])
    #         classf = test_voting(train, test)
    #         if classf >= 0.5:
    #             print(f"{size} PAINTS IN TRAIN, {16 - size} PAINTS IN TEST, permutations = {train_indxs}")
    #             print(f"score = {classf}")
    #             print("*" * 10)

    # rnd.seed(84)  # 84 - 0.5833333
    # x_train, x_test, y_train, y_test = split_data_random(data, 16, size)
    # train = [x_train, y_train]
    # test = [x_test, y_test]
    # # voting(train, [[test[0][0]], [test[1][0]]])
    # classf = test_voting(train, test)
    # if classf >= 0.5:
    #     print(f"{size} PAINTS IN TRAIN, {16 - size} PAINTS IN TEST, random seed = {84}")
    #     print(f"score = {classf}")
    #     print("*" * 10)
    classes = {0: "И. И. Шишкин", 1: "И. К. Айвазовский",
               2: "П. Пикассо", 3: "В. И. Суриков"}

    seed = 52  # seed=52 methods:  {'canny': 0.40625, 'color_hist': 0.625, 'random': 0.4375} voting --> 0.6875
    # (with weight to Histogram)
    '''
    9 PAINTS IN TRAIN, 7 PAINTS IN TEST, seed=38
    methods:  {'canny': 0.32142857142857145, 'color_hist': 0.6785714285714286, 'random': 0.35714285714285715}
    voting --> 0.6428571428571429
    '''

    print(f"seed={seed}")
    rnd.seed(seed)
    # for size in range(1, 16):
    x_train, x_test, y_train, y_test = split_data_random(data, 16, size, seed=seed)
    train = [x_train, y_train]
    test = [x_test, y_test]
    res = test_methods(train, test)
    classf = test_voting(train, test)
    # if classf >= 0.6:
    print(f"{size} PAINTS IN TRAIN, {16 - size} PAINTS IN TEST, seed={seed}")
    print("methods: ", res)
    print(f"voting --> {classf}")
    print("*" * 10)

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
    #         res = {}
    #         for method in get_methods():
    #             answers = classifier(train, [[test_image], [true_answer]], method)
    #             print(f"method {method.__name__} found {classes[answers[0]]}")
    #         print(test_methods(train, [[test_image], [true_answer]]))
    #         plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)), plt.axis("off")
    #         plt.show()
    #     count += 1
    #     result.append(summ / count)
    #     print(f"{count} images --> {summ / count}")
    # plt.plot(range(1, len(result) + 1),  result), plt.xlabel("amount of test images"), plt.ylabel("score"), plt.title("Voting")
    # plt.show()

    # count = 1
    # plt.rcParams["figure.figsize"] = (10, 6)
    # index = 1
    # for train_image, ans in zip(x_train, y_train):
    #     plt.subplot(2,4,index)
    #     plt.imshow(cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)), plt.axis("off")
    #     index += 1
    #     if index > 8:
    #         plt.savefig(f"./Results/{count}.jpg")
    #         count += 1
    #         index = 1
