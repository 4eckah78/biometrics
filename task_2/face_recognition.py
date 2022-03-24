import numpy as np
import random as rnd
from scipy.fftpack import dct as sc_dct
import cv2
import matplotlib.pyplot as plt


def histogram(image, BINS=30):
    histogram, bin_edges = np.histogram(image, bins=BINS, range=(0, 1))
    # plt.hist(image.flatten(), bins=BINS, range=(0, 1))
    return histogram


def dft(image, p=13):
    f = np.fft.fft2(image)
    f = np.abs(f[0:p, 0:p])
    zigzag = []
    for index in range(1, p + 1):
        slice = [row[:index] for row in f[:index]]
        diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    return zigzag[1:]


def dct(image, p=13):
    c = sc_dct(image, axis=1)
    c = sc_dct(c, axis=0)
    c = c[0:p, 0:p]
    zigzag = []
    for index in range(1, p + 1):
        slice = [row[:index] for row in c[:index]]
        diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    return zigzag


def gradient(image, window_width=2):
    height = image.shape[0]
    step, low = 0, 0
    up = window_width
    result = []

    while up <= height:
        # window = image[low:up, :]
        dist = distance(image[low:low + (up - low) // 2, :], image[low + (up - low) // 2:up - window_width % 2, :])
        # cv2.rectangle(image, (0, low), (image.shape[1], up), 0, 1)
        # cv2.line(image, (0, low+(up - low)//2), (image.shape[1], low+(up - low)//2), 60, 1)
        # plt.imshow(image, cmap="gray")
        # plt.show()
        result.append(dist)
        step += 1
        low = step * window_width
        up = (step + 1) * window_width
    result = np.array(result)
    return result


def scale(image, scale=0.35):
    h = image.shape[0]
    w = image.shape[1]
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def load_faces_from(data_folder):
    data_faces = []
    data_target = []
    for i in range(1, 41):
        for j in range(1, 11):
            image = cv2.cvtColor(cv2.imread(f"{data_folder}{i}/{j}.pgm"), cv2.COLOR_BGR2GRAY)
            data_faces.append(image / 255)
            data_target.append(i - 1)
    return [data_faces, data_target]


def draw_splited_data(x_train, x_test, images_per_person_in_train):
    K = int((len(x_train) + len(x_test)) / 10)
    i, j = 0, 0
    plt.rcParams.update({'font.size': 7})
    AMOUNT_OF_ROWS = 8
    list_of_classes = [AMOUNT_OF_ROWS for _ in range(K // AMOUNT_OF_ROWS)]
    list_of_classes.append(K % AMOUNT_OF_ROWS)
    for rows in list_of_classes:
        draw_train = images_per_person_in_train
        index = 1
        draw_test = 10 - images_per_person_in_train
        while index != rows * 10 + 1:
            if draw_train > 0:
                plt.subplot(rows, 10, index)
                plt.imshow(x_train[i], cmap='gray'), plt.xticks([]), plt.yticks([])
                if index <= 10:
                    plt.title('Train')
                draw_train -= 1
                i += 1
                index += 1
            elif draw_test > 0:
                plt.subplot(rows, 10, index)
                plt.imshow(x_test[j], cmap='gray'), plt.xticks([]), plt.yticks([])
                if index <= 10:
                    plt.title('Test')
                draw_test -= 1
                j += 1
                index += 1
            else:
                draw_train = images_per_person_in_train
                draw_test = 10 - images_per_person_in_train
        plt.show()


def split_data(data, images_per_person_in_train=5, DRAW=False):
    images_per_person = 10
    amount_of_images = len(data[0])

    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, amount_of_images, images_per_person):
        x_train.extend(data[0][i: i + images_per_person_in_train])
        y_train.extend(data[1][i: i + images_per_person_in_train])

        x_test.extend(data[0][i + images_per_person_in_train: i + images_per_person])
        y_test.extend(data[1][i + images_per_person_in_train: i + images_per_person])

    if DRAW:
        draw_splited_data(x_train, x_test, images_per_person_in_train)

    return x_train, x_test, y_train, y_test


def mesh_data(data):
    indexes = rnd.sample(range(0, len(data[0])), len(data[0]))
    return [data[0][index] for index in indexes], [data[1][index] for index in indexes]


def create_feature(images, method, parameter):
    return [method(image, parameter) for image in images]


def distance(el1, el2):
    return np.linalg.norm(np.array(el1) - np.array(el2))


def classifier(train, test, method, parameter):
    if method not in [histogram, dft, dct, gradient, scale]:
        return []
    featured_train = create_feature(train[0], method, parameter)
    featured_test = create_feature(test[0], method, parameter)
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


def test_classifier(train, test, method, parameter):
    if method not in [histogram, dft, dct, gradient, scale]:
        return []
    answers = classifier(train, test, method, parameter)
    correct_answers = 0
    for i in range(len(answers)):
        if answers[i] == test[1][i]:
            correct_answers += 1
    return correct_answers / len(answers)


def fit(train, test, method):
    if method not in [histogram, dft, dct, gradient, scale]:
        return []
    param = (0, 0, 0)
    if method == histogram:
        param = (8, 64, 5)
    if method == dft or method == dct:
        param = (6, 20, 2)
    if method == gradient:
        param = (2, 20, 1)
    if method == scale:
        param = (0.03, 0.5, 0.05)

    best_param = param[0]
    classf = test_classifier(train, test, method, best_param)
    stat = [[best_param], [classf]]

    for i in np.arange(param[0] + param[2], param[1], param[2]):
        new_classf = test_classifier(train, test, method, i)
        stat[0].append(i)
        stat[1].append(new_classf)
        if new_classf > classf:
            classf = new_classf
            best_param = i

    return [best_param, classf], stat


def voting(train, test, parameters):
    methods = [histogram, dft, dct, gradient, scale]
    res = {}
    for method in methods:
        res[method.__name__] = classifier(train, test, method, parameters[method.__name__])
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


def test_voting(train, test, parameters):
    res = voting(train, test, parameters)
    sum = 0
    for i in range(len(test[0])):
        if test[1][i] == res[i]:
            sum += 1
    return sum / len(test[0])


def cross_validation(data):
    methods = [histogram, dft, dct, gradient, scale]
    res = []
    start = 1
    end = 7
    for size in range(start, end):
        print(f"{size} FACES IN TRAIN, {10 - size} FACES IN TEST")
        X_train, X_test, y_train, y_test = split_data(data, size)
        train = mesh_data([X_train, y_train])
        test = mesh_data([X_test, y_test])
        parameters = {}
        for method in methods:
            print("-"*50)
            print(f"fitting params for {method.__name__}...")
            results = fit(train, test, method)
            parameters[method.__name__] = results[0][0]
            print(f"for {method.__name__} got param {results[0][0]} with score {results[0][1]}")
        print("-" * 50)
        print(f"Result of fitting for all methods: {parameters}")
        print("Voting...")
        classf = test_voting(train, test, parameters)
        print(f"voted accuracy: {classf}")
        res.append([parameters, classf])

    best_res = [[], 0]
    best = 0
    for i in range(end - start):
        if res[i][1] > best:
            best = res[i][1]
            best_res[0] = res[i][0]
            best_res[1] = i + start
    best_res.append(best)
    return best_res


def vote_classifier(data):
    parameters, train_size = cross_validation(data)
    x_train, y_train, x_test, _ = split_data(data, train_size)
    train = mesh_data([x_train, y_train])
    return voting(train, x_test, parameters)


if __name__ == "__main__":
    data = load_faces_from("./orl_faces/s")
    parameters, train_size, score = cross_validation(data)
    print("!"*50)
    print(f"best train size: {train_size}\n with parameters: {parameters}\n and score: {score}")
    print("!" * 50)
    x_train, x_test, y_train, y_test = split_data(data, train_size)
    train = mesh_data([x_train, y_train])
    test = (x_test, y_test)
    v = voting(train, test, parameters)
    print("=" * 50)
    print("true answers:\n", y_test[:10])
    print("voted answers:\n", v[:10])
