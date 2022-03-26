import numpy as np
import random as rnd
from scipy.fftpack import dct as sc_dct
import cv2
import matplotlib.pyplot as plt


def histogram(image, BINS=30, SHOW=False):
    hist, bin_edges = np.histogram(image, bins=BINS, range=(0, 1))
    if SHOW:
        plt.imshow(image, cmap="gray"), plt.xticks([]), plt.yticks([])
        plt.title("Original")
        plt.show()
        plt.hist(image.flatten(), bins=BINS, range=(0, 1))
        plt.title(f"hist, BINS={BINS}")
        plt.show()
    return hist


def dft(image, p=13, SHOW=False):
    f = np.fft.fft2(image)
    f = np.abs(f[0:p, 0:p])
    zigzag = []
    for index in range(1, p + 1):
        slice = [row[:index] for row in f[:index]]
        diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    if SHOW:
        plt.imshow(f, cmap="gray")
        plt.title(f"dft, p={p}")
        plt.show()
    return zigzag[1:]


def dct(image, p=13, SHOW=False):
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
    if SHOW:
        plt.imshow(c, cmap="gray")
        plt.title(f"dct, p={p}")
        plt.show()
    return zigzag


# def gradient(image, window_width=2, SHOW=False):
#     height = image.shape[0]
#     step, low = 0, 0
#     up = window_width
#     result = []
#
#     while up <= height:
#         # window = image[low:up, :]
#         dist = distance(image[low:low + (up - low) // 2, :], image[low + (up - low) // 2:up - window_width % 2, :])
#         # cv2.rectangle(image, (0, low), (image.shape[1], up), 0, 1)
#         # cv2.line(image, (0, low+(up - low)//2), (image.shape[1], low+(up - low)//2), 60, 1)
#         # plt.imshow(image, cmap="gray")
#         # plt.show()
#         result.append(dist)
#         step += 1
#         low = step * window_width
#         up = (step + 1) * window_width
#     result = np.array(result)
#     if SHOW:
#         plt.plot(range(0, len(result)), result)
#         plt.title(f"gradient, W={window_width}")
#         plt.show()
#     return result


def gradient(image, n = 2):
    shape = image.shape[0]
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r, :]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result


def scale(image, scale=0.35, SHOW=False):
    h = image.shape[0]
    w = image.shape[1]
    new_size = (int(w * scale), int(h * scale))
    if SHOW:
        plt.imshow(cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), cmap="gray")
        plt.title(f"scale, sc={scale}")
        plt.show()
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def load_faces_from(data_folder, type=".pgm"):
    data_faces = []
    data_target = []
    for i in range(1, 41):
        for j in range(1, 11):
            image = cv2.imread(f"{data_folder}{i}/{j}{type}")
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                data_faces.append(image / 255)
                data_target.append(i - 1)
    return [data_faces, data_target]


def draw_splited_data(x_train, x_test, images_per_person_in_train, rows=5):
    K = int((len(x_train) + len(x_test)) / 10)
    i, j = 0, 0
    plt.rcParams.update({'font.size': 7})
    # AMOUNT_OF_ROWS = 8
    # list_of_classes = [AMOUNT_OF_ROWS for _ in range(K // AMOUNT_OF_ROWS)]
    # list_of_classes.append(K % AMOUNT_OF_ROWS)
    list_of_classes = [rows]
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
        indexes = list(range(i, i + images_per_person))
        train_indexes = rnd.sample(indexes, images_per_person_in_train)
        x_train.extend([data[0][index] for index in train_indexes])
        y_train.extend([data[1][index] for index in train_indexes])

        test_indexes = set(indexes) - set(train_indexes)
        x_test.extend([data[0][index] for index in test_indexes])
        y_test.extend([data[1][index] for index in test_indexes])

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


def fit(train, test, method, SHOW=False):
    if method not in [histogram, dft, dct, gradient, scale]:
        return []
    param = (0, 0, 0)
    if method == histogram:
        param = (8, 30, 3)
    if method == dft or method == dct:
        param = (6, 30, 3)
    if method == gradient:
        param = (2, 30, 3)
    if method == scale:
        param = (0.05, 0.5, 0.05)

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

    if SHOW:
        plt.plot(stat[0], stat[1], label=method.__name__)
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
        for image, person in zip(train[0], train[1]):
            if person == best_size[0]:
                plt.subplot(1, 2, 1)
                plt.imshow(test[0][i], cmap="gray"), plt.xticks([]), plt.yticks([]), plt.title('Query Image')
                plt.subplot(1, 2, 2)
                plt.imshow(image, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.title('Result')
                plt.suptitle(f"Voting")
                plt.show()
                break

    return voted_answers


def test_voting(train, test, parameters):
    res = voting(train, test, parameters)
    sum = 0
    for i in range(len(test[0])):
        if test[1][i] == res[i]:
            sum += 1
    return sum / len(test[0])


def cross_validation(data, etalons_range=[5,6], SHOW=False):
    methods = [histogram, dft, dct, gradient, scale]
    res = []
    start, end = etalons_range
    faces_in_train_stats = {}
    vote_stats = []
    for size in range(start, end):
        print(f"{size} FACES IN TRAIN, {10 - size} FACES IN TEST")
        X_train, X_test, y_train, y_test = split_data(data, size)
        train = mesh_data([X_train, y_train])
        test = mesh_data([X_test, y_test])
        parameters = {}
        for method in methods:
            print("-"*50)
            print(f"fitting params for {method.__name__}...")
            results = fit(train, test, method, SHOW=SHOW)
            if method.__name__ in faces_in_train_stats:
                faces_in_train_stats[method.__name__].append(results[0][1])
            else:
                faces_in_train_stats[method.__name__] = [results[0][1]]
            parameters[method.__name__] = results[0][0]
            print(f"for {method.__name__} got param {results[0][0]} with score {results[0][1]}")
        if SHOW:
            plt.title(f"train size={size}"), plt.legend(loc='best'), plt.xlabel("parameter"), plt.ylabel("score")
            # plt.savefig(f"./cross_valid/dummy_gradient_train_size_{size}.jpg")
            plt.show()
            # draw_splited_data(X_train, X_test, size)
        print("-" * 50)
        print(f"Result of fitting for all methods: {parameters}")
        print("Voting...")
        classf = test_voting(train, test, parameters)
        vote_stats.append(classf)
        print(f"voted accuracy: {classf}")
        res.append([parameters, classf])

    if SHOW:
        for method, stats in faces_in_train_stats.items():
            plt.plot(range(1, len(stats) + 1), stats, label=method)
            plt.legend(loc='best'), plt.xlabel("train size"), plt.ylabel("score")
        plt.show()
        plt.plot(range(1, len(vote_stats) + 1), vote_stats), plt.title("Voting")
        plt.xlabel("train size"), plt.ylabel("voted_score")
        plt.show()

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
    # parameters, train_size = cross_validation(data)
    train_size = 5
    parameters = {'histogram': 17, 'dft': 18, 'dct': 9, 'gradient': 2, 'scale': 0.4}
    x_train, x_test, y_train, y_test = split_data(data, train_size, DRAW=True)
    train = mesh_data([x_train, y_train])
    test = mesh_data([x_test, y_test])
    return voting(train, test, parameters)


if __name__ == "__main__":
    print("Accessing to database...")
    data = load_faces_from("./orl_faces/s")
    print(f"Database is uploaded: ORL faces, {len(data[0])} images, 40 classes, 10 images in each class")
    print("="*50)
    #
    # print("Check train for each method:")
    # x_train, x_test, y_train, y_test = split_data(data, 1)
    # train = mesh_data([x_train, y_train])
    # test = mesh_data([x_test, y_test])
    #
    # for method, param in [(histogram, 18), (dft, 18), (dct, 8), (gradient, 8), (scale, 0.23)]:
    #     classf = test_classifier(train, train, method, param)
    #     print(f"for {method.__name__} got {int(classf)*100}% score")
    # print("="*50)
    #
    # print("Check test for each method:")
    # for method, param in [(histogram, 28), (dft, 18), (dct, 18), (gradient, 2), (scale, 0.23)]:
    #     classf = test_classifier(train, test, method, param)
    #     print(f"for {method.__name__} got {classf} score")
    # print("="*50)
    #
    # print("Cross-validation...")
    # parameters, train_size, score = cross_validation(data, etalons_range=[1, 3], SHOW=False)
    # print("!"*50)
    # print(f"best train size: {train_size}\n with parameters: {parameters}\n and score: {score}")
    # print("!" * 50)

    # vote_classifier(data)

    train_size = 5
    parameters = {'histogram': 17, 'dft': 18, 'dct': 9, 'gradient': 2, 'scale': 0.4}
    x_train, x_test, y_train, y_test = split_data(data, train_size, DRAW=True)
    # cloaked_data = load_faces_from("./orl_faces_high_cloaked/s", type='_cloaked.jpg')
    # _, x_test, _, y_test = split_data(cloaked_data, 5)
    # train = mesh_data([x_train, y_train])
    # test = mesh_data([x_test, y_test])
    # voting(train, test, parameters)


    masked_data = load_faces_from("./orl_faces_with_mask/s", type='-with-mask.jpg')
    test = [masked_data[0], masked_data[1]]
    voting(train, test, parameters)

'''
dft = 15
dct = 8
hist = 23
scale = 0.2
gradient = 4
dummy_grad = 2
faces_in_train = 5
'''