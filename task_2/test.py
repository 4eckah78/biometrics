import cv2
import numpy as np
from scipy.fftpack import dct as sc_dct
import matplotlib.pyplot as plt

from face_recognition import *


def show_dft(image, p=13):
    f = np.fft.fft2(image)
    return np.abs(f[0:p, 0:p])


def show_dct(image, p=13):
    c = sc_dct(image, axis=1)
    c = sc_dct(c, axis=0)
    return c[0:p, 0:p]


def methods_to_image_with_params(methods, image, params):
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap="gray"), plt.title("Original")
    for index, method, param in zip(range(2, 7), methods, params):
        plt.subplot(2, 3, index)
        if method == histogram:
            plt.hist(image.flatten(), bins=param, range=(0, 1))
        elif method == gradient:
            res = method(image, param)
            plt.plot(range(0, len(res)), res)
        else:
            plt.imshow(method(image, param), cmap="gray")
        plt.title(f'{method.__name__}, {param}')
    plt.show()


def my_classifier(train, test, method, parameter):
    if method not in [histogram, dft, dct, gradient, scale]:
        return []
    featured_train = create_feature(train[0], method, parameter)
    featured_test = create_feature(test[0], method, parameter)
    ans = []
    for test_element in featured_test:
        min_el = [100000, -1]
        for i in range(len(featured_train)):
            dist = distance(test_element, featured_train[i])
            if dist < min_el[0]:
                min_el = [dist, i]
        if min_el[1] < 0:
            ans = train[0][0]
        else:
            ans = train[0][min_el[1]]
    return ans


def classify_image(train, query_image, methods, params, train_size=5):
    plt.subplot(2, 3, 1)
    plt.imshow(query_image, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.title(f'Query Image')

    index = 2
    for method, parameter in zip(methods, params):
        answer = my_classifier(train, ([query_image], [0]), method, parameter)
        plt.subplot(2, 3, index)
        index += 1
        plt.imshow(answer, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.title(f"{method.__name__}, {parameter}")
    plt.suptitle(f"train size={train_size}")
    plt.show()


if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("1-with-mask.jpg"), cv2.COLOR_BGR2GRAY)
    image = image / 255

    show_methods = [histogram, show_dft, show_dct, gradient, scale]
    methods = [histogram, dft, dct, gradient, scale]
    params = [23, 15, 8, 4, 0.2]
    train_size = 1
    # methods_to_image_with_params(show_methods, image, params)

    data = load_faces_from("./orl_faces/s")
    x_train, x_test, y_train, y_test = split_data(data, train_size)
    train = mesh_data([x_train, y_train])
    test = mesh_data([x_test, y_test])

    # x_train, y_train = x_train[12], y_train[12]
    # train = ([x_train], [y_train])
    # train_image = train[0][0]
    # classify_image(train, train_image, methods, params)

    # cloaked_data = load_faces_from("./orl_faces_high_cloaked/s", type='_cloaked.jpg')
    # _, x_test, _, y_test = split_data(cloaked_data, train_size)

    # masked_data = load_faces_from("./orl_faces_with_mask/s", type='-with-mask.jpg')
    # test = [masked_data[0], masked_data[1]]

    for query_image, true_answer in zip(test[0], test[1]):
        # draw_train = [image for image, i in zip(x_train, y_train) if i == true_answer]
        # draw_test = [image for image, i in zip(x_test, y_test) if i == true_answer]
        # i, j = 0, 0
        # for ind in range(1, 11):
        #     plt.subplot(1, 10, ind)
        #     if i < len(draw_train):
        #         plt.imshow(draw_train[i], cmap="gray"), plt.xticks([]), plt.yticks([]), plt.title('Train')
        #         i += 1
        #     elif j < len(draw_test):
        #         plt.imshow(draw_test[j], cmap="gray"), plt.xticks([]), plt.yticks([]), plt.title("Test")
        #         j += 1
        # plt.show()
        classify_image(train, query_image, methods, params)

    # start, end = 1, 10
    # faces_in_train_stats = {}
    # for size in range(start, end):
    #     print(f"{size} FACES IN TRAIN, {10 - size} FACES IN TEST")
    #     X_train, X_test, y_train, y_test = split_data(data, size)
    #     train = mesh_data([X_train, y_train])
    #     test = mesh_data([X_test, y_test])
    #     params = [23, 15, 8, 4, 0.2]
    #     for method, parameter in zip(methods, params):
    #         classf = test_classifier(train, test, method, parameter)
    #         if method.__name__ in faces_in_train_stats:
    #             faces_in_train_stats[method.__name__].append(classf)
    #         else:
    #             faces_in_train_stats[method.__name__] = [classf]
    # for method, stats in faces_in_train_stats:
    #     plt.plot(range(1, len(stats) + 1), stats, label=method.__name__)
    #     plt.legend(loc='best'), plt.xlabel("train size"), plt.ylabel("score")
    # plt.show()
