import cv2
import matplotlib.pyplot as plt


def template_matching(image_path, template_path):
    global count
    image = cv2.imread(image_path)
    if image is None:
        return "No image"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, top_left = cv2.minMaxLoc(res)
    cv2.rectangle(image, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res[res < 0.3] = 0
    return image, res


images_folder = "./my_images"
templates_folder = "./my_images_templates"
import os

images_names = os.listdir(path=images_folder)
templates_names = os.listdir(path=templates_folder)
images = [images_folder + "/" + images_name for images_name in images_names]
templates = [templates_folder + "/" + templates_name for templates_name in templates_names]

to_draw = []
count = 20
SAVE = False
amount = 1
columns = 4
for template_path in templates:
    for image_path in images:
        print(f'template {template_path} to image {image_path}')
        image, res = template_matching(image_path, template_path)
        to_draw.append((image, res))
        if len(to_draw) == columns:
            index = 1
            for i in range(len(to_draw)):
                plt.subplot(2, columns, index)
                index += 1
                plt.imshow(to_draw[i][0], cmap='gray'), plt.title(f'Рис. 2.{amount}'), plt.xticks([]), plt.yticks([])
                plt.subplot(2, columns, index)
                index += 1
                amount += 1
                plt.imshow(to_draw[i][1], cmap='gray'), plt.title(f'Рис. 2.{amount}'), plt.xticks([]), plt.yticks([])
                amount += 1
            if SAVE:
                plt.savefig(f"./results/results{count}.jpg", format='jpg', pad_inches=0)
                count += 1
            else:
                plt.show()
            to_draw = []
    if len(to_draw) > 0:
        index = 1
        for i in range(len(to_draw)):
            plt.subplot(2, len(to_draw), index)
            index += 1
            plt.imshow(to_draw[i][0], cmap='gray'), plt.title(f'Рис. 2.{amount}'), plt.xticks([]), plt.yticks([])
            plt.subplot(2, len(to_draw), index)
            index += 1
            amount += 1
            plt.imshow(to_draw[i][1], cmap='gray'), plt.title(f'Рис. 2.{amount}'), plt.xticks([]), plt.yticks([])
            amount += 1
        if SAVE:
            plt.savefig(f"./results/results{count}.jpg", format='jpg', pad_inches=0)
            count += 1
        else:
            plt.show()
    to_draw = []
