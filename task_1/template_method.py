import cv2
import matplotlib.pyplot as plt

count = 1


def template_matching(image_path, template_path):
    global count
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, top_left = cv2.minMaxLoc(res)

    cv2.rectangle(image, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 1)

    # plt.figure(figsize=(5, 2))
    # plt.imshow(template, cmap='gray')
    # plt.title('template')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image, cmap='gray')
    # plt.title('Result')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # cv2.imwrite(f"my_result{count}.jpg", image)
    # # print(res.shape)
    # count += 1
    plt.subplot(131)
    plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(132)
    plt.imshow(image, cmap='gray')
    plt.title('Detected Area')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle('Template matching')
    plt.subplot(133)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.savefig(f"result{count}.jpg", format='jpg')
    count += 1
    # cv2.imshow('template', template)
    # cv2.imshow('Result', image)
    # cv2.waitKey(0)


images_folder = "./images"
templates_folder = "./images_templates"
import os

images_names = os.listdir(path=images_folder)
templates_names = os.listdir(path=templates_folder)
images = [images_folder + "/" + images_name for images_name in images_names]
templates = [templates_folder + "/" + templates_name for templates_name in templates_names]

for template_path in templates:
    for image_path in images:
        print(f'template {template_path} to image {image_path}')
        template_matching(image_path, template_path)

# template_matching("../ORL_face_database/Face.jpg", "../ORL_face_database/Temp.jpg")
