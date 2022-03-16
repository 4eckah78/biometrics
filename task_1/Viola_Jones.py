import cv2
import matplotlib.pyplot as plt


def detect_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "No image"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    # 1.01, 5 for images from ORL DataBase
    # 1.2, 5 for my faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y + w, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.8, 7)
        # 1.01, 1 for images from ORL DataBase
        # 1.8, 7 for my faces
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


images_folder = "./my_images"
import os

images_names = os.listdir(path=images_folder)
images = [images_folder + "/" + images_name for images_name in images_names]

to_draw = []
count = 1
SAVE = True
amount = 1
for image_path in images:
    print(f'detecting {image_path}...')
    image = detect_face(image_path)
    to_draw.append(image)
    if len(to_draw) == 8:
        index = 1
        for i in range(len(to_draw)):
            plt.subplot(2, 4, index)
            index += 1
            plt.imshow(to_draw[i], cmap='gray'), plt.title(f'Рис. 4.{amount}'), plt.xticks([]), plt.yticks([])
            amount += 1
        if SAVE:
            plt.savefig(f"./viola_results/results{count}.jpg", format='jpg', pad_inches=0)
            count += 1
        else:
            plt.show()
        to_draw = []
if len(to_draw) > 0:
    index = 1
    for i in range(len(to_draw)):
        plt.subplot(2, 4, index)
        index += 1
        plt.imshow(to_draw[i], cmap='gray'), plt.title(f'Рис. 4.{amount}'), plt.xticks([]), plt.yticks([])
        amount += 1
    if SAVE:
        plt.savefig(f"./viola_results/results{count}.jpg", format='jpg', pad_inches=0)
        count += 1
    else:
        plt.show()
