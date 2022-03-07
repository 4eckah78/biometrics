import cv2
import matplotlib.pyplot as plt

count = 1

def detect_face(image_path):
    global count

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        roi_gray = gray[y:y + w, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='gray')
    plt.show()
    # plt.savefig(f"./viola_results/results{count}.jpg", format='jpg')
    count += 1


images_folder = "./images"
import os

images_names = os.listdir(path=images_folder)
images = [images_folder + "/" + images_name for images_name in images_names]

for image_path in images:
    print(f'detecting {image_path}...')
    detect_face(image_path)
