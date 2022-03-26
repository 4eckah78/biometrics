import cv2
import os

data_folder = "./orl_faces_high_cloaked/s"
for i in range(1, 41):
    for j in range(1, 11):
        image = cv2.cvtColor(cv2.imread(f"{data_folder}{i}/{j}_cloaked.png"), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{data_folder}{i}/{j}_cloaked.jpg", image)
        os.remove(f"{data_folder}{i}/{j}_cloaked.png")
        os.remove(f"{data_folder}{i}/{j}.jpg")

# for i in range(1, 41):
#     os.system(f'fawkes -d ./orl_faces_cloaked/s{i} --mode high')