import os
import cv2


data_folder = "./orl_faces_with_masks/s"
# for i in range(1, 41):
#     for j in range(1, 11):
#         image = cv2.cvtColor(cv2.imread(f"{data_folder}{i}/{j}.pgm"), cv2.COLOR_BGR2GRAY)
#         cv2.imwrite(f"{data_folder}{i}/{j}.jpg", image)
#         os.remove(f"{data_folder}{i}/{j}.pgm")

for i in range(1, 41):
    for j in range(1, 11):
        # os.system(f'face-mask {data_folder}{i}/{j}.jpg')
        os.remove(f"{data_folder}{i}/{j}.jpg")
