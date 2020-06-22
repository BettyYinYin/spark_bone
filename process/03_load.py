import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt

# data = np.load(r'2020-06-14\lumbar_train51\data.npz', allow_pickle=True)
# data = np.load(r'2020-06-14\lumbar_train51\one\data.npz', allow_pickle=True)

# data = np.load(r'2020-06-14\lumbar_train150\data.npz', allow_pickle=True)
data = np.load(r'2020-06-14\lumbar_train51\one\data.npz', allow_pickle=True)
images = data['data']
land_marks = data['land_marks']
disease_class = data['disease_class']

predict_data = np.load('predict-1999.npz')
p_land_mark = predict_data['land_mark']


print(images.shape)
print(land_marks.shape)
print(disease_class.shape)
# print(disease_class[6])

for step, image in enumerate(images):
  # print(image.shape)
  # print(land_marks[step])
  # print(disease_class[step])
  if step == 0:
    print(image)
  # 标出landmark
  for i in range(int(len(land_marks[step]) / 2)):
    # if step == 0:
    #   mark = [123.05636, 68.5342, 121.48644, 79.98666, 120.01607, 90.83583, 117.53186, 101.58011, 117.65546, 114.426, 115.63515, 124.25839, 114.67534, 137.1989, 112.10615, 148.60127, 114.45643, 159.38953, 113.741295, 171.37212, 117.1571, 181.98251]
    #   cv2.circle(image, (int(mark[2*i]),int(int(mark[2*i+1]))), 2, (255, 255, 255), -1)
    #   cv2.circle(image, (int(land_marks[step][2*i]),int(int(land_marks[step][2*i+1]))), 2, (0, 0, 255), -1)
    # else:
    cv2.circle(image, (int(land_marks[step][2*i]),int(int(land_marks[step][2*i+1]))), 2, (255, 255, 255), -1)
    cv2.circle(image, (int(p_land_mark[step][2*i]),int(int(p_land_mark[step][2*i+1]))), 2, (0, 0, 255), 3)

  cv2.imshow('0', image)
  cv2.waitKey(0)
  
