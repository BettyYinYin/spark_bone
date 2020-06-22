import cv2
import numpy as np

def crop(image):
  gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
  gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

  gradient = cv2.subtract(gradX, gradY)
  gradient = cv2.convertScaleAbs(gradient)
  # print('gradient', gradient)
  # cv2.imshow('0', gradient)
  # cv2.waitKey(0)

  # blur and threshold the image
  blurred = cv2.blur(gradient, (9, 9))
  (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
  # print('thresh', thresh)
  # cv2.imshow('0', thresh)
  # cv2.waitKey(0)


  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
  closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
  # cv2.imshow('0', closed)
  # cv2.waitKey(0)

  # perform a series of erosions and dilations
  closed = cv2.erode(closed, None, iterations=4)
  closed = cv2.dilate(closed, None, iterations=4)
  # cv2.imshow('0', closed)
  # cv2.waitKey(0)

  (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

  # compute the rotated bounding box of the largest contour
  rect = cv2.minAreaRect(c)
  print('rect', rect)
  box = np.int0(cv2.boxPoints(rect))
  print('box', box)
  # box = np.int0(cv2.cv.BoxPoints(rect))

  # draw a bounding box arounded the detected barcode and display the image
  cv2.drawContours(images[0], [box], -1, (255, 255, 255), 3)
  cv2.imshow("Image", image)
  cv2.imwrite("contoursImage2.jpg", image)
  cv2.waitKey(0)

  Xs = [i[0] for i in box]
  Ys = [i[1] for i in box]
  x1 = max(min(Xs), 0)
  x2 = min(max(Xs), images[0].shape[1]-1)
  y1 = max(min(Ys), 0)
  y2 = min(max(Ys), images[0].shape[0]-1)
  cropImg = images[0][y1:y2, x1:x2]
  return cropImg

if __name__ == "__main__":
  data = np.load(r'2020-06-14\lumbar_train51\one\data.npz', allow_pickle=True)
  images = data['data']
  cv2.imshow('crop', crop(images[0]))
  cv2.waitKey(0)