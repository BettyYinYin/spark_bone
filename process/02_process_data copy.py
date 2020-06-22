import numpy as np
import SimpleITK as sitk
import cv2

# 图片转换为array
def dicom2array(dcm_path):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
        image = sitk.Cast(image, sitk.sitkUInt8)
        # image = sitk.Cast(image, sitk.sitkFloat64)
    img_x = sitk.GetArrayFromImage(image)[0]
    return img_x

# 图片转换为统一大小
def cv2_letterbox_image(image, expected_size=(224, 224)):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left

    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    # new_img = new_img.astype('float32')
    return new_img, scale, top, left

# 定义脊柱的顺序，landmarks按照这样的顺序存储
identifications = {
  'T12-L1': 0,
  'L1': 1,
  'L1-L2': 2,
  'L2': 3,
  'L2-L3': 4,
  'L3': 5,
  'L3-L4': 6,
  'L4': 7,
  'L4-L5': 8,
  'L5': 9,
  'L5-S1': 10, 
}

# 定义脊柱的病变类型
rank = {
  'v1': 0,
  'v2': 1,
  'v3': 2,
  'v4': 3,
  'v5': 4
}

def processData(imageJsonPath, labelJsonPath, saveDir):
  # 加载图片信息和标注信息
  imageJson = np.load(imageJsonPath, allow_pickle=True)['data'][()]
  labelJson = np.load(labelJsonPath, allow_pickle=True)['data']

  # 初始化
  # (N, H, W)
  imageData = []
  # (N, 11, 22)
  land_marks = np.ones((len(labelJson), 11, 2), dtype='float32')
  # (N, 11, 5)
  disease_class = np.zeros((len(labelJson), 11, 5), dtype='float32')
  # 遍历标注文件，找出对应的图像
  for labelIndex, label in enumerate(labelJson):
    studyUid = label['studyUid']
    seriesUid = label['data'][0]['seriesUid']
    instanceUid = label['data'][0]['instanceUid']
    # 根据studyUid取出一个样本文件夹对应的所有图像
    images = imageJson[studyUid]
    points = label['data'][0]['annotation'][0]['data']['point']

    # 处理land_marks像素点 -> (N, 22)
    # 病变类型，可能会输出multi-label -> (N, 11, 5)
    for point in points:
      land_marks[labelIndex][identifications[point['tag']['identification']]] = point['coord']
      # land_mark[identifications[point['tag']['identification']]] = point['coord']
      try:
        # 如果没有disc字段就取vertebra字段
        disease = point['tag']['disc']
        # 如果disc为空就取vertebra
        if disease == '':
          disease = point['tag']['vertebra']
      except Exception as err:
        disease = point['tag']['vertebra']
      disease = disease.split(',')
      for i in disease:
        try:
          # disease_c[identifications[point['tag']['identification']]][rank[i]] = 1
          disease_class[labelIndex][identifications[point['tag']['identification']]][rank[i]] = 1
        except Exception as e:
          print('此节点标注有误,studyuid: %s' % labelJson[labelIndex]['studyUid'])
    # land_marks.append(land_mark)
    # disease_class.append(disease_c)
    # image_seriesUids = []


    for image in images:
      if image['seriesUid'] == seriesUid and image['instanceUid'] == instanceUid:
        img_x = dicom2array(image['path'])
        imageData.append(img_x)
  land_marks = np.array(land_marks).reshape(-1, 22)
  resize_images = []
  resize_land_marks = []
  for step, image in enumerate(imageData):
    # print(image.shape)
    new_img, scale, top, left = cv2_letterbox_image(image)
    # print(new_img.shape)
    resize_land_mark = [val * scale + left if i % 2 == 0 else val * scale + top for i, val in enumerate(land_marks[step])]
    # cv2.imshow('0', new_img)
    # cv2.waitKey(0)
    resize_images.append(new_img)
    resize_land_marks.append(resize_land_mark)
  # print('disease_class', disease_class[:5])
  print('resize_images len', len(resize_images))
  # 存储原始大小的图片，landmarks和病变类型
  np.savez(saveDir + '/imageData.npz', data=imageData, land_marks=land_marks, disease_class=disease_class)
  # 存储resize后的图片，landmarks和病变类型
  np.savez(saveDir + '/data.npz', data=resize_images, land_marks=resize_land_marks, disease_class=disease_class)

# 处理51数据
# print('开始处理51')
# imageJsonPath = r'2020-06-14\lumbar_train51\imagesInfo.npz'
# labelJsonPath = r'2020-06-14\lumbar_train51\labelJson.npz'
# saveDir = r'2020-06-14\lumbar_train51\one'
# processData(imageJsonPath, labelJsonPath, saveDir)
# print('51处理完成')

# 处理150数据
print('开始处理150')
imageJsonPath = r'2020-06-14\lumbar_train150\imagesInfo.npz'
labelJsonPath = r'2020-06-14\lumbar_train150\labelJson.npz'
saveDir = r'2020-06-14\lumbar_train150\one'
processData(imageJsonPath, labelJsonPath, saveDir)
print('150处理完成')
