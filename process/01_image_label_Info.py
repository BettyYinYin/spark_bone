import SimpleITK as sitk
import glob
import json
import numpy as np

def saveLabelJson(readPath, saveDir):
  with open(readPath) as f:
    labelJson = json.load(f)
    np.savez(saveDir+'/labelJson.npz', data=labelJson)

def saveImageInfo(readPath, saveDir):
  imagesInfo = {}
  # studyUids = []
  list_tag = ['0020|000d', '0020|000e', '0008|0018']
  for sample in glob.glob(readPath):
    # print('sample', sample)
    for imagePath in glob.glob(sample + '/*'):
      r = dicom_metainfo(imagePath, list_tag)
      if r[0] == '1.2.3.4.5.90760256.989':
        print('seriesUid', r[1])
        print('instanceUid', r[2])
      if r[0] not in imagesInfo:
        # 以studyUid作为key存储一个study文件夹的所有图像
        imagesInfo[r[0]] = []  
      
      imagesInfo[r[0]].append({
        'path': imagePath,
        'seriesUid': r[1],
        'instanceUid': r[2]
      })
    print('end')
  np.savez(saveDir + '/imagesInfo.npz', data=imagesInfo)
  # np.savetxt(saveDir+ '/studyUids.txt', studyUids)

def dicom_metainfo(dicm_path, list_tag):
    '''
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    '''
    try:

      reader = sitk.ImageFileReader()
      reader.LoadPrivateTagsOn()
      reader.SetFileName(dicm_path)
      reader.ReadImageInformation()
      print('ReadImageInformation', reader.ReadImageInformation())
      return [reader.GetMetaData(t) for t in list_tag]
    except:
      print('读取文件异常')
      return ['null', 'null', 'null']

# 处理train51数据
label_json_src = r'2020-06-14\lumbar_train51_annotation.json'
image_src = r'2020-06-14\lumbar_train51\train\study*'
saveDir = r'2020-06-14\lumbar_train51'

saveImageInfo(image_src, saveDir)
saveLabelJson(label_json_src, saveDir)


# 处理train150数据
# label_json_src = r'2020-06-14\lumbar_train150_annotation.json'
# image_src = r'2020-06-14\lumbar_train150\study*'
# saveDir = r'2020-06-14\lumbar_train150'

# saveImageInfo(image_src, saveDir)
# saveLabelJson(label_json_src, saveDir)

    
