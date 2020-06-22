import numpy as np 
import tensorflow as tf

image = tf.placeholder(tf.float32, [None, 224, 224, 1])

def Model(image):
  # 第一层卷积
  net_1 = tf.layers.conv2d(
    image, 
    kernel_size=7, 
    filters=64,
    strides=2,
    activation=tf.nn.relu, 
    padding="same"
  )
  net_1 = tf.layers.max_pooling2d(net_1, 3, 2, padding='same')
  net_1 = tf.nn.relu(net_1)

  # 第二层卷积
  net_2 = tf.layers.conv2d(
    net_1, 
    kernel_size=3, 
    filters=192,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )

  net_2 = tf.layers.max_pooling2d(net_2, 3, 2, padding='same')
  net_2 = tf.nn.relu(net_2)

  # 3a第一个分支
  net_3a_1= tf.layers.conv2d(
    net_2, 
    kernel_size=1, 
    filters=64,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 3a第二个分支第一层
  net_3a_2_1= tf.layers.conv2d(
    net_2, 
    kernel_size=1, 
    filters=96,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 3a第二个分支第二层
  net_3a_2_2 = tf.layers.conv2d(
    net_3a_2_1, 
    kernel_size=3, 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 3a第三个分支第一层
  net_3a_3_1= tf.layers.conv2d(
    net_2, 
    kernel_size=1, 
    filters=16,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )

  # 3a第三个分支第二层
  # net_3a_3_2 = tf.layers.conv2d(
  #   net_3a_3_1, 
  #   kernel_size=5, 
  #   filters=32,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )

  net_3a_3_2 = tf.layers.conv2d(
    net_3a_3_1, 
    kernel_size=3, 
    filters=32,
    strides=1,
    activation=None, 
    padding="same"
  )

  net_3a_3_2 = tf.layers.conv2d(
    net_3a_3_2, 
    kernel_size=3, 
    filters=32,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 3a第四个分支第一层
  net_3a_4_1= tf.layers.max_pooling2d(
    net_2, 
    3,
    1,
    padding='same'
  )
  # 3a第四个分支第二层
  net_3a_4_2 = tf.layers.conv2d(
    net_3a_4_1,
    kernel_size=1, 
    filters=32,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 3a concat
  net_3a = tf.concat([net_3a_1, net_3a_2_2, net_3a_3_2, net_3a_4_2], 3)

  # 3b 第一分支
  net_3b_1 = tf.layers.conv2d(
    net_3a, 
    kernel_size=1, 
    filters=128,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )

  # 3b 第二分支第一层
  net_3b_2_1 = tf.layers.conv2d(
    net_3a, 
    kernel_size=1, 
    filters=128,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )

  # 3b 第二分支第二层
  net_3b_2_2 = tf.layers.conv2d(
    net_3b_2_1, 
    kernel_size=3, 
    filters=192,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 3b 第三分支第一层
  net_3b_3_1 = tf.layers.conv2d(
    net_3a, 
    kernel_size=1, 
    filters=32,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )

  # net_3b_3_2 = tf.layers.conv2d(
  #   net_3b_3_1, 
  #   kernel_size=5, 
  #   filters=96,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_3b_3_2 = tf.layers.conv2d(
    net_3b_3_1, 
    kernel_size=3, 
    filters=96,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_3b_3_2 = tf.layers.conv2d(
    net_3b_3_2, 
    kernel_size=3, 
    filters=96,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 3b第四分支第一层
  net_3b_4_1= tf.layers.max_pooling2d(
    net_3a, 
    3,
    1,
    padding='same'
  )
  # 3b第四个分支第二层
  net_3b_4_2 = tf.layers.conv2d(
    net_3b_4_1,
    kernel_size=1, 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 3b concat
  net_3b = tf.concat([net_3b_1, net_3b_2_2, net_3b_3_2, net_3b_4_2], 3)
  
  # max_pool
  max_pool_3b = tf.layers.max_pooling2d(net_3b, 3, 2, padding='same')
  max_pool_3b = tf.nn.relu(max_pool_3b)

  # 4a
  # 4a 第一分支
  net_4a_2_1 = tf.layers.conv2d(
    max_pool_3b, 
    kernel_size=1, 
    filters=192,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )

  # 4a 第二个分支第一层
  net_4a_2_2_1= tf.layers.conv2d(
    max_pool_3b, 
    kernel_size=1, 
    filters=96,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4a 第二个分支第二层
  # net_4a_2_2_2 = tf.layers.conv2d(
  #   net_4a_2_2_1, 
  #   kernel_size=3, 
  #   filters=208,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4a_2_2_2 = tf.layers.conv2d(
    net_4a_2_2_1, 
    kernel_size=(1, 3), 
    filters=208,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4a_2_2_2 = tf.layers.conv2d(
    net_4a_2_2_2, 
    kernel_size=(3, 1), 
    filters=208,
    strides=1,
    activation=None, 
    padding="same"
  )
  
  # 4a 第三个分支第一层
  net_4a_2_3_1= tf.layers.conv2d(
    max_pool_3b, 
    kernel_size=1, 
    filters=16,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )

  # 4a 第三个分支第二层
  # net_4a_2_3_2 = tf.layers.conv2d(
  #   net_4a_2_3_1, 
  #   kernel_size=5, 
  #   filters=48,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  # net_4a_2_3_2 = tf.layers.conv2d(
  #   net_4a_2_3_1, 
  #   kernel_size=3, 
  #   filters=48,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4a_2_3_2 = tf.layers.conv2d(
    net_4a_2_3_1, 
    kernel_size=(1, 3), 
    filters=48,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4a_2_3_2 = tf.layers.conv2d(
    net_4a_2_3_2, 
    kernel_size=(3, 1), 
    filters=48,
    strides=1,
    activation=None, 
    padding="same"
  )
  
  # net_4a_2_3_2 = tf.layers.conv2d(
  #   net_4a_2_3_2, 
  #   kernel_size=3, 
  #   filters=48,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4a_2_3_2 = tf.layers.conv2d(
    net_4a_2_3_2, 
    kernel_size=(1, 3), 
    filters=48,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4a_2_3_2 = tf.layers.conv2d(
    net_4a_2_3_2, 
    kernel_size=(3, 1), 
    filters=48,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 4a 第四个分支第一层
  net_4a_2_4_1= tf.layers.max_pooling2d(
    max_pool_3b, 
    3,
    1,
    padding='same'
  )

  # 4a 第四个分支第二层
  net_4a_2_4_2 = tf.layers.conv2d(
    net_4a_2_4_1,
    kernel_size=1, 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 4a concat
  net_4a = tf.concat([net_4a_2_1, net_4a_2_2_2, net_4a_2_3_2, net_4a_2_4_2], 3)


  # 4b 第一个分支
  net_4b_1 = tf.layers.conv2d(
    net_4a, 
    kernel_size=1, 
    filters=160,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4b 第二个分支第一层
  net_4b_2_1 = tf.layers.conv2d(
    net_4a, 
    kernel_size=1, 
    filters=112,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4b 第二个分支第二层
  # net_4b_2_2 = tf.layers.conv2d(
  #   net_4b_2_1, 
  #   kernel_size=3, 
  #   filters=224,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4b_2_2 = tf.layers.conv2d(
    net_4b_2_1, 
    kernel_size=(1, 3), 
    filters=224,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4b_2_2 = tf.layers.conv2d(
    net_4b_2_2, 
    kernel_size=(3, 1), 
    filters=224,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 4b 第三个分支第一层
  net_4b_3_1 = tf.layers.conv2d(
    net_4a, 
    kernel_size=1, 
    filters=24,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4b 第三个分支第二层
  # net_4b_3_2 = tf.layers.conv2d(
  #   net_4b_3_1, 
  #   kernel_size=5, 
  #   filters=64,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  # net_4b_3_2 = tf.layers.conv2d(
  #   net_4b_3_1, 
  #   kernel_size=3, 
  #   filters=64,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  # net_4b_3_2 = tf.layers.conv2d(
  #   net_4b_3_2, 
  #   kernel_size=3, 
  #   filters=64,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4b_3_2 = tf.layers.conv2d(
    net_4b_3_1, 
    kernel_size=(1, 3), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4b_3_2 = tf.layers.conv2d(
    net_4b_3_2, 
    kernel_size=(3, 1), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4b_3_2 = tf.layers.conv2d(
    net_4b_3_2, 
    kernel_size=(1, 3), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4b_3_2 = tf.layers.conv2d(
    net_4b_3_2, 
    kernel_size=(3, 1), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 4b 第四分支第一层
  net_4b_4_1= tf.layers.max_pooling2d(
    net_4a, 
    3,
    1,
    padding='same'
  )
  # 4b 第四个分支第二层
  net_4b_4_2 = tf.layers.conv2d(
    net_4b_4_1,
    kernel_size=1, 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 4b 第五分支第一层
  net_4b_5_1= tf.layers.average_pooling2d(
    net_4a, 
    5,
    1,
    padding='valid'
  )
  # 4b 第五个分支第二层
  net_4b_5_2 = tf.layers.conv2d(
    net_4b_5_1,
    kernel_size=1, 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 辅助分类器
  # pass

  # 4b concat
  net_4b = tf.concat([net_4b_1, net_4b_2_2, net_4b_3_2, net_4b_4_2], 3)
  print('net_4b', net_4b.shape)

  # 4c 第一个分支
  net_4c_1 = tf.layers.conv2d(
    net_4b, 
    kernel_size=1, 
    filters=128,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4c 第二个分支第一层
  net_4c_2_1 = tf.layers.conv2d(
    net_4b, 
    kernel_size=1, 
    filters=128,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4c 第二个分支第二层
  # net_4c_2_2 = tf.layers.conv2d(
  #   net_4c_2_1, 
  #   kernel_size=3, 
  #   filters=256,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4c_2_2 = tf.layers.conv2d(
    net_4c_2_1, 
    kernel_size=(1, 3), 
    filters=256,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4c_2_2 = tf.layers.conv2d(
    net_4c_2_2, 
    kernel_size=(3, 1), 
    filters=256,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 4c 第三个分支第一层
  net_4c_3_1 = tf.layers.conv2d(
    net_4b, 
    kernel_size=1, 
    filters=24,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4c 第三个分支第二层
  # net_4c_3_2 = tf.layers.conv2d(
  #   net_4c_3_1, 
  #   kernel_size=5, 
  #   filters=64,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  # net_4c_3_2 = tf.layers.conv2d(
  #   net_4c_3_1, 
  #   kernel_size=3, 
  #   filters=64,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4c_3_2 = tf.layers.conv2d(
    net_4c_3_1, 
    kernel_size=(1, 3), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4c_3_2 = tf.layers.conv2d(
    net_4c_3_2, 
    kernel_size=(3, 1), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  # net_4c_3_2 = tf.layers.conv2d(
  #   net_4c_3_2, 
  #   kernel_size=3, 
  #   filters=64,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4c_3_2 = tf.layers.conv2d(
    net_4c_3_2, 
    kernel_size=(1, 3), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4c_3_2 = tf.layers.conv2d(
    net_4c_3_2, 
    kernel_size=(3, 1), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 4c 第四分支第一层
  net_4c_4_1= tf.layers.max_pooling2d(
    net_4b, 
    3,
    1,
    padding='same'
  )
  # 4c 第四个分支第二层
  net_4c_4_2 = tf.layers.conv2d(
    net_4c_4_1,
    kernel_size=1, 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 4c concat
  net_4c = tf.concat([net_4c_1, net_4c_2_2, net_4c_3_2, net_4c_4_2], 3)
  print('net_4c', net_4c.shape)
  # 4d 第一分支
  net_4d_1 = tf.layers.conv2d(
    net_4c, 
    kernel_size=1, 
    filters=112,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4d 第二个分支第一层
  net_4d_2_1 = tf.layers.conv2d(
    net_4c, 
    kernel_size=1, 
    filters=144,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4d 第二个分支第二层
  # net_4d_2_2 = tf.layers.conv2d(
  #   net_4d_2_1, 
  #   kernel_size=3, 
  #   filters=288,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4d_2_2 = tf.layers.conv2d(
    net_4d_2_1, 
    kernel_size=(1, 3), 
    filters=288,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4d_2_2 = tf.layers.conv2d(
    net_4d_2_2, 
    kernel_size=(3, 1), 
    filters=288,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 4d 第三个分支第一层
  net_4d_3_1 = tf.layers.conv2d(
    net_4c, 
    kernel_size=1, 
    filters=32,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4d 第三个分支第二层
  # net_4d_3_2 = tf.layers.conv2d(
  #   net_4d_3_1, 
  #   kernel_size=5, 
  #   filters=64,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  # net_4d_3_2 = tf.layers.conv2d(
  #   net_4d_3_1, 
  #   kernel_size=3, 
  #   filters=64,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4d_3_2 = tf.layers.conv2d(
    net_4d_3_1, 
    kernel_size=(1, 3), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4d_3_2 = tf.layers.conv2d(
    net_4d_3_1, 
    kernel_size=(3, 1), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  # net_4d_3_2 = tf.layers.conv2d(
  #   net_4d_3_2, 
  #   kernel_size=3, 
  #   filters=64,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4d_3_2 = tf.layers.conv2d(
    net_4d_3_2, 
    kernel_size=(1, 3), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4d_3_2 = tf.layers.conv2d(
    net_4d_3_2, 
    kernel_size=(3, 1), 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 4d 第四分支第一层
  net_4d_4_1= tf.layers.max_pooling2d(
    net_4c, 
    3,
    1,
    padding='same'
  )
  # 4d 第四个分支第二层
  net_4d_4_2 = tf.layers.conv2d(
    net_4d_4_1,
    kernel_size=1, 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 4d concat
  net_4d = tf.concat([net_4d_1, net_4d_2_2, net_4d_3_2, net_4d_4_2], 3)

  # 4e 
  # 4e 第一分支
  net_4e_1 = tf.layers.conv2d(
    net_4d, 
    kernel_size=1, 
    filters=256,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4e 第二个分支第一层
  net_4e_2_1 = tf.layers.conv2d(
    net_4d, 
    kernel_size=1, 
    filters=160,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4e 第二个分支第二层
  # net_4e_2_2 = tf.layers.conv2d(
  #   net_4e_2_1, 
  #   kernel_size=3, 
  #   filters=320,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4e_2_2 = tf.layers.conv2d(
    net_4e_2_1, 
    kernel_size=(1, 3), 
    filters=320,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4e_2_2 = tf.layers.conv2d(
    net_4e_2_2, 
    kernel_size=(3, 1), 
    filters=320,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 4e 第三个分支第一层
  net_4e_3_1 = tf.layers.conv2d(
    net_4d, 
    kernel_size=1, 
    filters=32,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 4e 第三个分支第二层
  # net_4e_3_2 = tf.layers.conv2d(
  #   net_4e_3_1, 
  #   kernel_size=5, 
  #   filters=128,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  # net_4e_3_2 = tf.layers.conv2d(
  #   net_4e_3_1, 
  #   kernel_size=3, 
  #   filters=128,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4e_3_2 = tf.layers.conv2d(
    net_4e_3_1, 
    kernel_size=(1, 3), 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4e_3_2 = tf.layers.conv2d(
    net_4e_3_2, 
    kernel_size=(3, 1), 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )
  # net_4e_3_2 = tf.layers.conv2d(
  #   net_4e_3_2, 
  #   kernel_size=3, 
  #   filters=128,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_4e_3_2 = tf.layers.conv2d(
    net_4e_3_2, 
    kernel_size=(1, 3), 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_4e_3_2 = tf.layers.conv2d(
    net_4e_3_2, 
    kernel_size=(3, 1), 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 4e 第四分支第一层
  net_4e_4_1= tf.layers.max_pooling2d(
    net_4d, 
    3,
    1,
    padding='same'
  )
  # 4e 第四个分支第二层
  net_4e_4_2 = tf.layers.conv2d(
    net_4e_4_1,
    kernel_size=1, 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 4e 第五分支第一层
  net_4e_5_1= tf.layers.average_pooling2d(
    net_4d, 
    5,
    1,
    padding='valid'
  )
  # 4e 第五个分支第二层
  net_4e_5_2 = tf.layers.conv2d(
    net_4e_5_1,
    kernel_size=1, 
    filters=64,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 辅助分类器
  # pass

  # 4e concat
  net_4e = tf.concat([net_4e_1, net_4e_2_2, net_4e_3_2, net_4e_4_2], 3)

  # max_pool
  max_pool_4e = tf.layers.max_pooling2d(net_4e, 3, 2, padding='same')
  max_pool_4e = tf.nn.relu(max_pool_4e)

  # 5a
  # 5a 第一分支
  net_5a_1 = tf.layers.conv2d(
    max_pool_4e, 
    kernel_size=1, 
    filters=256,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 5a 第二个分支第一层
  net_5a_2_1 = tf.layers.conv2d(
    max_pool_4e, 
    kernel_size=1, 
    filters=160,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 5a 第二个分支第二层
  net_5a_2_2 = tf.layers.conv2d(
    net_5a_2_1, 
    kernel_size=3, 
    filters=320,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 5a 第三个分支第一层
  net_5a_3_1 = tf.layers.conv2d(
    max_pool_4e, 
    kernel_size=1, 
    filters=32,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 5a 第三个分支第二层
  # net_5a_3_2 = tf.layers.conv2d(
  #   net_5a_3_1, 
  #   kernel_size=5, 
  #   filters=128,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_5a_3_2 = tf.layers.conv2d(
    net_5a_3_1, 
    kernel_size=3, 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_5a_3_2 = tf.layers.conv2d(
    net_5a_3_2, 
    kernel_size=3, 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 5a 第四分支第一层
  net_5a_4_1= tf.layers.max_pooling2d(
    max_pool_4e, 
    3,
    1,
    padding='same'
  )
  # 5a 第四个分支第二层
  net_5a_4_2 = tf.layers.conv2d(
    net_5a_4_1,
    kernel_size=1, 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )

  net_5a = tf.concat([net_5a_1, net_5a_2_2, net_5a_3_2, net_5a_4_2], 3)


  # 5b 第一分支
  net_5b_1 = tf.layers.conv2d(
    net_5a, 
    kernel_size=1, 
    filters=384,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 5b 第二个分支第一层
  net_5b_2_1 = tf.layers.conv2d(
    net_5a, 
    kernel_size=1, 
    filters=192,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 5b 第二个分支第二层
  net_5b_2_2 = tf.layers.conv2d(
    net_5b_2_1, 
    kernel_size=3, 
    filters=384,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 5b 第三个分支第一层
  net_5b_3_1 = tf.layers.conv2d(
    net_5a, 
    kernel_size=1, 
    filters=48,
    strides=1,
    activation=tf.nn.relu, 
    padding="same"
  )
  # 5b 第三个分支第二层
  # net_5b_3_2 = tf.layers.conv2d(
  #   net_5b_3_1, 
  #   kernel_size=5, 
  #   filters=128,
  #   strides=1,
  #   activation=None, 
  #   padding="same"
  # )
  net_5b_3_2 = tf.layers.conv2d(
    net_5b_3_1, 
    kernel_size=3, 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )
  net_5b_3_2 = tf.layers.conv2d(
    net_5b_3_2, 
    kernel_size=3, 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )

  # 5b 第四分支第一层
  net_5b_4_1= tf.layers.max_pooling2d(
    net_5a, 
    3,
    1,
    padding='same'
  )
  # 5b 第四个分支第二层
  net_5b_4_2 = tf.layers.conv2d(
    net_5b_4_1,
    kernel_size=1, 
    filters=128,
    strides=1,
    activation=None, 
    padding="same"
  )
  # 5b concat
  net_5b = tf.concat([net_5b_1, net_5b_2_2, net_5b_3_2, net_5b_4_2], 3)

  print('net_5b', net_5b.shape)

  # avg pool
  avg_pool_5b = tf.layers.average_pooling2d(net_5b, 7, 1, padding='valid')

  net = tf.layers.dropout(avg_pool_5b, 0.4)
  net = tf.layers.dense(net, 1000, activation=tf.nn.relu)
  return net


net = Model(image)