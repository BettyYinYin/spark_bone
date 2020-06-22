import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np 


# Inception-Renset-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Renset-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net
  
def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v2(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v2(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV2'):
    """Creates the Inception Resnet V2 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
  
    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 192
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_5a_3x3')
                end_points['MaxPool_5a_3x3'] = net
        
                # 35 x 35 x 320
                with tf.variable_scope('Mixed_5b'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                                    scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                        tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                                    scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                                     scope='AvgPool_0a_3x3')
                        tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                                   scope='Conv2d_0b_1x1')
                    net = tf.concat([tower_conv, tower_conv1_1,
                                        tower_conv2_2, tower_pool_1], 3)
        
                end_points['Mixed_5b'] = net
                net = slim.repeat(net, 10, block35, scale=0.17)
        
                # 17 x 17 x 1024
                with tf.variable_scope('Mixed_6a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                                 scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                                    stride=2, padding='VALID',
                                                    scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
        
                end_points['Mixed_6a'] = net
                net = slim.repeat(net, 20, block17, scale=0.10)
        
                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                                   padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                    net = tf.concat([tower_conv_1, tower_conv1_1,
                                        tower_conv2_2, tower_pool], 3)
        
                end_points['Mixed_7a'] = net
        
                net = slim.repeat(net, 9, block8, scale=0.20)
                net = block8(net, activation_fn=None)
        
                net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
                end_points['Conv2d_7b_1x1'] = net
        
                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
          
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')
          
                    end_points['PreLogitsFlatten'] = net
                
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False)
  
    return net, end_points


image = tf.placeholder(tf.float32, [None, 224, 224, 1])
land_mark = tf.placeholder(tf.float32, [None, 22])
disease_class = tf.placeholder(tf.float32, [None, 11, 5])

net, end_points = inception_resnet_v2(image)
print(net.shape)


predict_land_mark = tf.layers.dense(net, 22, activation=tf.nn.relu)
predict_logits = tf.layers.dense(net, 55, activation=tf.nn.relu)

predict_logits = tf.squeeze(predict_logits)
predict_land_mark = tf.squeeze(predict_land_mark)
print('predict_logits', predict_logits.shape)
print('predict_land_mark', predict_land_mark.shape)

predict_logits = tf.reshape(predict_logits, [-1, 11, 5])
predict_logits = tf.nn.sigmoid(predict_logits)

loss_box = tf.sqrt(tf.reduce_mean((predict_land_mark - land_mark)** 2))

loss_class = tf.nn.sigmoid_cross_entropy_with_logits(labels=disease_class, logits=predict_logits)
loss_class = tf.reduce_mean(
    tf.reduce_sum(loss_class, axis=1)
)

one_hot_prediction = tf.cast(tf.greater(predict_logits, 0.5), tf.float32)
print('one_hot_prediction', one_hot_prediction.shape)
accuracy = tf.reduce_mean(tf.cast(tf.equal(disease_class,one_hot_prediction), tf.float32))
#用梯度迭代算法
loss = loss_box + loss_class
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

# #定义会话
sess = tf.Session()
#初始化所有变量
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint("model"))

train_files = np.load(r"2020-06-14\lumbar_train150\data.npz") 
train_file_image = train_files["data"] 
train_file_label = train_files["land_marks"]
train_file_class = train_files['disease_class']

test_files = np.load(r"2020-06-14\lumbar_train51\data.npz") 
test_file_image = test_files["data"] 
test_file_label = test_files["land_marks"]
test_file_class = test_files['disease_class']

train_file_image = train_file_image.astype('float32')
train_file_image -= np.mean(train_file_image, axis=1, keepdims=True)
train_file_image /= (np.max(np.abs(train_file_image), axis=1, keepdims=True) + 1e-6)

test_file_image = test_file_image.astype('float32')
test_file_image -= np.mean(test_file_image, axis=1, keepdims=True)
test_file_image /= (np.max(np.abs(test_file_image), axis=1, keepdims=True) + 1e-6)

train_file_image = np.reshape(train_file_image, [-1, 224, 224, 1])
test_file_image = np.reshape(test_file_image, [-1, 224, 224, 1])

for itr in range(2000):
    idx = np.random.randint(0, len(train_file_label), [5])
    batch_xs, batch_ys, batch_zs = train_file_image[idx], train_file_label[idx], train_file_class[idx] 
    sess.run(train_step, 
    feed_dict={image: batch_xs, land_mark: batch_ys, disease_class:batch_zs})
    if itr % 10 == 0:
      print('itr', itr)
      accu_list = []
      loss_list = []
      for num in range(15):
        num_acc, num_loss = sess.run([accuracy, loss_box], feed_dict={image: train_file_image[num * 10:(num + 1) * 10],
                          land_mark: train_file_label[num * 10:(num + 1) * 10], disease_class: train_file_class[num * 10:(num + 1) * 10]})
        accu_list.append(num_acc)
        loss_list.append(num_loss)
      # train_acc, train_loss = sess.run([accuracy, loss], feed_dict={image: batch_xs,
      #                     land_mark: batch_ys, disease_class: batch_zs})
      print('training accuracy %f, loss: %f' % (np.mean(accu_list), np.mean(loss_list)))

      test_accu, test_loss, p_class, p_land_mark = sess.run([accuracy, loss_box, one_hot_prediction, predict_land_mark], feed_dict={image: test_file_image,
                          land_mark: test_file_label, disease_class: test_file_class})               
      print('test accuracy %f, loss: %f' % (test_accu, test_loss))

    if itr == 999 or itr == 1999:
      print('p_class', p_class.shape)
      np.savez('predict-%d.npz' % itr, disease_class=p_class, land_mark=p_land_mark)
      pass
      saver.save(sess, "model/a", global_step=itr)



