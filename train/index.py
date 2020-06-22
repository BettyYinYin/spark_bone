import numpy as np 
import tensorflow as tf

image = tf.placeholder(tf.float32, [None, 224, 224, 1])
land_mark = tf.placeholder(tf.float32, [None, 22])
disease_class = tf.placeholder(tf.float32, [None, 11, 7])

def block(net, filters, repeat=2):
  for i in range(repeat):
    net = tf.layers.conv2d(
      net, 
      kernel_size=3, 
      filters=filters, 
      activation=tf.nn.relu, 
      padding="same"
    )
    print('net.shape', net.shape)
  net = tf.layers.max_pooling2d(
    net, 
    2, 
    2
  )
  return net 
net = block(image, 64, 2)
net = block(net, 128, 2) 
net = block(net, 256, 3) 
net = block(net, 512, 3)
net = block(net, 512, 3)

net = tf.layers.flatten(net)
# net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
# net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
net = tf.layers.dense(net, 1024, activation=tf.nn.relu)

predict_land_mark = tf.layers.dense(net, 22, activation=None)
predict_logits = tf.layers.dense(net, 77, activation=None)
predict_logits = tf.reshape(predict_logits, [-1, 11, 7])

predict_logits = tf.nn.sigmoid(predict_logits)

loss_box = tf.sqrt(tf.reduce_mean((predict_land_mark - land_mark)** 2))

loss_class = tf.nn.sigmoid_cross_entropy_with_logits(labels=disease_class, logits=predict_logits)
loss_class = tf.reduce_mean(
    tf.reduce_sum(loss_class, axis=1)
)
one_hot_prediction = tf.cast(tf.greater(predict_logits, 0.5), tf.float32)
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

# file_image = file_image.astype('float32')
# file_image -= np.mean(file_image, axis=1, keepdims=True)
# file_image /= (np.max(np.abs(file_image), axis=1, keepdims=True)+1e-6)
train_file_image = np.reshape(train_file_image, [-1, 224, 224, 1])
test_file_image = np.reshape(test_file_image, [-1, 224, 224, 1])

for itr in range(1000):
    idx = np.random.randint(0, len(train_file_label), [1])
    batch_xs, batch_ys, batch_zs = train_file_image[idx], train_file_label[idx], train_file_class[idx] 
    sess.run(train_step, 
    feed_dict={image: batch_xs, land_mark: batch_ys, disease_class:batch_zs})
    if itr % 10 == 0:
      print('itr', itr)
      accu_list = []
      # loss_list = []
      # for num in range(15):
      #   num_acc, num_loss = sess.run([accuracy, loss], feed_dict={image: train_file_image[num * 10:(num + 1) * 10],
      #                     land_mark: train_file_label[num * 10:(num + 1) * 10], disease_class: train_file_class[num * 10:(num + 1) * 10]})

      #   accu_list.append(num_acc)
      #   loss_list.append(num_loss)
      train_acc, train_loss = sess.run([accuracy, loss], feed_dict={image: batch_xs,
                          land_mark: batch_ys, disease_class: batch_zs})
      print('training accuracy %f, loss: %f' % (train_acc, train_loss))

      test_accu, test_loss = sess.run([accuracy, loss], feed_dict={image: test_file_image,
                          land_mark: test_file_label, disease_class: test_file_class})               
      print('test accuracy %f, loss: %f' % (test_accu, test_loss))

      if itr == 999:
        pass
      saver.save(sess, "model/a", global_step=itr)