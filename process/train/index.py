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
net = tf.layers.dense(net, 1000, activation=tf.nn.relu)

predict_land_mark = tf.layers.dense(net, 22, activation=None)
predict_logits = tf.layers.dense(net, 77, activation=None)
predict_logits = tf.reshape(predict_logits, [-1, 11, 7])

predict_logits = tf.nn.sigmoid(predict_logits)

loss_box = tf.reduce_mean((predict_land_mark - land_mark)** 2)

loss_class = tf.nn.sigmoid_cross_entropy_with_logits(labels=disease_class, logits=predict_logits)
loss_class = tf.reduce_mean(
    tf.reduce_sum(loss_class, axis=1)
)
one_hot_prediction = tf.cast(tf.greater(predict_logits, 0.5), tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(disease_class,one_hot_prediction), tf.float32))
#用梯度迭代算法
loss = loss_box + loss_class
train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(loss)

# #定义会话
sess = tf.Session()
#初始化所有变量
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint("model"))

files = np.load(r"lumbar_train51\data.npz") 
file_image = files["data"] 
file_label = files["land_marks"]
file_class = files['disease_class']

file_image = file_image.astype('float32')
file_image -= np.mean(file_image, axis=1, keepdims=True)
file_image /= (np.max(np.abs(file_image), axis=1, keepdims=True)+1e-6)
file_image = np.reshape(file_image, [-1, 224, 224, 1])

for itr in range(1000):
    idx = np.random.randint(0, len(file_label), [1])
    batch_xs, batch_ys, batch_zs = file_image[idx], file_label[idx], file_class[idx] 
    sess.run(train_step, 
    feed_dict={image: batch_xs, land_mark: batch_ys, disease_class:batch_zs})

    if itr % 10 == 0:
      los, acc, p_land_mark, p_one_hot = sess.run([loss, accuracy, predict_land_mark, one_hot_prediction], 
      feed_dict={image: batch_xs,
                      land_mark: batch_ys, disease_class:batch_zs})
      print(itr, acc, los)
      if itr == 999:
        print(p_land_mark[:5])
        print(p_one_hot[:5])
      saver.save(sess, "model/a", global_step=itr)