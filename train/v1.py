import numpy as np 
import tensorflow as tf
import cv2
from model import Model

image = tf.placeholder(tf.float32, [None, 224, 224, 1])
land_mark = tf.placeholder(tf.float32, [None, 22])
learning_rate = tf.placeholder(tf.float32)

net = Model(image)

# predict_land_mark = tf.layers.dense(net, 22, activation=None)
predict_land_mark = tf.layers.dense(net, 22, activation=tf.nn.relu)
# predict_logits = tf.layers.dense(net, 55, activation=tf.nn.relu)

# predict_logits = tf.squeeze(predict_logits)
predict_land_mark = tf.squeeze(predict_land_mark)
# print('predict_logits', predict_logits.shape)
print('predict_land_mark', predict_land_mark.shape)

# predict_logits = tf.reshape(predict_logits, [-1, 11, 5])
# predict_logits = tf.nn.sigmoid(predict_logits)

loss_box = tf.sqrt(tf.reduce_mean((predict_land_mark - land_mark)** 2))

# loss_class = tf.nn.sigmoid_cross_entropy_with_logits(labels=disease_class, logits=predict_logits)
# loss_class = tf.reduce_mean(
#     tf.reduce_sum(loss_class, axis=1)
# )

# one_hot_prediction = tf.cast(tf.greater(predict_logits, 0.5), tf.float32)
# print('one_hot_prediction', one_hot_prediction.shape)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(disease_class,one_hot_prediction), tf.float32))
#用梯度迭代算法
# loss = loss_box + loss_class
# train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_box)
# train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_box)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_box)

# #定义会话
sess = tf.Session()
#初始化所有变量
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint("model"))

train_files = np.load(r"2020-06-14\lumbar_train150\one\data.npz") 
train_images = train_files["data"] 
train_land_marks = train_files["land_marks"]

test_files = np.load(r"2020-06-14\lumbar_train51\one\data.npz") 
test_images = test_files["data"] 
test_land_marks = test_files["land_marks"]

data_train = [(img, mark) for img, mark in zip(train_images, train_land_marks)]
np.random.shuffle(data_train)

train_images = np.array([img for img, mark in data_train])
train_land_marks = np.array([mark for img, mark in data_train])


train_images = train_images.astype('float32')
train_images -= np.mean(train_images, axis=1, keepdims=True)
train_images /= (np.max(np.abs(train_images), axis=1, keepdims=True) + 1e-6)

test_images = test_images.astype('float32')
test_images -= np.mean(test_images, axis=1, keepdims=True)
test_images /= (np.max(np.abs(test_images), axis=1, keepdims=True) + 1e-6)

train_images = np.reshape(train_images, [-1, 224, 224, 1])
test_images = np.reshape(test_images, [-1, 224, 224, 1])

for itr in range(2000):
  idx = np.random.randint(0, len(train_land_marks), [20])
  batch_xs, batch_ys = train_images[idx], train_land_marks[idx]
  sess.run(train_step, 
  feed_dict={image: batch_xs, land_mark: batch_ys, learning_rate: 1e-3})
  if itr % 10 == 0:
    print('itr', itr)
    loss_list = []
    for num in range(15):
      num_loss = sess.run([loss_box], feed_dict={image: train_images[num * 10:(num + 1) * 10],
                        land_mark: train_land_marks[num * 10:(num + 1) * 10]})
      loss_list.append(num_loss)
    # train_acc, train_loss = sess.run([accuracy, loss], feed_dict={image: batch_xs,
    #                     land_mark: batch_ys, disease_class: batch_zs})
    print('training loss: %f' % (np.mean(loss_list)))

    test_loss, p_land_mark = sess.run([loss_box, predict_land_mark], feed_dict={image: test_images,
                        land_mark: test_land_marks})               
    print('test loss: %f' % ( test_loss))

  if itr == 999 or itr == 1999:
    np.savez('predict-%d.npz' % itr, land_mark=p_land_mark)
    pass
    saver.save(sess, "model/a", global_step=itr)













