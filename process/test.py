import  numpy as np
import tensorflow as tf

learning_rate = tf.placeholder(tf.float32)

# file = np.load('predict_999.npz')
# print(file['land_mark'][:5])
# print(file['disease_class'][:5])

with tf.Session() as sess:
  print(sess.run(learning_rate, feed_dict={learning_rate: 0.001}))


