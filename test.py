import tensorflow as tf
import numpy as np
y = tf.constant([[4, 10, 10], [2, 3, 4]], dtype=tf.float32)
y2 = tf.constant([[2, 7, 6], [2, 3, 5]], dtype=tf.float32)
label = tf.constant([[1, 0, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]], dtype=tf.int32)


with tf.Session() as sess:
  
  # print(sess.run(tf.sigmoid(y)))
  # # print(sess.run(tf.nn.softmax(y)))
  # one_hot_prediction = tf.cast(tf.greater(tf.sigmoid(y), 0.5), tf.int32)
  
  # # acc, accuracy_update = tf.metrics.accuracy(
  # #                       label,
  # #                       one_hot_prediction
  # #                   )
  # equal = tf.reduce_mean(tf.cast(tf.equal(label,one_hot_prediction), tf.float32))
  # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
  # print(sess.run(one_hot_prediction))
  # # print(sess.run(acc))
  # print(sess.run(tf.reduce_mean((y2 - y) ** 2)))
  # print(sess.run(y2 - y))
  # print(sess.run(y2 - y)**2)
  # print(sess.run(equal))
  
  for num in range(15):
    print(num)

  x = [[[1, 0, 1, 1, 0]]]
  prediction = [[1, 0, 1, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]
  accuracy, accuracy_update = tf.metrics.accuracy(
                        prediction,
                        label,
                    )
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  print(sess.run([accuracy, accuracy_update, tf.reduce_mean(tf.cast(tf.equal(prediction,label), tf.float32))]))
  # print(sess.run(accuracy_update))
  # accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,label), tf.float32))
  print('predict-%d.npz' % 999)



  
  

    



