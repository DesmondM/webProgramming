import tensorflow as tf

# Disable eager execution so we can use sessions like TF1
tf.compat.v1.disable_eager_execution()

# Create a simple computational graph
a = tf.constant(5)
b = tf.constant(2)
c = a * b + 2

# Create and run a session
with tf.compat.v1.Session() as sess:
    result = sess.run(c)
    print("Result:", result)
