import tensorflow as tf

# Create some tensors
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

print("Tensor a:\n", a)
print("Tensor b:\n", b)

# 1️⃣ Element-wise addition
add_result = tf.add(a, b)
print("\nAddition:\n", add_result)

# 2️⃣ Element-wise multiplication
mul_result = tf.multiply(a, b)
print("\nMultiplication:\n", mul_result)

# 3️⃣ Matrix multiplication
matmul_result = tf.matmul(a, b)
print("\nMatrix Multiplication:\n", matmul_result)

# 4️⃣ Transpose
transpose_result = tf.transpose(a)
print("\nTranspose of a:\n", transpose_result)

# 5️⃣ Reshape
reshape_result = tf.reshape(a, [4, 1])
print("\nReshaped a to 4x1:\n", reshape_result)

# 6️⃣ Reduction (sum)
sum_result = tf.reduce_sum(a)
print("\nSum of all elements in a:", sum_result.numpy())

# 7️⃣ Broadcasting
c = tf.constant([1, 2], dtype=tf.float32)
broadcast_result = a + c
print("\nBroadcasting add (a + [1, 2]):\n", broadcast_result)
