import tensorflow as tf
import matplotlib.pyplot as plt

# "tf.set_random_seed(seed): 그래프 수준의 난수 시트 설정-세션간 반복이 가능"
tf.set_random_seed(777)

X = [1, 2, 3]
Y = [1, 2, 3]
W = tf.placeholder(tf.float32)

hypothesis = X*W
cost = tf.reduce_mean(tf.square(hypothesis-Y))
sess = tf.Session()

W_history = []
cost_history = []

for i in range(-30, 50) :
    curr_W = i*0.1
    curr_cost = sess.run(cost, feed_dict={W: curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)

# show the cost function
plt.plot(W_history, cost_history)
plt.show()
