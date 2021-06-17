import numpy as np
import matplotlib.pyplot as plt

"""
b        -b给出初始值
w        -w给出初始值
points   -输入的点
"""
# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # MSE
        totalError += (y - (w * x + b)) ** 2
    # average loss for each point

    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b = 2(wx+b-y)
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        # grad_w = 2(wx+b-y)*x
        w_gradient += (2 / N) * x * ((w_current * x + b_current) - y)
    # update w'
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]









def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    # update for several times
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
        """
        if i%100 == 0:
            for j in range (0,len(points)):
                x_prt = points[j, 0]
                y_prt = x_prt*w+b
                plt.plot(x_prt, y_prt,"r-")
                plt.show()
        """

    return [b, w]

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        plt.scatter(x, y,label = "true_data")  # 画点
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )

    for j in range(0, len(points)):
        x_prt = points[j, 0]
        y_prt = x_prt * w + b
        #plt.scatter(x_prt, y_prt, label = "pred_data")
        plt.plot_date(x_prt, y_prt, color="red", linewidth=5, linestyle="-",  label="pred_data")
        #plt.plot(x_prt, y_prt,'r-',lw = 5)
        #plt.legend()
    plt.show()


if __name__ == '__main__':
    run()