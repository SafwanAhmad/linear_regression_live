from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    
    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    bs = [0] * num_iterations
    ms = [0] * num_iterations
    err_path = [0] * num_iterations

    for i in range(num_iterations):
        # Plot line for the current value of b and m
        plt.plot(points[:,0], m * points[:,0] + b)
        bs[i] = b
        ms[i] = m
        err_path[i] = compute_error_for_line_given_points(b,m,points)
        b, m = step_gradient(b, m, array(points), learning_rate)
        
        
    B,M = meshgrid(bs,ms)
    
    errs = [ [0 for i in range(num_iterations)] for j in range(num_iterations)]
    
    for i in range(B.shape[0]):
        for j in range(B.shape[0]):
            errs[i][j] = compute_error_for_line_given_points(B[i][j], M[i][j], points)

    print(B.shape, M.shape, array(errs).shape)
    
    fig = plt.figure()
    fig.suptitle('Mean Squared Error Function')
    ax = fig.gca(projection='3d')
    ax.plot_surface(B,M,errs)
    ax.plot(bs,ms,err_path,'r',label='Path of Gradient Descent')
    ax.legend(loc='best')
    plt.show()
    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 100
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

if __name__ == '__main__':
    run()
