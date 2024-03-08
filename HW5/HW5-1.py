import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data(file):
    #load data from file input.data
    f = open(file, mode='r')
    x = np.zeros((34))
    y = np.zeros((34))
    for i in range(34):
        tmp = f.readline()
        x[i], y[i] = tmp.split()
    f.close()
    return x, y

def rational_quadratic_kernel(x1, x2, sigma, alpha, l):
    #calculate the rational quadratic kernel by sigma^2 * (1 + (x1-x2)^2 / (2 * alpha * l^2)) ^ (-alpha)
    kernel = np.zeros((len(x1),len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            kernel[i][j] = pow(sigma, 2) * pow((1 + pow(x1[i]-x2[j], 2) / (2 * alpha * pow(l,2))), -alpha)
    return kernel

def gaussian_process(x1, x2, sigma, alpha, l, beta):
    #calculate covariance C by C = kernel(xn, xm) + 1/beta * tmp
    #tmp[i][j]=1 if i==j
    tmp = np.zeros((len(x1),len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            if i==j:
                tmp[i][j]=1
    C = rational_quadratic_kernel(x1, x2, sigma, alpha, l) + 1/beta * tmp
    return C

def data_for_plot(x, y, x_plot, sigma, alpha, l, beta):
    #mu = k(x, x*).T * C^(-1) * y
    #var = (k(x*, x*) + 1/beta) - k(x, x*).T * C^(-1) * k(x, x*)
    kernel = rational_quadratic_kernel(x, x_plot, sigma, alpha, l)
    C = gaussian_process(x, x, sigma, alpha, l, beta)
    mu = kernel.T.dot(np.linalg.inv(C)).dot(y)
    k_star = rational_quadratic_kernel(x_plot, x_plot, sigma, alpha, l) + 1/beta
    var = k_star - kernel.T.dot(np.linalg.inv(C)).dot(kernel)
    return mu, var


def plot(x, y, x_plot, mu, var, mu1, var1, origin, opt):
    #draw the picture for sigma = 1 , alpha = 1 and l = 1
    plt.subplot(1,2,1)
    plt.plot(x, y, 'o', color = 'b')
    plt.plot(x_plot, mu, color = 'k') # draw the curve of prediction of f by mu
    confidence = np.zeros((len(var))) 
    for i in range(len(var)):
        confidence[i] = 1.96 * pow(var[i][i], 0.5) #calculate the confidence interval 1.96 * (var ^ 0.5)
    plt.fill_between(x_plot, mu+confidence, mu-confidence, color='g', alpha=0.3)
    plt.title('origin\nsigma=%.2f, alpha=%.2f, l=%.2f' %(origin[0],origin[1],origin[2]))

    #draw the picture for using minimize function to find better parameter
    plt.subplot(1,2,2)
    plt.plot(x, y, 'o', color = 'b')
    plt.plot(x_plot, mu1, color = 'k')
    confidence = np.zeros((len(var1)))
    for i in range(len(var)):
        confidence[i] = 1.96 * pow(var1[i][i], 0.5)
    plt.fill_between(x_plot, mu1+confidence, mu1-confidence, color='g', alpha=0.3)
    plt.title('opt\nsigma=%.2f, alpha=%.2f, l=%.2f' %(opt[0],opt[1],opt[2]))
    plt.show()


def marginal_log_likelihood(theta, x, y, beta):
    # -ln(p) = 0.5 * ln(det(C)) + 0.5 * y.T * C^(-1) * y + N/2 * ln(2*pi)
    C = gaussian_process(x, x, theta[0], theta[1], theta[2], beta)
    lnp = 0.5 * np.log(np.linalg.det(C)) + 0.5 * y.T.dot(np.linalg.inv(C)).dot(y) + (len(x) / 2) * np.log(2 * np.pi)
    return lnp

if __name__ == '__main__':
    x , y = load_data('data/input.data')
    
    #part 1
    sigma = 1.0
    l = 1.0
    alpha = 1.0
    beta = 5.0
    origin = [sigma, alpha, l] #save the parameter and use to print in the title of picture

    x_plot = np.zeros((121))
    for i in range(121):
        x_plot[i] = i - 60
    mu, var = data_for_plot(x, y, x_plot, sigma, alpha, l, beta)

    #part 2
    theta = [1.0,1.0,1.0]
    opt_theta = minimize(marginal_log_likelihood, theta, args=(x, y, beta), method='CG')
    mu1, var1 = data_for_plot(x, y, x_plot, opt_theta.x[0], opt_theta.x[1], opt_theta.x[2], beta)
    
    plot(x, y, x_plot, mu, var, mu1, var1, origin, opt_theta.x)




