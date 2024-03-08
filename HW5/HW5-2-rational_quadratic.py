import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from libsvm.svmutil import *

def load_data(file, row, col):
    f = open(file, mode='r')
    data = np.zeros((row,col))
    for i in range(row):
      tmp = f.readline()
      data[i] = np.array(tmp.strip().split(","), dtype='float')
    f.close()
    return data

def svm_part1(x, y, parameter, x_test, y_test):
    #y is label and x is vector for image and set the SVM problem
    prob  = svm_problem(y, x)
    #set svm parameter
    param = svm_parameter(parameter)
    #train a svm model
    model = svm_train(prob, param)
    #predict the label of image in test dataset and return the accuracy
    labels, acc, values = svm_predict(y_test, x_test, model)
    return acc

def gridsearch(kernel, x, y, x_test, y_test):
    # try cost 0.01, 0.1, 1, 10, 100
    cost = [0.01,0.1,1,10,100]
    # By default, gamma = 1/784. So we try 1/784, 0.01, 0.1, 1. 
    # Gamma is only used for poly and RBF kernel
    gamma = [1/784, 0.01, 0.1, 1]
    best = []
    best_cost = 0.0
    best_gamma = 0.0
    best_acc = 0.0
    if(kernel=='linear'):
        print("Svm with linear kernel")
        for i in range(len(cost)):
            parameter = '-q -t 0 -v 4 -c '+str(cost[i])
            print("Cost= %f" %(cost[i]))
            # y is label and x is vector for image and set the SVM problem
            prob  = svm_problem(y, x)
            #set svm parameter
            param = svm_parameter(parameter)
            #train a svm model
            acc = svm_train(prob, param)
            # if accuracy is larger than old acc, save new acc and parameter
            if(acc>best_acc):
                best_acc = acc
                best_cost = cost[i]
        parameter = '-q -t 0 -c '+str(best_cost)
        print("Test with best cost= %f" %(best_cost))
        labels, test_acc, values = svm_part1(x, y, parameter, x_test, y_test)
        best.append(best_acc)
        best.append(best_cost)
    elif(kernel=='poly'):
        print("Svm with polynomial kernel")
        for i in range(len(cost)):
            for j in range(len(gamma)):
                parameter = '-q -t 1 -v 4 -c '+str(cost[i]) + ' -g '+str(gamma[j])
                print("Cost= %f, Gamma= %f" %(cost[i],gamma[j]))
                # y is label and x is vector for image and set the SVM problem
                prob  = svm_problem(y, x)
                #set svm parameter
                param = svm_parameter(parameter)
                #train a svm model
                acc = svm_train(prob, param)
                if(acc>best_acc):
                    best_acc = acc
                    best_cost = cost[i]
                    best_gamma = gamma[j]
        parameter = '-q -t 1 -c '+str(best_cost) + ' -g '+str(best_gamma)
        print("Test with best cost= %f, gamma= %f" %(best_cost,best_gamma))
        labels, test_acc, values = svm_part1(x, y, parameter, x_test, y_test)
        best.append(best_acc)
        best.append(best_cost)
        best.append(best_gamma)
    elif(kernel=='RBF'):
        print("Svm with RBF kernel")
        for i in range(len(cost)):
            for j in range(len(gamma)):
                parameter = '-q -t 2 -v 4 -c '+str(cost[i]) + ' -g '+str(gamma[j])
                print("Cost= %f, Gamma= %f" %(cost[i],gamma[j]))
                # y is label and x is vector for image and set the SVM problem
                prob  = svm_problem(y, x)
                #set svm parameter
                param = svm_parameter(parameter)
                #train a svm model
                acc = svm_train(prob, param)
                if(acc>best_acc):
                    best_acc = acc
                    best_cost = cost[i]
                    best_gamma = gamma[j]
        parameter = '-q -t 2 -c '+str(best_cost) + ' -g '+str(best_gamma)
        print("Test with best cost= %f, gamma= %f" %(best_cost,best_gamma))
        labels, test_acc, values = svm_part1(x, y, parameter, x_test, y_test)
        best.append(best_acc)
        best.append(best_cost)
        best.append(best_gamma)
    return best
            
def user_define_kernel(x1, x2, dataset_row):
    sigma = 1.0
    alpha = 1.0
    l = 1.0
    rational_quadratic_kernel = pow(sigma, 2) * pow((1 + cdist(x1, x2, 'sqeuclidean') / (2 * alpha * pow(l,2))), -alpha)
    kernel = np.hstack((np.arange(1, dataset_row + 1).reshape(-1, 1), rational_quadratic_kernel))
    return kernel


def gridsearch_user(x, y, x_test, y_test):
    cost = [0.01,0.1,1,10,100]
    best = []
    best_cost = 0.0
    best_acc = 0.0
    print("Svm with rational quadratic kernel")
    for i in range(len(cost)):
        kernel_train = user_define_kernel(x, x, 5000)
        # use user-defined kernel to train and predict
        # isKernel=True must be set for precomputed kernel
        prob  = svm_problem(y, kernel_train, isKernel=True)
        # -t 4 means use user defined kernel
        parameter = '-q -t 4 -v 4 -c '+str(cost[i])
        print("Cost= %f" %(cost[i]))
        param = svm_parameter(parameter)
        acc = svm_train(prob, param)
        if(acc>best_acc):
            best_acc = acc
            best_cost = cost[i]
    kernel_train = user_define_kernel(x, x, 5000)
    kernel_test = user_define_kernel(x_test, x, 2500)
    parameter = '-q -t 4 -c '+str(best_cost)
    print("Test with best cost= %f" %(best_cost))
    # isKernel=True must be set for precomputed kernel
    prob  = svm_problem(y, kernel_train, isKernel=True)
    param = svm_parameter(parameter)
    model = svm_train(prob, param)
    labels, test_acc, values = svm_predict(y_test, kernel_test, model)
    best.append(best_acc)
    best.append(best_cost)
    return best

if __name__ == '__main__':
    #load data from csv file
    x_train = load_data("data/X_train.csv", 5000, 784)
    y_train = load_data("data/Y_train.csv", 5000, 1).flatten()
    x_test = load_data("data/X_test.csv", 2500, 784)
    y_test = load_data("data/Y_test.csv", 2500, 1).flatten()

    #input for choose part
    part = input('Please choose part 1, 2 or 3: ')

    if(part=='1'):
        print('Svm with linear kernel: ',end='')
        # -t 0 means linear kernel
        linear_acc = svm_part1(x_train, y_train, '-q -t 0', x_test, y_test)
        print('Svm with polynomial kernel: ',end='')
        # -t 1 means polynomial kernel
        poly_acc = svm_part1(x_train, y_train, '-q -t 1', x_test, y_test)
        print('Svm with RBF kernel: ',end='')
        # -t 2 means RBF kernel
        RBF_acc = svm_part1(x_train, y_train, '-q -t 2', x_test, y_test)
    elif(part=='2'):
        best = gridsearch('linear', x_train, y_train, x_test, y_test)
        print("Svm with linear kernel\nbest cost: %f\nbest cross validation accuracy: %f%%" %(best[1],best[0]))
        best = gridsearch('poly', x_train, y_train, x_test, y_test)
        print("Svm with polynomial kernel\nbest cost: %f\nbest gamma: %f\nbest cross validation accuracy: %f%%" %(best[1], best[2], best[0]))
        best = gridsearch('RBF', x_train, y_train, x_test, y_test)
        print("Svm with RBF kernel\nbest cost: %f\nbest gamma: %f\nbest cross validation accuracy: %f%%" %(best[1], best[2], best[0]))
    elif(part=='3'):
        best = gridsearch_user(x_train, y_train, x_test, y_test)
        print("Svm with linear + RBF kernel\nbest cost: %f\nbest cross validation accuracy: %f%%" %(best[1], best[0]))
