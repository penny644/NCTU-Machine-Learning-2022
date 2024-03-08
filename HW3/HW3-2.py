import numpy as np

def random_generator(mean, var):
    #An easy-to-program approximate approach
    random = sum(np.random.uniform(0.0,1.0,12))-6
    random = random * pow(var,0.5) + mean
    return random

if __name__ == '__main__':
    m = float(input("please input m: "))
    s = float(input("please input s: "))
    print("Data point source function: N(%f, %f)" %(m,s))
    mean = 0.0
    diff = 0.0
    original_mean = 0.0
    n = 0
    var = 0.0
    original_var = 0.0
    while 1:
        point = random_generator(m,s)
        print("Add data point: %.10f" %(point))
        original_mean = mean
        original_var = var
        mean = (mean * n + point) / (n + 1)
        #use Welford's online algorithm to update var
        diff += (point - mean) * (point - original_mean)
        var = diff / (n + 1)
        print("Mean = %.10f  Variance = %.10f" %(mean,var))
        n += 1
        if (abs(original_mean - mean) < 1e-4) & (abs(original_var - var) < 1e-4):
            break