import gzip
import math
import numpy as np

if __name__ == '__main__':
    #load train image
    with gzip.open("train-images-idx3-ubyte.gz",'rb') as file:
        magic_num = int.from_bytes(file.read(4),'big')
        num_image = int.from_bytes(file.read(4),'big')
        image_row = int.from_bytes(file.read(4),'big')
        image_col = int.from_bytes(file.read(4),'big')
        pixel = np.zeros((num_image,image_row*image_col),dtype='uint8')
        for i in range(num_image):
            for j in range(image_row*image_col):
                pixel[i][j] = int.from_bytes(file.read(1),'big')
    
    #load train label
    with gzip.open("train-labels-idx1-ubyte.gz",'rb') as file:
        magic_num = int.from_bytes(file.read(4),'big')
        num_label = int.from_bytes(file.read(4),'big')
        label = np.zeros((num_label),dtype='uint8')
        for i in range(num_label):
            label[i] = int.from_bytes(file.read(1),'big')

    #load test image
    with gzip.open("t10k-images-idx3-ubyte.gz",'rb') as file:
        magic_num_test = int.from_bytes(file.read(4),'big')
        num_image_test = int.from_bytes(file.read(4),'big')
        image_row_test = int.from_bytes(file.read(4),'big')
        image_col_test = int.from_bytes(file.read(4),'big')
        pixel_test = np.zeros((num_image_test,image_row_test*image_col_test),dtype='uint8')
        for i in range(num_image_test):
            for j in range(image_row_test*image_col_test):
                pixel_test[i][j] = int.from_bytes(file.read(1),'big')
    
    #load test label
    with gzip.open("t10k-labels-idx1-ubyte.gz",'rb') as file:
        magic_num_test = int.from_bytes(file.read(4),'big')
        num_label_test = int.from_bytes(file.read(4),'big')
        label_test = np.zeros((num_label_test),dtype='uint8')
        for i in range(num_label_test):
            label_test[i] = int.from_bytes(file.read(1),'big')
    
    option = input("toggle option: ")
    
    #discrete
    if option == "0":
        count = np.zeros(10,dtype='float')
        prior = np.zeros((10, image_row*image_col, 32),dtype='float')
        for i in range(num_image):
            classifier = label[i]
            count[classifier] += 1
            #calculate prior
            for j in range(image_row*image_col):
                prior[classifier][j][math.floor(pixel[i][j]/8)] += 1
        for i in range(10):
            for j in range(image_row*image_col):
                for k in range(32):
                    prior[i][j][k] /= count[i]
            #calculate p(class)
            count[i] /= num_image
        
        
        error = 0.0
        for i in range(num_image_test):
            print(i)
            print("Posterior (in log scale):")
            posterior = np.zeros(10,dtype='float')
            #posterior = p(class) * p(pixel1|class) * p(pixel2|class)... = log(p(class)) + log(p(pixel1|class)) + ...
            #p(class) is prior and p(pixel1|class) is likelihood
            for j in range(10):
                posterior[j] += math.log(count[j])
                for k in range(image_row_test*image_col_test):
                    posterior[j] += math.log(max(0.00001,prior[j][k][math.floor(pixel_test[i][k]/8)]))
            #normalize the sum of posterior to 1
            posterior = posterior / sum(posterior)
            for j in range(10):
                print("%d: %.10f" %(j,posterior[j]))
            print("Prediction: %d, Ans: %d" %(np.argmin(posterior),label_test[i]))
            print("")
            #because posterior is negative originally so the sum is , too. Therefore, we need to find argmin.
            if(np.argmin(posterior)!=label_test[i]):
                error += 1
        
        print("Imagination of numbers in Bayesian classifier:")
        print("")

        #the argmax of prior < 16 is zero (128/8=16)
        for i in range(10):
            print("%d:" %(i))
            for j in range(image_row):
                for k in range(image_col):
                    if k == image_col-1:
                        if np.argmax(prior[i][j*image_col+k]) < 16:
                            print("0")
                        else:
                            print("1")
                    else:
                        if np.argmax(prior[i][j*image_col+k]) < 16:
                            print("0 ",end='')
                        else:
                            print("1 ",end='')
            print("")

        print("Error rate: %.4f" %(error/num_image_test))
    
    else:
        count = np.zeros(10,dtype='float')
        mean = np.zeros((10, image_row*image_col),dtype='float')
        x2 = np.zeros((10, image_row*image_col),dtype='float')
        var = np.zeros((10, image_row*image_col),dtype='float')
        #calculate the mean and var of prior for gaussian distribution
        for i in range(num_image):
            classifier = label[i]
            count[classifier] += 1
            for j in range(image_row*image_col):
                mean[classifier][j] += pixel[i][j]
                x2[classifier][j] += pow(pixel[i][j],2)
        for i in range(10):
            for j in range(image_row*image_col):
                mean[i][j] /= count[i]
                x2[i][j] /= count[i]
                var[i][j] = x2[i][j] - pow(mean[i][j],2)
                #because all var is too small or is zero, we can't know which looks like the train data. So, we plus 1000.
                var[i][j] += 1000      
            count[i] /= num_image
        
        error = 0.0
        for i in range(num_image_test):
            print(i)
            print("Posterior (in log scale):")
            posterior = np.zeros(10,dtype='float')
            #posterior = p(class) * p(pixel1|class) * p(pixel2|class)... = log(p(class)) + log(p(pixel1|class)) + ...
            for j in range(10):
                posterior[j] += math.log(count[j])
                for k in range(image_row_test*image_col_test):
                    posterior[j] += -0.5 * math.log(var[j][k] * 2 * math.pi) -0.5 * pow((pixel_test[i][k] - mean[j][k]), 2) / var[j][k]
            #normalize the sum of posterior to 1
            posterior = posterior / sum(posterior)
            for j in range(10):
                print("%d: %.10f" %(j,posterior[j]))
            #because posterior is negative originally so the sum is , too. Therefore, we need to find argmin.
            print("Prediction: %d, Ans: %d" %(np.argmin(posterior),label_test[i]))
            print("")
            if(np.argmin(posterior)!=label_test[i]):
                error += 1

        #mean < 128 is zero.
        print("Imagination of numbers in Bayesian classifier:")
        print("")
        for i in range(10):
            print("%d:" %(i))
            for j in range(image_row):
                for k in range(image_col):
                    if k == image_col-1:
                        if mean[i][j*image_col+k] < 128:
                            print("0")
                        else:
                            print("1")
                    else:
                        if mean[i][j*image_col+k] < 128:
                            print("0 ",end='')
                        else:
                            print("1 ",end='')
            print("")

        print("Error rate: %.4f" %(error/num_image_test))


                