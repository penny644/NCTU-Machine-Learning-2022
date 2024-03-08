import gzip
import math
import numpy as np
from tqdm import trange
def transpose(A):
    A_tran = np.zeros((len(A[0]),len(A)))
    for i in range(len(A)):
        for j in range(len(A[0])):
            A_tran[j][i] = A[i][j]
    return A_tran

def mult(A, B):
    ans = np.zeros((len(A),len(B[0])))
    for i in range(len(A)):
        for j in range(len(B[0])):
            tmp = 0
            for k in range(len(A[0])):
                tmp += A[i][k] * B[k][j]
            ans[i][j] = tmp
    return ans

if __name__ == '__main__':
    #load train image
    file1 = open("result.txt", 'w')
    with gzip.open("train-images-idx3-ubyte.gz",'rb') as file:
        magic_num = int.from_bytes(file.read(4),'big')
        num_image = int.from_bytes(file.read(4),'big')
        image_row = int.from_bytes(file.read(4),'big')
        image_col = int.from_bytes(file.read(4),'big')
        pixel = np.zeros((num_image,image_row*image_col),dtype='uint8')
        for i in trange(num_image):
            for j in range(image_row*image_col):
                pixel[i][j] = int.from_bytes(file.read(1),'big')
    
    #load train label
    with gzip.open("train-labels-idx1-ubyte.gz",'rb') as file:
        magic_num = int.from_bytes(file.read(4),'big')
        num_label = int.from_bytes(file.read(4),'big')
        label = np.zeros((num_label),dtype='uint8')
        for i in trange(num_label):
            label[i] = int.from_bytes(file.read(1),'big')

    
    #transfer to 2 bins
    img_bin = np.zeros((num_image,image_row*image_col))
    for i in trange(num_image):
        for j in range(image_row*image_col):
            if(pixel[i][j]>127):
                img_bin[i][j] = 1
            else:
                img_bin[i][j] = 0
    p = np.random.rand(10,image_row*image_col)
    
    lam = np.ones((10,1))
    lam = lam / 10
    w = np.zeros((num_image,10))
    old_p = np.zeros((10,image_row*image_col))
    iteration = 0
    while iteration < 15:
        iteration += 1
        # E step: random choose p and calculate w by log
        for i in trange(num_image):
            for j in range(10):
                if(lam[j]<1e-8):
                    lam[j] = 1e-8
                w[i][j] = np.log(lam[j])
                for k in range(image_row*image_col):
                    p_tmp = 1-p[j][k]
                    if p[j][k] < 1e-8:
                        p[j][k] = 1e-8
                    if p_tmp < 1e-8:
                        p_tmp = 1e-8
                    w[i][j] += img_bin[i][k]*(np.log(p[j][k]))
                    w[i][j] += (1-img_bin[i][k])*np.log(p_tmp)
            w[i] = w[i] - np.max(w[i])
            w[i] = np.exp(w[i])
            if(sum(w[i])!=0):
                w[i] /= sum(w[i])
        # M step:use MLE to update p and lambda
        # p = sigma(wx) / sigma(w); lambda = sigma(w) / n
        p = mult(transpose(w),img_bin)
        for i in range(10):
            for j in trange(num_image):
                lam[i] += w[j][i]
            if lam[i]!=0:
                p[i] /= lam[i]      
            lam[i] /= num_image
        for i in range(10):
            for j in range(image_row*image_col):
                if p[i][j] == 0:
                    p[i][j] = 1e-5

        for i in range(10):
            print("class %d:" %(i))
            file1.write("class %d:\n" %(i))
            for j in range(image_row):
                for k in range(image_col):
                    if(p[i][j*image_col+k]>0.5):
                        print("1 ",end='')
                        file1.write("1 ")
                    else:
                        print("0 ",end='')
                        file1.write("0 ")
                print("")
                file1.write("\n")
        file1.write("No. of Iteration: %d, Difference: %f\n" %(iteration,sum(sum(abs(old_p-p)))))
        print("No. of Iteration: %d, Difference: %f" %(iteration,sum(sum(abs(old_p-p)))))
        
        if sum(sum(abs(old_p-p))) < 20:
            break
        old_p = p.copy()
    
    mapping = np.zeros((10,1))
    mapping_class_label = np.zeros((10,1))
    counter = np.zeros((10,10))
    for i in range(num_image):
        counter[label[i]][np.argmax(w[i])] += 1
    for i in range(10):
        tmp = np.argmax(counter)
        mapping_class_label[int(tmp%10)] = int(tmp/10)
        mapping[int(tmp/10)] = int(tmp%10)
        counter[:,tmp%10] = -1.0*math.inf
        counter[int(tmp/10),:] = -1.0*math.inf

    for i in range(10):
        print("labeled %d:" %(i))
        file1.write("labeled %d:\n" %(i))
        for j in range(image_row):
            for k in range(image_col):
                if(p[int(mapping[i][0])][j*image_col+k]>0.5):
                    print("1 ",end='')
                    file1.write("1 ")
                else:
                    print("0 ",end='')
                    file1.write("0 ")
            print("")
            file1.write("\n")
    confusion = np.zeros((10,2,2))
    err = 0
    for i in range(num_image):
        for j in range(10):
            tmp = np.argmax(w[i])
            if label[i] == j:
                if mapping_class_label[tmp][0] == j:
                    confusion[j][0][0] += 1
                else:
                    confusion[j][0][1] += 1
            else:
                if mapping_class_label[tmp][0] == j:
                    confusion[j][1][0] += 1
                else:
                    confusion[j][1][1] += 1
    for i in range(10):
        err += confusion[i][0][1]
    for i in range(10):
        print("Confusion Matrix %d:" %(i))
        print("\t\t Predict number %d \t Predict not number %d" %(i,i))
        print("Is number %d\t\t   %d\t\t\t   %d" %(i,confusion[i][0][0],confusion[i][0][1]))
        print("Isn't number %d \t\t   %d\t\t\t   %d" %(i,confusion[i][1][0],confusion[i][1][1]))
        print("")
        print("Sensitivity (Successfully predict number %d): %f" %(i,confusion[i][0][0]/(confusion[i][0][0]+confusion[i][0][1])))
        print("Specificity (Successfully predict not number %d): %f" %(i,confusion[i][1][1]/(confusion[i][1][0]+confusion[i][1][1])))
        
        file1.write("Confusion Matrix %d:\n" %(i))
        file1.write("\t\t\t\t\t\t\t Predict number %d \t Predict not number %d\n" %(i,i))
        file1.write("Is number %d\t\t\t\t\t   %d\t\t\t\t\t\t\t\t   %d\n" %(i,confusion[i][0][0],confusion[i][0][1]))
        file1.write("Isn't number %d \t\t\t   %d\t\t\t\t\t\t\t\t   %d\n" %(i,confusion[i][1][0],confusion[i][1][1]))
        file1.write("\n")
        file1.write("Sensitivity (Successfully predict number %d): %f\n" %(i,confusion[i][0][0]/(confusion[i][0][0]+confusion[i][0][1])))
        file1.write("Specificity (Successfully predict not number %d): %f\n" %(i,confusion[i][1][1]/(confusion[i][1][0]+confusion[i][1][1])))
    print("Total iteration to converge: %d" %(iteration))
    print("Total error rate: %f" %((err)/num_image))
    file1.write("Total iteration to converge: %d\n" %(iteration))
    file1.write("Total error rate: %f\n" %((err)/num_image))
    file1.close()

    print(mapping)


        
