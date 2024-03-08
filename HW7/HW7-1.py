from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def load_image(dir, num_image):
    count = 0
    subject = 0
    index = 0
    label = np.zeros(num_image*15)
    image = np.zeros((num_image*15, 45045))
    for file in os.listdir(dir):
        # label depends on the class id
        label[index] = subject
        count += 1
        tmp = Image.open(dir + file)
        # load image and make them 1 * 45045
        image[index] = (np.array(tmp).reshape(-1))
        index += 1
        if count == num_image:
            subject += 1
            count = 0
    return image, label

def eigenface(eigrn_vector):
    # print eigenface and the size of image is (231, 195)
    show_num = int(pow(len(eigrn_vector[0]), 0.5))
    for i in range(len(eigrn_vector[0])):
        plt.subplot(show_num, show_num, i+1)
        plt.imshow(eigrn_vector[:,i].reshape(231,195), cmap='gray')
    plt.show()

def reconstruction(origin_image, reconstruction):
    # choose 10 images randomly and show
    image_num = np.random.randint(0, len(origin_image), 10)
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(np.reshape(origin_image[int(image_num[i])], (231,195)), cmap='gray')
        plt.subplot(2, 10, i+11)
        plt.imshow(reconstruction[:,int(image_num[i])].reshape(231,195), cmap='gray')
    plt.show()

def classify(z, train_label, test_image, test_label, eigen_vector, mean, k):
    correct = 0
    # calculate the projection of test image
    test_center = test_image - mean
    test_z = np.matmul(eigen_vector.T, test_center.T)
    # calculate the distance between each projection of test image and projection of train image
    for i in range(len(test_z[0])):
        distance = np.zeros(len(z[0]))
        for j in range(len(z[0])):
            distance[j] = np.linalg.norm(test_z[:,i].reshape(1, -1) - z[:,j].reshape(1, -1))
        
        # find the k neighbors and which class is the most in these k neighbors
        # if k neighbors are from different classes, choose the nearest one
        index = np.argsort(distance)
        dis_sort = train_label[index[0:k]]
        sort_no_repeat, count = np.unique(dis_sort, return_counts = 1)
        predict =  sort_no_repeat[np.argmax(count)]
        if predict == test_label[i]:
            correct += 1
    return correct/len(test_z[0])*100

# simple PCA: no kernel
def simple_PCA(train_image):
    # calculate the mean and covariance matrix of training data
    mean = np.mean(train_image, axis = 0)
    S = np.cov(train_image)
    # calculate the eigen-vector and eigen-value of covariance matrix
    eigen_value, eigen_vector = np.linalg.eig(S)
    # sort the eigenvalue from big to small
    sort_index = np.argsort(-eigen_value)

    # remove the negative eigen-value
    for i in range(len(sort_index)):
        if(eigen_value[sort_index[i]] <= 0):
            sort_index = sort_index[0:i]
            break
    
    # since the size of (train_image - mean).T @ (train_image - mean) is too big
    # so we use cov function to calculate (train_image - mean) @ (train_image - mean).T
    # (train_image - mean).T @ eigen_vector of (train_image - mean) @ (train_image - mean).T 
    # to find the eigen_vector of (train_image - mean).T @ (train_image - mean)
    eigen_vector = np.matmul((train_image - mean).T, eigen_vector[:, sort_index].real)
    return eigen_vector

# kernel PCA: linear or RBF
def kernel_PCA(train_image, which_kernel):
    # calculate the mean and kernel of training data
    mean = np.mean(train_image, axis = 0)

    if(which_kernel == 'linear'):
        S = np.matmul(train_image, train_image.T)
    elif(which_kernel == 'RBF'):
        dis = cdist(train_image, train_image, 'sqeuclidean')
        S = np.exp(-0.1 * dis)
    # one = lall element = 1/n
    indicator = np.ones((len(S),len(S))) / len(S)
    tmp = np.matmul(indicator, S)
    # S = S - one @ S - S @ one + one @ S @ one
    S = S - tmp - np.matmul(S, indicator) + np.matmul(tmp, indicator)

    # calculate the eigen-vector and eigen-value of covariance matrix
    eigen_value, eigen_vector = np.linalg.eig(S)
    # sort the eigenvalue from big to small
    sort_index = np.argsort(-eigen_value)

    # remove the negative eigen-value
    for i in range(len(sort_index)):
        if(eigen_value[sort_index[i]] <= 0):
            sort_index = sort_index[0:i]
            break
    
    # since the size of (train_image - mean).T @ (train_image - mean) is too big
    # so we use cov function to calculate (train_image - mean) @ (train_image - mean).T
    # (train_image - mean).T @ eigen_vector of (train_image - mean) @ (train_image - mean).T 
    # to find the eigen_vector of (train_image - mean).T @ (train_image - mean)
    eigen_vector = np.matmul((train_image - mean).T, eigen_vector[:, sort_index].real)
    return eigen_vector

def LDA(pca, pca_eigen):
    mean = np.mean(pca, axis = 1)
    
    class_mean = np.zeros((len(pca), 15))
    for i in range(15):
        class_mean[:,i] = np.mean(pca[:,i*9:i*9+9], axis = 1)
    
    # within class scatter: Sw = sigma(sigma((x - class_mean)@(x - class_mean).T))
    Sw = np.zeros((len(pca), len(pca)))
    #rint(np.tile(pca[:,i].reshape(-1,1), 9))
    for i in range(15):
        Sw += np.matmul((pca[:,i*9:i*9+9] - np.tile(class_mean[:,i].reshape(-1,1), 9)),(pca[:,i*9:i*9+9] - np.tile(class_mean[:,i].reshape(-1,1), 9)).T)

    # between class scatter: Sb = sigma(n * (class_mean - mean) @ (class_mean - mean).T)
    Sb = np.zeros((len(pca), len(pca)))
    for i in range(15):
        tmp = 9 * (class_mean[:,i]-mean)
        Sb +=  np.matmul(tmp, (class_mean[:,i]-mean).T)

    # S = Sw-1 @ Sb
    S = np.matmul(np.linalg.inv(Sw), Sb)

    eigen_value, eigen_vector = np.linalg.eig(S)

    # sort the eigenvalue from big to small
    sort_index = np.argsort(-eigen_value)
    # remove the negative eigen-value
    for i in range(len(sort_index)):
        if(eigen_value[sort_index[i]] <= 0):
            sort_index = sort_index[0:i]
            break
    
    # the projection eigen_vector is composed og pca_eigen and LDA_eigen
    eigen_vector = np.matmul(pca_eigen, eigen_vector[:, sort_index].real)
    return eigen_vector

# the main function for simple PCA
def simple_PCA_main(train_image, train_label, test_image, test_label):
    # find the eigen-vector of covariance matrix
    eigen_vector = simple_PCA(train_image)
    # show 25 eigenface
    eigenface(eigen_vector[:,0:25])

    # reconstruct the face by eigen-vector
    mean = np.mean(train_image, axis=0)
    center = (train_image - mean)
    # since we use (train_image - x_mean) and eigen-vector to calculate the projection of (train_image - x_mean), z
    z = np.matmul(eigen_vector.T, center.T)
    # so the real value after reconstructing of train image is eigen_vector @ z + mean
    reconstruct = np.matmul(eigen_vector, z) + mean.reshape(-1,1)
    reconstruction(train_image, reconstruct)

    # calculate the accuracy of classifying
    correct = classify(z, train_label, test_image, test_label, eigen_vector, mean, 3)
    print(f"PCA acc: {correct:.1f}%")

    # return z and eigen-vector because LDA needs
    return z, eigen_vector

# the main function for LDA
def LDA_main(pca, pca_eigen, train_image, train_label, test_image, test_label):
    # find the eigen-vector of LDA
    eigen_vector = LDA(pca, pca_eigen)
    # show 25 eigenface
    eigenface(eigen_vector[:,0:25])

     # reconstruct the face by eigen-vector
    mean = np.mean(train_image, axis=0)
    center = (train_image - mean)
    # since we use (train_image - x_mean) and eigen-vector to calculate the projection of (train_image - x_mean), z
    z = np.matmul(eigen_vector.T, center.T)
    # so the real value after reconstructing of train image is eigen_vector @ z + mean
    reconstruct = np.matmul(eigen_vector, z) + mean.reshape(-1,1)
    reconstruction(train_image, reconstruct)

    # calculate the accuracy of classifying
    correct = classify(z, train_label, test_image, test_label, eigen_vector, mean, 3)
    print(f"LDA acc: {correct:.1f}%")

def kernel_PCA_main(train_image, train_label, test_image, test_label, which_kernel):
    # find the eigen-vector of covariance matrix
    eigen_vector = kernel_PCA(train_image, which_kernel)
    # show 25 eigenface
    eigenface(eigen_vector[:,0:25])

    # reconstruct the face by eigen-vector
    mean = np.mean(train_image, axis=0)
    center = (train_image - mean)
    # since we use (train_image - x_mean) and eigen-vector to calculate the projection of (train_image - x_mean), z
    z = np.matmul(eigen_vector.T, center.T)
    # so the real value after reconstructing of train image is eigen_vector @ z + mean
    reconstruct = np.matmul(eigen_vector, z) + mean.reshape(-1,1)
    reconstruction(train_image, reconstruct)

    # calculate the accuracy of classifying
    correct = classify(z, train_label, test_image, test_label, eigen_vector, mean, 3)
    print(f"PCA acc: {correct:.1f}%")

    # return z and eigen-vector because LDA needs
    return z, eigen_vector
if __name__=='__main__':
    # load image data
    train_dir = "./Yale_Face_Database/Training/"
    train_image, train_label = load_image(train_dir, 9)
    test_dir = "./Yale_Face_Database/Testing/"
    test_image, test_label = load_image(test_dir, 2)

    # no kernel PCA
    print("No kernel")
    z, eigen_vector = simple_PCA_main(train_image, train_label, test_image, test_label)
    LDA_main(z, eigen_vector, train_image, train_label, test_image, test_label)

    # linear kernel PCA
    print("Linear kernel")
    z, eigen_vector = kernel_PCA_main(train_image, train_label, test_image, test_label, 'linear')
    LDA_main(z, eigen_vector, train_image, train_label, test_image, test_label)

    # RBF kernel PCA
    print("RBF kernel")
    z, eigen_vector = kernel_PCA_main(train_image, train_label, test_image, test_label, 'RBF')
    LDA_main(z, eigen_vector, train_image, train_label, test_image, test_label)