import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist
import time

# read information of image
def read_image(file):
    image = cv2.imread(file)
    # the information of color
    imageC = image.reshape((10000, 3))
    for i in range(100):
        for j in range(100):
            imageC[i*100+j] = image[i][j]
    # use the location(index) as the spatial information
    imageS = np.zeros((10000, 2))
    for i in range(100):
        for j in range(100):
            imageS[i*100+j][0] = i
            imageS[i*100+j][1] = j
    return imageC, imageS

def random_center(k, kernel):
    # choose k centers randomly
    center_kernel = np.zeros((k,10000))
    center_idx = np.random.randint(0,10000,size=k)
    # check the centers are non-repetition
    while 1:
        flag = 0
        for i in range(k):
            for j in range(i+1,k):
                if(center_idx[i]==center_idx[j]):
                    flag = 1
                    break
            if(flag==1):
                break
        if(flag==1):
            center_idx = np.random.randint(0,10000,size=k)
        else:
            break
    # save the center kernel to decide if it converges or not
    for i in range(k):
        center_kernel[i] = kernel[center_idx[i]]
    return center_kernel, center_idx

def k_means_plus_plus(k, kernel):
    center_kernel = np.zeros((k,10000))
    center_idx = np.zeros((k), dtype=int)
    center_idx_tmp = np.random.randint(10000, size=1)

    # choose the first one center
    center_idx[0] = center_idx_tmp
    center_kernel[0] = kernel[center_idx_tmp]
    D = np.zeros((10000))
    P = np.zeros((10000))
    sum_D = np.zeros((10000))
    for i in range(1,k,1):
        # calculate the min distance between this point to center points
        for j in range(10000):
            tmp = np.zeros(i)
            for k in range(i):
                tmp[k] = np.linalg.norm(kernel[j]-center_kernel[k])
            D[j] = pow(min(tmp), 2)
        # calculate the probability of choosing each point as center
        P = D / sum(D)
        # accumulate the probability
        sum_D[0] = P[0]
        for j in range(1,10000):
            sum_D[j] = sum_D[j-1] + P[j]
        # choose a number randomly in [0,1) and find this number in which interval
        # decide the new center by this number in which interval
        tmp = np.random.random()
        tmp_idx = 0
        for j in range(10000):
            if j == 0:
                if((tmp>=0)&(tmp<sum_D[j])):
                    tmp_idx = j
                    break
            else:
                if((tmp>=sum_D[j-1])&(tmp<sum_D[j])):
                    tmp_idx = j
                    break
        center_kernel[i] = kernel[tmp_idx]
        center_idx[i] = tmp_idx
    return center_kernel, center_idx

def calculate_kernel(gammaC, imageC, gammaS, imageS):
    # k(x, x') = exp(-gamma_s * ||S(x)-S(x')||^2 * exp(-gamma_c * ||C(x)-C(x')||^2
    L2normC = cdist(imageC, imageC, 'sqeuclidean')
    L2normS = cdist(imageS, imageS, 'sqeuclidean')
    k = np.exp(-gammaC * L2normC) * np.exp(-gammaS * L2normS)
    return k

# dis = k(xj, xj) - 2/|Ck| * sigma(alpha_kn * k(xj, xn)) + 1/(|Ck|^2) sigmap sigmaq(alpha_kp * alpha_kq * k(xp, xq))
def thirdterm(clusteridx, kernel, cluster):
    tmp = 0.0
    kernel_tmp = kernel.copy()
    for i in range(len(cluster)):
        if(cluster[i]!=clusteridx):
            kernel_tmp[i,:] = 0
            kernel_tmp[:,i] = 0
    tmp = np.sum(kernel_tmp)
    return tmp

def calculate_dis(dataidx, clusteridx, kernel_dataidx, cluster, pq):
    dist = 0.0
    C = 0
    tmp = 0.0
    kernel_tmp = kernel_dataidx.copy()
    mask = (cluster == clusteridx)
    C = np.sum(mask == 1)
    tmp = np.sum(kernel_tmp[mask])
    dist = kernel_tmp[dataidx] - (2.0/C) * tmp
    dist += (1.0/pow(C,2)) * pq
    return dist

# in the first time, we don't have the cluster result of the last time
# dis = k(xi, xi) - 2 * k(xi, centerj) + k(centerj, centerj)
def first_clustering(k, kernel, center_idx):
    cluster = np.zeros(10000, dtype=int)
    for i in range(10000):
        tmp = np.zeros(k)
        for j in range(k):
            tmp[j] = kernel[i][i] - 2 * kernel[i][center_idx[j]] + kernel[center_idx[j]][center_idx[j]]
        cluster[i] = np.argmin(tmp)
    return cluster

# calculate the distance between the center and the point, find the nearest center and assign the point to this cluster
def clustering(k, kernel, cluster):
    pq = np.zeros((k))
    new_cluster = np.zeros(10000, dtype=int)
    for i in range(k):
        pq[i] = thirdterm(i, kernel, cluster)
    for i in range(10000):
        tmp = np.zeros(k)
        for j in range(k):
            tmp[j] = calculate_dis(i, j, kernel[i], cluster, pq[j])
        new_cluster[i] = np.argmin(tmp)
    cluster = new_cluster
    return cluster

# muk = 1/|Ck| * sigma(alpha_kn * k(xn))
def update_center(k, kernel, cluster):
    num_of_element = np.zeros(k)
    sum_of_cluster = np.zeros((k,10000))
    mean_of_cluster = np.zeros((k,10000))
    for i in range(10000):
        num_of_element[cluster[i]] += 1
        sum_of_cluster[cluster[i]] += kernel[i]
    for i in range(k):
        if num_of_element[i] != 0:
            mean_of_cluster[i] = sum_of_cluster[i] / num_of_element[i]
    return mean_of_cluster

# the different cluster use different color and draw the result of clustering
def generate_picture(cluster):
    color = [[0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0]]
    image = Image.new("RGB", (100,100))
    for i in range(10000):
        image.putpixel((int(i%100), int(i/100)),(color[cluster[i]][0],color[cluster[i]][1],color[cluster[i]][2]))
    return image

if __name__ == '__main__':
    # choose random or kmeans++
    initial = input("Please choose an initialization method: ")
    # choose number of clusters
    k = int(input("Please choose number of cluster k: "))
    # choose which image
    image_num = input("Please choose which image you want to load: ")
    gif = []
    iter = 0
    # set the episilin which is used to decide whether it converges or not
    episilon = 1e-10
    # read image and calculate kernel
    imageC, imageS = read_image('image'+image_num+'.png')
    time_start = time.time()
    kernel = calculate_kernel(1e-5, imageC, 1e-5, imageS)
    # initial center
    if initial=='1':
        center_kernel, center_idx = random_center(k, kernel)
    else:
        center_kernel, center_idx = k_means_plus_plus(k, kernel)

    # kmeans algorithm
    while 1:
        iter += 1
        new_center_kernel = center_kernel
        # E-step: cluster all points
        if(iter == 1):
            cluster = first_clustering(k, kernel, center_idx)
        else:
            cluster = clustering(k, kernel, cluster)
        # M-step: calculate the center of each cluster and update centers
        center_kernel = update_center(k, kernel, cluster)
        # draw the picture of the clustering result for this one iteration
        tmp = generate_picture(cluster)
        gif.append(tmp)
        # Decide whether it converges or not
        if((np.linalg.norm(new_center_kernel-center_kernel) <= episilon) or iter >= 100):
            break
    time_end = time.time()
    sum_t=(time_end - time_start)
    print('time cost', sum_t, 's')
    # save gif and the last image(clustering result)
    tmp = "kmeans_image"+image_num+"_method"+initial+"_k"+str(k)+".gif"
    gif[0].save(tmp, save_all=True, append_images=gif[1:], duration=500, loop=0, disposal=0)
    tmp = "kmeans_image"+image_num+"_method"+initial+"_k"+str(k)+".png"
    gif[-1].save(tmp, save_all=False)