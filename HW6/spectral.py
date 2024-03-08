import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

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
    center_kernel = np.zeros((k,len(kernel[0])))
    center_idx = np.random.randint(10000,size=k)
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
            center_idx = np.random.randint(10000,size=k)
        else:
            break
    # save the center kernel to decide if it converges or not
    for i in range(k):
        center_kernel[i] = kernel[center_idx[i]]
    return center_kernel

def k_means_plus_plus(k, kernel):
    center_kernel = np.zeros((k,len(kernel[0])))
    center_idx = np.random.randint(10000, size=1)

    # choose the first one center
    center_kernel[0] = kernel[center_idx]
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
    return center_kernel

def Laplacian(normalize, kernel):
    # unnormalize: ratio cut
    if(normalize==0):
        W = kernel
        # Degree matrix D = sigmaj Wij
        D = np.diag(np.sum(W, axis=1))
        # L = D - W
        L = D - W
    # normalize: normalized cut
    else:
        W = kernel
        # Degree matrix D = sigmaj Wij
        D = np.diag(np.sum(W, axis=1))
        # L = D - W
        L = D - W
        # Because D is a diagonal matrix, the inverse matrix is 1/(every diagonal element)
        # D^-(1/2) = 1/((every diagonal element)^0.5)
        D_tmp = np.diag(1/np.sqrt(np.diag(D)))
        # L = D^1/2 * L * D^1/2
        L = np.matmul(np.matmul(D_tmp, L), D_tmp)
    return L

def eigen(normalize, L):
    # Compute the eigenvector of L
    eigen_value, eigen_vector = np.linalg.eig(L)
    # Select k smallest non-null eigenvalues and corresponding eigenvectors
    index = eigen_value.argsort()
    # Following the slides, I choose the sorting eigenvalue1 to eigenvaluek because eigenvalue0 is null and their corresponding eigenvectors
    U = eigen_vector[:,index[1:k+1]]
    # In normalized spectral clustering, uij = uij / {sigmak[(uik)^2]}^0.5
    if(normalize==1):
        U = U / (np.sqrt(np.sum(pow(U, 2), axis=1)).reshape(-1,1))
    return U

def calculate_kernel(gammaC, imageC, gammaS, imageS):
    # k(x, x') = exp(-gamma_s * ||S(x)-S(x')||^2 * exp(-gamma_c * ||C(x)-C(x')||^2
    L2normC = cdist(imageC, imageC, 'sqeuclidean')
    L2normS = cdist(imageS, imageS, 'sqeuclidean')
    k = np.exp(-gammaC * L2normC) * np.exp(-gammaS * L2normS)
    return k

# calculate the distance between the center and the point, find the nearest center and assign the point to this cluster
def clustering(k, kernel, center_kernel):
    cluster = np.zeros(10000, dtype=int)
    for i in range(len(kernel)):
        tmp = np.zeros(k)
        for j in range(k):
            # calculate L2 norm to know the distance between the centers and the point
            tmp[j] = np.linalg.norm(kernel[i]-center_kernel[j])
        cluster[i] = np.argmin(tmp)
    return cluster

# muk = 1/|Ck| * sigma(alpha_kn * k(xn))
def update_center(k, kernel, cluster):
    num_of_element = np.zeros(k)
    sum_of_cluster = np.zeros((k,len(kernel[0])), dtype = kernel.dtype)
    mean_of_cluster = np.zeros((k,len(kernel[0])), dtype = kernel.dtype)
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

# plot the coordinates in the eigenspace of graph Laplacian
def plot(U, cluster, k):
    if(k==2):
        x = U[:,0]
        y = U[:,1]
        for i in range(k):
            idx = (cluster == i)
            plt.plot(x[idx], y[idx], 'o')
    elif(k==3):
        x = U[:,0]
        y = U[:,1]
        z = U[:,2]
        fig = plt.figure().gca(projection='3d')
        for i in range(k):
            idx = (cluster == i)
            fig.plot(x[idx], y[idx], z[idx], 'o')
    plt.show()

if __name__ == '__main__':
    # choose random or kmeans++
    initial = input("Please choose an initialization method: ")
    # choose number of clusters
    k = int(input("Please choose number of cluster k: "))
    # choose which image
    image_num = input("Please choose which image you want to load: ")
    # choose to use normalized or unnormalized spectral clustering
    normalize = int(input("Please choose unnormalize(0) or normalize(1): "))
    gif = []
    iter = 0
    # set the episilin which is used to decide whether it converges or not
    episilon = 1e-10
    # read image and calculate kernel
    imageC, imageS = read_image('image'+image_num+'.png')
    kernel = calculate_kernel(1e-3, imageC, 1e-3, imageS)
    # calculate L and U
    L = Laplacian(normalize, kernel)
    U = eigen(normalize, L)
    # initial center
    if initial=='1':
        center_kernel = random_center(k, U)
    else:
        center_kernel = k_means_plus_plus(k, U)
    # kmeans algorithm
    while 1:
        print(iter)
        iter += 1
        new_center_kernel = center_kernel
        # E-step: cluster all points
        cluster = clustering(k, U, center_kernel)
        # M-step: calculate the center of each cluster and update centers
        center_kernel = update_center(k, U, cluster)
        # draw the picture of the clustering result for this one iteration
        tmp = generate_picture(cluster)
        gif.append(tmp)
        # Decide whether it converges or not
        if((np.linalg.norm(new_center_kernel-center_kernel) <= episilon) or iter >= 100):
            break
    # plot the coordinates in the eigenspace of graph Laplacian
    plot(U, cluster, k)
    # save gif and the last image(clustering result)
    tmp = "spectral_image"+image_num+"_method"+initial+"_k"+str(k)+".gif"
    gif[0].save(tmp, save_all=True, append_images=gif[1:], duration=500, loop=0, disposal=0)
    tmp = "spectral_image"+image_num+"_method"+initial+"_k"+str(k)+".png"
    gif[-1].save(tmp, save_all=False)