from cmath import exp
import numpy as np
import matplotlib.pyplot as plt

def random_generator(mean, var):
    #An easy-to-program approximate approach
    random = sum(np.random.uniform(0.0,1.0,12))-6
    random = random * pow(var,0.5) + mean
    return random
#find a line to classify data (ax + by + c)
def design_matrix(x1,x2,y1,y2):
    phi = np.zeros((len(x1)+len(x2),3))
    for i in range(len(x1)+len(x2)):
        if i < len(x1):
            phi[i][0] = 1
            phi[i][1] = x1[i]
            phi[i][2] = y1[i]
        else : 
            phi[i][0] = 1
            phi[i][1] = x2[i-len(x1)]
            phi[i][2] = y2[i-len(x1)]
    return phi

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

def LU(A):
    #find L and U by guass
    L = np.zeros((len(A),len(A[0])))
    for i in range(len(A)):
        L[i][i] = 1
    A_tmp = A.copy()
    for i in range(len(A_tmp)):
        for j in range(i+1,len(A_tmp)):
            L[j][i] = A_tmp[j][i]/A_tmp[i][i]
            for k in range(len(A_tmp[0])):
                A_tmp[j][k] = A_tmp[j][k]-L[j][i]*A_tmp[i][k]
    U = A_tmp

    #use guass to find L inverse
    L_tmp = np.zeros((len(L),2*len(L[0])))
    for i in range(len(L_tmp)):
        for j in range(len(L_tmp[0])):
            if(j<len(L[0])):
               L_tmp[i][j] = L[i][j] #original L
            else:
                #right hand side is I
                if(i==(j-len(L[0]))):
                    L_tmp[i][j] = 1
                else:
                    L_tmp[i][j] = 0
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            tmp = L_tmp[j][i]/L_tmp[i][i]
            for k in range(len(L_tmp[0])):
                L_tmp[j][k] = L_tmp[j][k]-tmp*L_tmp[i][k]
    #get right hand side
    Lin = np.zeros((len(L),len(L[0])))
    for i in range(len(L_tmp)):
        for j in range(len(L[0]),2*len(L[0])):
            Lin[i][j-len(L[0])]=L_tmp[i][j]

    #use guass to find U inverse
    U_tmp = np.zeros((len(U),2*len(U[0])))
    for i in range(len(U_tmp)):
        for j in range(len(U_tmp[0])):
            if(j<len(U[0])):
               U_tmp[i][j] = U[i][j] #original U
            else:
                #right hand side is I
                if(i==(j-len(U[0]))):
                    U_tmp[i][j] = 1
                else:
                    U_tmp[i][j] = 0
    for i in range(len(U)):
        for j in range(i+1,len(U)):
            tmp = U_tmp[len(U)-j-1][len(U)-i-1]/U_tmp[len(U)-i-1][len(U)-i-1]
            for k in range(len(U_tmp[0])):
                U_tmp[len(U)-j-1][k] = U_tmp[len(U)-j-1][k]-tmp*U_tmp[len(U)-i-1][k]
    #get right hand side
    Uin = np.zeros((len(U),len(U[0])))
    for i in range(len(U_tmp)):
        for j in range(len(U[0]),2*len(U[0])):
            Uin[i][j-len(U[0])]=U_tmp[i][j]
    #let diagonal = 1
    for i in range(len(U_tmp)):
        if(U_tmp[i][i]!=1):
            tmp = U_tmp[i][i]
            for j in range(len(Uin[0])):
                Uin[i][j] = Uin[i][j]/tmp
    
    #Ain = U-1 * L-1
    Ain = mult(Uin,Lin)
    return Ain

if __name__ == '__main__':
    N = 50
    #mx1, vx1, my1, vy1, mx2,vx2,my2,vy2 = (1,3,1,3,2,4,2,4)
    mx1, vx1, my1, vy1, mx2,vx2,my2,vy2 = (1,2,1,2,10,2,10,2)
    D1x = np.zeros((N,1))
    D2x = np.zeros((N,1))
    D1y = np.zeros((N,1))
    D2y = np.zeros((N,1))
    count = np.zeros((2,2))
    gx1 = []
    gy1 = []
    gx2 = []
    gy2 = []
    
    #generate data
    for i in range(N):
        D1x[i][0] = random_generator(mx1,vx1)
        D2x[i][0] = random_generator(mx2,vx2)
        D1y[i][0] = random_generator(my1,vy1)
        D2y[i][0] = random_generator(my2,vy2)
    y = np.zeros((2*N,1))
    design_mat = design_matrix(D1x,D2x,D1y,D2y)
    # data set2 is one and data1 is 0
    for i in range(N):
        y[i+N][0] = 1
    w = np.zeros((3,1))
    tmp1 = np.zeros((2*N,1))

    #wn+1 = wn + XT * (y - 1/(1+exp(-Xw)))
    for i in range(int(10000)):
        tmp = mult(design_mat,w)
           
        tmp1 = y-1/(1+np.exp(-1*tmp))

        gradient = mult(transpose(design_mat),tmp1)
        w += gradient
        if(np.linalg.norm(gradient)<1e-2):
            break
    #if 1/(1+exp(-Xw) > 0.5 is data2 else it's data1
    tmp = mult(design_mat,w)
    classify = 1/(1+np.exp(-1*tmp))
    for i in range(N):
        if classify[i] > 0.5:
            count[0][1] += 1
            gx2.append(D1x[i])
            gy2.append(D1y[i])
        else:
            count[0][0] += 1
            gx1.append(D1x[i])
            gy1.append(D1y[i])
    for i in range(N):
        if classify[N+i] > 0.5:
            count[1][1] += 1
            gx2.append(D2x[i])
            gy2.append(D2y[i])
        else:
            count[1][0] += 1
            gx1.append(D2x[i])
            gy1.append(D2y[i])
    
    count1 = np.zeros((2,2))
    last_w1 =  np.zeros((3,1))
    w1 = np.zeros((3,1))
    tmp1 = np.zeros((2*N,1))
    D = np.zeros((2*N,2*N))
    H = np.zeros((3,3))
    nx1 = []
    ny1 = []
    nx2 = []
    ny2 = []
    #wn+1 = wn + H-1 * gradient
    for i in range(int(10000)):
        #graient = XT * (y - 1/(1+exp(-Xw)))
        tmp = mult(design_mat,w1)
        tmp1 = y-1/(1+np.exp(-1*tmp))
        gradient = mult(transpose(design_mat),tmp1)
        
        # H = XT * D * X ; D = diagonal(exp(-Xw) / (1+exp(-Xw))^2)
        tmp = mult(design_mat,w1)
        tmp1 = np.exp(-1*tmp)/pow((1+np.exp(-1*tmp)),2)
        for i in range(2*N):
            D[i][i] = tmp1[i]
        H = mult(transpose(design_mat),D)
        H = mult(H,design_mat)
        if np.linalg.det(H)==0:
            w1 += gradient
        else:
            w1 += mult(LU(H),gradient)
        
        if(np.linalg.norm(last_w1 - w1)<1e-2):
            break
        last_w1 = w1.copy()
    tmp = mult(design_mat,w1)
    classify1 = 1/(1+np.exp(-1*tmp))
    for i in range(N):
        if classify1[i] > 0.5:
            count1[0][1] += 1
            nx2.append(D1x[i])
            ny2.append(D1y[i])
        else:
            count1[0][0] += 1
            nx1.append(D1x[i])
            ny1.append(D1y[i])
    for i in range(N):
        if classify1[N+i] > 0.5:
            count1[1][1] += 1
            nx2.append(D2x[i])
            ny2.append(D2y[i])
        else:
            count1[1][0] += 1
            nx1.append(D2x[i])
            ny1.append(D2y[i])


    print("Gradient descent:")
    print("")
    print("w:")
    for i in range(len(w)):
        print(w[i][0])
    print("")
    print("Confusion Matrix:")
    print("\t\t Predict cluster 1\t Predict cluster 2")
    print("Is cluster 1\t\t   %d\t\t\t   %d" %(count[0][0],count[0][1]))
    print("Is cluster 2\t\t   %d\t\t\t   %d" %(count[1][0],count[1][1]))
    print("")
    print("Sensitivity (Successfully predict cluster 1): %f" %(count[0][0]/(count[0][0]+count[0][1])))
    print("Specificity (Successfully predict cluster 2): %f" %(count[1][1]/(count[1][0]+count[1][1])))


    print("Newton's method:")
    print("")
    print("w:")
    for i in range(len(w1)):
        print(w1[i][0])
    print("")
    print("Confusion Matrix:")
    print("\t\t Predict cluster 1\t Predict cluster 2")
    print("Is cluster 1\t\t   %d\t\t\t   %d" %(count1[0][0],count1[0][1]))
    print("Is cluster 2\t\t   %d\t\t\t   %d" %(count1[1][0],count1[1][1]))
    print("")
    print("Sensitivity (Successfully predict cluster 1): %f" %(count1[0][0]/(count1[0][0]+count1[0][1])))
    print("Specificity (Successfully predict cluster 2): %f" %(count1[1][1]/(count1[1][0]+count1[1][1])))

    plt.subplot(1,3,1)
    plt.title("Ground truth")
    plt.plot(D1x,D1y,"o", color='r')
    plt.plot(D2x,D2y,"o", color='b')
    
    plt.subplot(1,3,2)
    plt.title("Gradient descent")
    plt.plot(gx1,gy1,"o", color='r')
    plt.plot(gx2,gy2,"o", color='b')

    plt.subplot(1,3,3)
    plt.title("Newton's method")
    plt.plot(nx1,ny1,"o", color='r')
    plt.plot(nx2,ny2,"o", color='b')

    plt.show()