import numpy as np
import matplotlib.pyplot as plt

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
    #read input and parameters
    path = input(("path= "))
    filename = input(("filename= "))
    n = int(input("n = "))
    l = float(input("lambda = "))
    f = open(path+"\\"+filename,"r")
    A = []
    b = []
    for line in f.readlines():
        tmp = line.split(',')
        A.append(float(tmp[0]))
        b.append(float(tmp[1]))
    
    #transfer A and b to nparray
    A_arr = np.zeros((len(A),n))
    b_arr = np.zeros((len(b),1))
    for i in range(len(A)):
        b_arr[i][0] = b[i]
        for j in range(n):
            A_arr[i][j] = pow(A[i],n-1-j)
    
    #calculate ATA
    ATA = mult(transpose(A_arr), A_arr)

    #LSE
    #calculate ATA + lambda*I
    ATA_lambda = ATA.copy()
    for i in range(len(ATA)):
        ATA_lambda[i][i] = ATA_lambda[i][i] + l
    
    #use LU decomposition to calculate the inverse of (ATA + lambda*I)
    ATA_lambdain = LU(ATA_lambda)

    #calculate (ATA + lambda*I)-1 * AT * b
    LSE = mult(ATA_lambdain, transpose(A_arr))
    LSE = mult(LSE, b_arr)

    #calculate the total error
    LSE_error = mult(A_arr, LSE)
    LSE_total_error = 0.0
    for i in range(len(LSE_error)):
        LSE_total_error += pow((LSE_error[i][0] - b_arr[i][0]),2)

    #print the answer
    print("LSE:")
    print("Fitting line: ",end='')
    for i in range(len(LSE)):
        if(LSE[i]<0):
            if(i<len(LSE)-1):
                if(LSE[i+1]<0):
                    print("%.10fX^%d - " %(LSE[i]*(-1),len(LSE)-i-1),end='')
                else:
                    print("%.10fX^%d + " %(LSE[i]*(-1),len(LSE)-i-1),end='')
            else:
                print("%.10f" %(LSE[i]*(-1)))
        else:
            if(i<len(LSE)-1):
                if(LSE[i+1]<0):
                    print("%.10fX^%d - " %(LSE[i],len(LSE)-i-1),end='')
                else:
                    print("%.10fX^%d + " %(LSE[i],len(LSE)-i-1),end='')
            else:
                print("%.10f" %(LSE[i]))
    print("Total error: %.10f" %(LSE_total_error))

    #Newton's method
    #calculate the gradient 2(ATAx - ATb)
    X = np.zeros((len(ATA[0]),1))
    g_tmp1 = mult(ATA, X)
    g_tmp2 = mult(transpose(A_arr), b_arr)
    gradient = np.zeros((len(g_tmp1),len(g_tmp1[0])))
    for i in range(len(g_tmp1)):
        for j in range(len(g_tmp1[0])):
            gradient[i][j] = 2*(g_tmp1[i][j]-g_tmp2[i][j])
    
    #calculate the Hessian 2ATA
    H = np.zeros((len(ATA),len(ATA[0])))
    for i in range(len(ATA)):
        for j in range(len(ATA[0])):
            H[i][j] = ATA[i][j]*2
    
    #calculate Xn+1 = Xn - (H-1) * gradient 
    Hin = LU(H)
    Newton = mult(Hin,gradient)
    for i in range(len(X)):
        Newton[i] = X[i]-Newton[i]

    #calculate the total error
    Newton_error = mult(A_arr, Newton)
    Newton_total_error = 0.0
    for i in range(len(Newton_error)):
        Newton_total_error += pow((Newton_error[i][0] - b_arr[i][0]),2)

    #print the answer
    print("Newton's Method:")
    print("Fitting line: ",end='')
    for i in range(len(Newton)):
        if(Newton[i]<0):
            if(i<len(Newton)-1):
                if(Newton[i+1]<0):
                    print("%.10fX^%d - " %(Newton[i]*(-1),len(Newton)-i-1),end='')
                else:
                    print("%.10fX^%d + " %(Newton[i]*(-1),len(Newton)-i-1),end='')
            else:
                print("%.10f" %(Newton[i]*(-1)))
        else:
            if(i<len(Newton)-1):
                if(Newton[i+1]<0):
                    print("%.10fX^%d - " %(Newton[i],len(Newton)-i-1),end='')
                else:
                    print("%.10fX^%d + " %(Newton[i],len(Newton)-i-1),end='')
            else:
                print("%.10f" %(Newton[i]))
    print("Total error: %.10f" %(Newton_total_error))

    #Visualization
    x_min = int(min(A))-1
    x_max = int(max(A))+1
    #draw the picture for LSE
    plt.subplot(2,1,1)
    LSE_x_curve = np.linspace(x_min, x_max, 10000)
    LSE_y_curve = np.zeros(len(LSE_x_curve))
    LSE_x_curve_tmp = np.zeros((len(LSE_x_curve), n))
    for i in range(len(LSE_x_curve)):
        for j in range(n):
            LSE_x_curve_tmp[i][j] = pow(LSE_x_curve[i], n-1-j) 
    LSE_y_curve = mult(LSE_x_curve_tmp, LSE)
    plt.plot(A, b, 'o', color='r')
    plt.plot(LSE_x_curve, LSE_y_curve, color='k')

    #draw the picture for Newton's method
    plt.subplot(2,1,2)
    Newton_x_curve = np.linspace(x_min, x_max, 10000)
    Newton_y_curve = np.zeros(len(Newton_x_curve))
    Newton_x_curve_tmp = np.zeros((len(Newton_x_curve), n))
    for i in range(len(Newton_x_curve)):
        for j in range(n):
            Newton_x_curve_tmp[i][j] = pow(Newton_x_curve[i], n-1-j) 
    Newton_y_curve = mult(Newton_x_curve_tmp, Newton)
    plt.plot(A, b, 'o', color='r')
    plt.plot(Newton_x_curve, Newton_y_curve, color='k')
    plt.show()
