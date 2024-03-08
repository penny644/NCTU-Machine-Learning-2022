import numpy as np
import matplotlib.pyplot as plt

def random_generator(mean, var):
    #An easy-to-program approximate approach
    random = sum(np.random.uniform(0.0,1.0,12))-6
    random = random * pow(var,0.5) + mean
    return random

def basis_gen(n,a,w):
    #y = wx + e where e ~ N(0,a)
    x = np.random.uniform(-1,1)
    y = 0.0
    for i in range(n):
        y += w[i] * pow(x,i)
    y += random_generator(0,a)
    return x,y

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
def scalar_mul(a,A):
    ans = np.zeros((len(A),len(A[0])))
    for i in range(len(A)):
        for j in range(len(A[0])):
            ans[i][j] = a * A[i][j]
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
    # input b,n,a,w
    b = float(input("please input precision b: "))
    n = int(input("please input basis number n: "))
    a = float(input("please input a: "))
    w = input('please input w: ')
    w = w.strip()
    w = w[1:len(w)-1]
    w = [float(i.strip()) for i in w.split(',')]

    #declare some arrays
    prior_mean = np.zeros((n,1))
    prior_var = np.zeros((n,n))
    posterior_mean = np.zeros(n)
    posterior_var = np.zeros((n,n))
    plot_mean_10 = 0.0
    plot_var_10 = 0.0
    plot_mean_50 = 0.0
    plot_var_50 = 0.0
    predictive_mean = 0.0
    predictive_var = 0.0
    X = np.zeros((n,1))
    x_plot = []
    y_plot = []
    count = 0
    for i in range(n):
        #prior ~ N(0 , b-1I)
        #but we use prior_var = true_var-1
        prior_var[i][i] = b
    while 1:
        # use generator to generate x
        x,y = basis_gen(n,a,w)
        x_plot.append(x)
        y_plot.append(y)
        count += 1
        print("Add data point (%.5f, %.5f):" %(x,y))
        print("")
        for i in range(n):
            X[i][0] = pow(x,i)
        
        # a is 1/var of likelihood(input a)
        # C = aXXT + lambda, lambda-1 is prior var but here prior var is lambda
        # posterior = C-1
        tmp = mult(X, transpose(X))
        tmp = scalar_mul(1/a, tmp)
        C = tmp + prior_var
        posterior_var = LU(C)
        # posterior mean = C-1 (aXy + lambda * prior mean)
        tmp = scalar_mul(1/a,X)
        tmp = scalar_mul(y,tmp)
        tmp = tmp + mult(prior_var, prior_mean)
        posterior_mean = mult(posterior_var,tmp)
        
        #print Postirior mean and var
        print("Postirior mean:")
        for i in range(len(posterior_mean)):
            print("%.10f" %(posterior_mean[i][0]))
        print("")
        print("Postirior variance:")
        for i in range(len(posterior_var)):
            for j in range(len(posterior_var[0])):
                if j < len(posterior_var[0])-1:
                    print("%.10f, " %(posterior_var[i][j]), end='')
                else:
                    print("%.10f" %(posterior_var[i][j]))
        print("")

        #posterior predictive distribution
        #predictive_mean = (posterior_mean)T * X
        predictive_mean = mult(transpose(posterior_mean),X)

        #predictive_var = 1/a + (X)T * posterior_var * X
        predictive_var = mult(transpose(X),posterior_var)
        predictive_var = mult(predictive_var,X)
        predictive_var = a + predictive_var

        #when this x and the distribution of y
        print("Predictive distribution ~ N(%.5f, %.5f)" %(predictive_mean,predictive_var))
        print("")

        #we need information of 10th and 50th iteration
        if count == 10:
            plot_mean_10 = posterior_mean
            plot_var_10 = posterior_var
        elif count == 50:
            plot_mean_50 = posterior_mean
            plot_var_50 = posterior_var
        
        #check converge
        converge = 0.0
        for i in range(n):
            converge += abs(posterior_mean[i][0]-prior_mean[i][0])
        if converge < 1e-5:
            break
        prior_var = C
        prior_mean = posterior_mean
    
    #ground truth : w*x and varience is a
    plt.subplot(2,2,1)
    plt.title("Ground truth")
    x_plot_tmp = np.linspace(-2, 2, 3000)
    y_plot_tmp = np.zeros(3000)
    for i in range(3000):
        for j in range(n):
            y_plot_tmp[i] += w[j] * pow(x_plot_tmp[i],j)
    plt.plot(x_plot_tmp, y_plot_tmp, color='k')
    plt.plot(x_plot_tmp, (y_plot_tmp - a), color='r')
    plt.plot(x_plot_tmp, (y_plot_tmp + a), color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)

    #predict result : posterior_mean * x and var = a + XT * posterior_var * X
    plt.subplot(2,2,2)
    plt.title("Predict result")
    y_plot_tmp = np.zeros(3000)
    plot_var = np.zeros(3000)
    for i in range(3000):
        for j in range(n):
            X[j][0] = pow(x_plot_tmp[i],j)
        tmp = mult(transpose(X),posterior_var)
        tmp = mult(tmp,X)
        plot_var[i] = a + tmp
        for j in range(n):
            y_plot_tmp[i] += posterior_mean[j] * pow(x_plot_tmp[i],j)
    plt.plot(x_plot, y_plot, "o", markersize=10)
    plt.plot(x_plot_tmp, y_plot_tmp, color='k')
    plt.plot(x_plot_tmp, (y_plot_tmp - plot_var), color='r')
    plt.plot(x_plot_tmp, (y_plot_tmp + plot_var), color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)

    plt.subplot(2,2,3)
    plt.title("After 10 incomes")
    y_plot_tmp = np.zeros(3000)
    plot_var = np.zeros(3000)
    for i in range(3000):
        for j in range(n):
            X[j][0] = pow(x_plot_tmp[i],j)
        tmp = mult(transpose(X),plot_var_10)
        tmp = mult(tmp,X)
        plot_var[i] = a + tmp
        for j in range(n):
            y_plot_tmp[i] += plot_mean_10[j] * pow(x_plot_tmp[i],j)
    plt.plot(x_plot[0:10], y_plot[0:10], "o", markersize=10)
    plt.plot(x_plot_tmp, y_plot_tmp, color='k')
    plt.plot(x_plot_tmp, (y_plot_tmp - plot_var), color='r')
    plt.plot(x_plot_tmp, (y_plot_tmp + plot_var), color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)

    plt.subplot(2,2,4)
    plt.title("After 50 incomes")
    y_plot_tmp = np.zeros(3000)
    plot_var = np.zeros(3000)
    for i in range(3000):
        for j in range(n):
            X[j][0] = pow(x_plot_tmp[i],j)
        tmp = mult(transpose(X),plot_var_50)
        tmp = mult(tmp,X)
        plot_var[i] = a + tmp
        for j in range(n):
            y_plot_tmp[i] += plot_mean_50[j] * pow(x_plot_tmp[i],j)
    plt.plot(x_plot[0:50], y_plot[0:50], "o", markersize=10)
    plt.plot(x_plot_tmp, y_plot_tmp, color='k')
    plt.plot(x_plot_tmp, (y_plot_tmp - plot_var), color='r')
    plt.plot(x_plot_tmp, (y_plot_tmp + plot_var), color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)
    plt.show()

