"""Scientific Computation Project 3
02099078
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.signal import welch
from scipy.linalg import solve_banded
import time

#===== Code for Part 1=====#

def plot_field(lat,lon,u,time,levels=20):
    """
    Generate contour plot of u at particular time
    Use if/as needed
    Input:
    lat,lon: latitude and longitude arrays
    u: full array of wind speed data
    time: time at which wind speed will be plotted (index between 0 and 364)
    levels: number of contour levels in plot
    """
    plt.figure()
    plt.contourf(lon,lat,u[time,:,:],levels)
    plt.axis('equal')
    plt.grid()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    
    return None


def part1():
    """
    Code for part 1
    """ 
        
    #--- load data ---#
    d = np.load('data1.npz')
    lat = d['lat'];lon = d['lon'];u=d['u']
    days = np.array(range(365))
    #-------------------------------------#
    
    #--- Perform PCA ---#
    
    #Flatten the array so that each day is a data point
    X = u.reshape(365, 16*144).T

    #Centre the data
    A = np.transpose(X.T-X.mean(axis=1))

    #Perform Singular Value Decomposition (SVD)
    U,S,WT = np.linalg.svd(A)

    #Transform the data
    Atilde = np.dot(U.T,A)

    #Generate #plot_field visualisations
    plot_field(lat,lon,u,0,levels=20)
    plot_field(lat,lon,u,1,levels=20)
    plot_field(lat,lon,u,2,levels=20)
    plot_field(lat,lon,u,3,levels=20)
    plot_field(lat,lon,u,4,levels=20)
    plot_field(lat,lon,u,5,levels=20)
    
    #Plot average wind speed over time
    plt.figure(figsize=(8, 6))
    plt.plot(days, np.mean(u, axis=1))
    plt.title('Average wind speed over time')
    plt.xlabel('Day')
    plt.ylabel('Wind Speed')
    plt.grid()
    plt.show()
    
    #Plot average wind speed for different latitudes and longitudes
    plt.figure(figsize=(8, 6)) 
    plt.contourf(lon, lat, np.mean(u, axis=0))
    plt.colorbar(label='Wind Speed')
    plt.title('Average wind speed for different latitudes and longitudes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    
    #Plot 3D relative wind speed for different latitudes and longitudes
    wind_reshaped = U[:, 0].reshape(16,144)
    plt.figure(figsize=(9, 9))
    ax = plt.axes(projection='3d')
    ax.contourf(lon, lat, wind_reshaped, 400)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Relative Wind Speed")
    ax.xaxis.labelpad=20
    ax.yaxis.labelpad=20
    ax.zaxis.labelpad=2
    plt.grid()
    plt.show()
    
    # --- Perform PCA Method 2 ---
    #Flatten the array so that each day is a data point
    flattened = u.reshape(u.shape[0], -1)
    #Centre the data
    mean = np.mean(flattened, axis=0)
    centred = flattened - mean
    #Perform Singular Value Decomposition (SVD)
    U2, S2, WT2 = np.linalg.svd(centred)
    #Number of principal components to retain
    ncomponents = 1
    svalues = np.diag(S2[:ncomponents])
    components = U2[:, :ncomponents]
    #Transform the data
    data = np.dot(components, svalues)
    #Plot the dominant frequencies of wind speed using the first principal component
    plt.figure(figsize=(8, 6))
    plt.plot(data[:, 0])
    plt.xlabel('Day')
    plt.ylabel('Weight')
    plt.title('First principal component against time')
    plt.grid()
    plt.show()
    
    #Plot scree plot
    eigenvalues = S ** 2 
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(eigenvalues) + 1), np.log(eigenvalues), marker='x', linestyle='-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Log of Eigenvalue')
    plt.grid()
    plt.show()
    
    #Plot cumulative variance explained by each principal component
    total = np.sum(eigenvalues)
    explained = eigenvalues / total
    cumulative_explained = np.cumsum(explained)
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(1, len(eigenvalues) + 1), cumulative_explained, marker='x', linestyle='-')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative explained variance for number of principal components')
    plt.grid()
    plt.show()
    
    #Plot Welch's Method
    fxx, Pxx = welch(u.flatten())
    plt.figure(figsize=(8, 6))
    plt.semilogy(fxx, Pxx)
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.title('Dominant values of wind speed using Welch\'s Method')
    plt.grid()
    plt.show()
    f = fxx[Pxx==Pxx.max()][0]
    print("f=",f)
    print("1/f=",1/f) 
    
    return None



#===== Code for Part 2=====#
from scipy.linalg import solve_banded
import scipy as sp

def part2(f, method=2):
    """
    Question 2.1 i)
    Input:
        f: m x n array
        method: 1 or 2, interpolation method to use
    Output:
        fI: interpolated data (using method)
    """

    m, n = f.shape
    fI = np.zeros((m - 1, n))  # Initialize interpolated array

    if method == 1:
        fI = 0.5 * (f[:-1, :] + f[1:, :])  # Method 1 interpolation
    else:
        # Coefficients for method 2
        alpha = 0.3
        a = 1.5
        b = 0.1

        # Coefficients for near-boundary points
        a_bc, b_bc, c_bc, d_bc = (5 / 16, 15 / 16, -5 / 16, 1 / 16)
        
        #Construct LHS
        lhs_diag = np.zeros((3, m-1))
        lhs_diag[0, 2:] = alpha * np.ones(m - 3)
        lhs_diag[1] = np.ones(m - 1)
        lhs_diag[2, :-2] = alpha * np.ones(m - 3)
        
        #Construct banded matrix
        rhs_diag = np.empty(6, dtype=object)
        rhs_diag[0] = np.zeros(m-3)
        rhs_diag[1] = b/2 * np.ones(m-2)
        rhs_diag[2] = a/2 * np.ones(m-1)
        rhs_diag[3] = a/2 * np.ones(m-1)
        rhs_diag[4] = b/2 * np.ones(m-2)
        rhs_diag[5] = np.zeros(m-2)

        #Assign coefficients
        rhs_diag[0][-1] = d_bc
        rhs_diag[1][-1] = c_bc
        rhs_diag[2][0] = a_bc 
        rhs_diag[2][-1] = b_bc
        rhs_diag[3][0] = b_bc 
        rhs_diag[3][-1] = a_bc
        rhs_diag[4][0] = c_bc
        rhs_diag[5][0] = d_bc

        #Create banded matrix from coefficients
        rhs_coeffs = sp.sparse.diags(rhs_diag, np.array([-2, -1, 0, 1, 2, 3]), shape = (m-1,m))
        rhs_coeffs = rhs_coeffs.todense()
        
        #Solve system using solve_banded
        rhs = rhs_coeffs @ f
        fI = solve_banded((1, 1), lhs_diag, rhs)

    return fI



def part2_analyze(m=50, n=40):
    
    def multiscale_data(m, n):
        #Generate data
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, m)
        xg, yg = np.meshgrid(x, y)
        data = np.sin(10 * np.pi * xg) + np.sin(2 * np.pi * yg)
        return x, data[0], data[1] 

    #Store generated data
    x, original_data_1, original_data_2 = multiscale_data(m, n)
    
    #Interpolate using method 1 and 2
    method_1 = part2(original_data_1, method=1)
    method_2 = part2(original_data_2, method=2)
    
    #Plot method 1 interpolation
    plt.figure(figsize=(8, 6))
    plt.plot(x, original_data_1, label='Original Data', linestyle='-', marker='o')
    plt.plot(x, method_1, label='Method 1', linestyle='--', marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolation method 1 applied to generated data')
    plt.legend()
    plt.grid()
    plt.show()
    
    #Plot method 2 interpolation
    plt.figure(figsize=(8, 6))
    plt.plot(x, original_data_2, label='Original Data', linestyle='-', marker='o')
    plt.plot(x, method_2, label='Method 2', linestyle='--', marker='x', color = 'green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolation method 2 applied to generated data')
    plt.legend()
    plt.grid()
    plt.show()
    
    return None



def part2_analyze2():
    
    def multiscale_data(m, n):
        #Generate data
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, m)
        xg, yg = np.meshgrid(x, y)
        data = np.sin(10 * np.pi * xg) + np.sin(2 * np.pi * yg)
        return data
    
    #Create grid sizes for testing
    grid_sizes = [(50, 40), (100, 80), (150, 120), (200, 160), (250, 200)]
    
    #Initialise lists
    method_1_times = []
    method_2_times = []
    
    for m, n in grid_sizes:
        #Store generated data
        original_data = multiscale_data(m, n)
        
        #Interpolate using method 1
        start = time.time()
        method_1 = part2(original_data, method=1)
        dt = time.time() - start
        method_1_times.append(dt)
        
        #Interpolate using method 2
        start = time.time()
        method_2 = part2(original_data, method=2)
        dt = time.time() - start
        method_2_times.append(dt)
        
    #Plot wall times for different grid sizes
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(grid_sizes)), method_1_times, marker='x', label='Method 1')
    plt.plot(range(len(grid_sizes)), method_2_times, marker='x', label='Method 2')
    plt.xticks(range(len(grid_sizes)), grid_sizes)
    plt.xlabel('Grid Sizes (m, n)')
    plt.ylabel('Wall Time')
    plt.title('Wall times for different array sizes of f')
    plt.legend()
    plt.grid()
    plt.show()
    
    return None



#===== Code for Part 3=====#
def part3q1(y0,alpha,beta,b,c,tf=200,Nt=800,err=1e-6,method="RK45"):
    """
    Part 3 question 1
    Simulate system of 2n nonlinear ODEs

    Input:
    y0: Initial condition, size 2*n array
    alpha,beta,b,c: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x 2*n array containing y at
            each time step including the initial condition.
    """
    
    #Set up parameters, arrays

    n = y0.size//2
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,2*n))
    yarray[0,:] = y0


    def RHS(t,y):
        """
        Compute RHS of model
        """
        #add code here
        u = y[:n];v=y[n:]
        r2 = u**2+v**2
        nu = r2*u
        nv = r2*v
        cu = np.roll(u,1)+np.roll(u,-1)
        cv = np.roll(v,1)+np.roll(v,-1)

        dydt = alpha*y
        dydt[:n] += beta*(cu-b*cv)-nu+c*nv+b*(1-alpha)*v
        dydt[n:] += beta*(cv+b*cu)-nv-c*nu-b*(1-alpha)*u

        return dydt


    sol = solve_ivp(RHS, (tarray[0],tarray[-1]), y0, t_eval=tarray, method=method,atol=err,rtol=err)
    yarray = sol.y.T 
    return tarray,yarray



def part3_analyze(display = False, c=0.5):
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """

    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0 = np.zeros(2*n)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #---Example code for computing solution, use/modify/discard as needed---#
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    u,v = y[:,:n],y[:,n:]

    if display:
        plt.figure()
        plt.contourf(np.arange(n),t,u,20)

    #-------------------------------------------#
    
    return None

#Contour Plots (uncomment to run)
#for c in np.linspace(0.5,1.5, 11):
    #part3_analyze(display = True, c=c)
    #plt.title(f"c ={c}")



def part3_analyze2(u_out = False, c = 0.5):#add/remove input variables if needed
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """

    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0 = np.zeros(2*n)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #---Example code for computing solution, use/modify/discard as needed---#
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    u,v = y[:,:n],y[:,n:]

    #-------------------------------------------#

    if u_out:
        #only consider i from 100 to n-100
        return u[:,100:-100]

    return None

#Trajectories (uncomment to run)
#u05 = part3_analyze(u_out = True, c = 0.5)
#plt.figure()
#plt.plot(u05[:,::100])
#plt.title(f"c = 0.5")

#u10 = part3_analyze(u_out = True, c = 1.0)
#plt.figure()
#plt.plot(u10[:,::100])
#plt.title(f"c = 1.0")

#u13 = part3_analyze(u_out = True, c = 1.3)
#plt.figure()
#plt.plot(u13[:,::100])
#plt.title(f"c = 1.3")

#u15 = part3_analyze(u_out = True, c = 1.5)
#plt.figure()
#plt.plot(u15[:,::100])
#plt.title(f"c = 1.5")

def part3_analyze3(c = 1.3):
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """

    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0 = np.zeros(2*n)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #---Example code for computing solution, use/modify/discard as needed---#
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    u,v = y[:,:n],y[:,n:]

    #-------------------------------------------#
    
    #Calculate the correlation dimension
    epsilon_values = np.logspace(-1 , 1, n)
    D = sp.spatial.distance.pdist(u)
    distances = []
    nc = 2 / (n*(n-1))
    for epsilon in epsilon_values:
        distances.append((len(D[D<i])) * nc)
    linear_regress = np.polyfit(np.log(epsilon_values[3000:n]), np.log(distances[3000:n]), 1)
    return linear_regress[0]

def correlation_dimension():
    """
    Computes and plots correlation dimension for values of c between 0.5 and 1.5
    """
    c_values = np.linspace(0.5, 1.5, 21)
    dimensions = []

    for c in c_values:
        #Compute the correlation dimension for each c value
        dimension = part3_analyze3(c)
        dimensions.append(dimension)

    #Plot the correlation dimension against c values
    plt.figure(figsize=(8, 5))
    plt.plot(c_values, dimensions, marker='x', linestyle='-')
    plt.xlabel('Value of c')
    plt.ylabel('Correlation Dimension')
    plt.title('Correlation dimension for various c values')
    plt.grid()
    plt.show()
    
    return None

def part3_analyze4(c = 1.3):
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """

    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0 = np.zeros(2*n)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #---Example code for computing solution, use/modify/discard as needed---#
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    u,v = y[:,:n],y[:,n:]

    #-------------------------------------------#
    
    #Welch's Method
    dt = 200/800
    u_new = u[:, 2000]
    fxx,Pxx = welch(u_new)
    plt.figure()
    plt.semilogy(fxx,Pxx)
    plt.title(f"The power density of different frequencies, c = {c}.")
    plt.xlabel(r'$f$')
    plt.ylabel(r'$P_{xx}$')
    plt.grid()
    f = fxx[Pxx==Pxx.max()][0]
    print("f=",f)
    print("dt,1/f=",t[1]-t[0],1/f)

    return None



def part3q2(x,c=1.0):
    """
    Code for part 3, question 2
    """
    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    y0 = np.zeros(2*n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)

    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #Compute solution
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    A = y[:,:n]

    #Analyze code here
    l1,v1 = np.linalg.eigh(A.T.dot(A))
    v2 = A.dot(v1)
    A2 = (v2[:,:x]).dot((v1[:,:x]).T)
    e = np.sum((A2.real-A)**2)

    return A2.real,e


if __name__=='__main__':
    x=None #Included so file can be imported
    #Add code here to call functions above if needed
