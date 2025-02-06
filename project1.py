"""
Code for Scientific Computation Project 1
Please add college id here
CID: 02099078
"""


#===== Code for Part 1=====#
def part1(Xin,istar):
    """
    Sort list of integers in non-decreasing order
    Input:
    Xin: List of N integers
    istar: integer between 0 and N-1 (0<=istar<=N-1)
    Output:
    X: Sorted list
    """
    X = Xin.copy() 
    for i,x in enumerate(X[1:],1):
        if i<=istar:
            ind = 0
            for j in range(i-1,-1,-1):
                if x>=X[j]:
                    ind = j+1
                    break                   
        else:
            a = 0
            b = i-1
            while a <= b:
                c = (a+b) // 2
                if X[c] < x:
                    a = c + 1
                else:
                    b = c - 1
            ind = a
        
        X[ind+1:i+1] = X[ind:i]
        X[ind] = x

    return X



from time import time
from time import perf_counter_ns as time #alternate timer
import numpy as np
import matplotlib.pyplot as plt

def part1_time(Nistar):
    """Examine dependence of walltimes of part1 function on N and istar
        You may modify the input/output as needed.
    """
    #Initialise empty list of wall times
    tlist = []
    
    #Check wall time for different values of N
    Nlist = [10*(i+1) for i in range(200)]
    for N in Nlist:
        Xin = [np.random.randint(1, N) for i in range(N)]
        t1 = time()
        part1(Xin, Nistar)
        t2 = time()
        tlist.append((t2 - t1))
    
    #Compute least-squares fit
    a,b = np.polyfit(np.log(Nlist[8:]),np.log(tlist[8:]),1)
    print("a,b=",a,b)
    
    #Create plot
    plt.figure()
    plt.loglog(Nlist,tlist,'x--',label='computation')
    plt.loglog(Nlist,np.exp(b)*Nlist**a,'k:',label='least-squares fit, gradient=%f' %(a))
    plt.grid()
    plt.xlabel('N')
    plt.ylabel('wall time (ns)')
    plt.ylim([5*10**3, 5*10**7])
    plt.title('Wall time against N for istar='+ str(Nistar))
    plt.legend()
    
    return None

def part1_time2():
    #Initialise empty list of wall times
    tlist = []
    
    #Check wall time for different values of istar
    istar = [10*(i+1) for i in range(150)]
    for i in istar:
        rlist = [np.random.randint(1, 1500) for i in range(1500)]
        t1 = time()
        part1(rlist, i)
        t2 = time()
        tlist.append(t2 - t1)
    
    #Create plot
    plt.figure()
    plt.plot(istar, tlist)
    plt.grid()
    plt.xlabel('istar')
    plt.ylabel('wall time (ns)')
    plt.ylim([5*10**-3, 5*10**-2])
    plt.title('Wall time against istar')
    
    return None



#===== Code for Part 2=====#

from collections import defaultdict

def char2base4(S):
    #Convert seqeuence string to list of integers
    c2b = {'A':0 , 'C':1, 'G':2, 'T':3}
    L = [c2b[s] for s in S]
    return L

def heval(L, Base, Prime):
    #Convert list L to base-10 number mod Prime where Base specifies the base of L
    f = 0
    for l in L[:-1]:
        f = Base * (l + f)      
    h = (f + (L[-1])) % Prime
    return h

def part2(S,T,m):
    """Find locations in S of all length-m sequences in T
    Input:
    S,T: length-n and length-l gene sequences provided as strings

    Output:
    L: A list of lists where L[i] is a list containing all locations 
    in S where the length-m sequence starting at T[i] can be found.
    """
    
    #Size parameters
    n = len(S)
    l = len(T)
    Prime = 15485863
    Base = 4
    X = char2base4(S)
    Z = char2base4(T)
    
    L = [[] for i in range(l - m + 1)]
    
    d: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hp = heval(X[0:m], Base, Prime)
    d[hp][T[:m]] = 0
    
    bm = (Base ** m) % Prime
    
    #Calculate the hash values of the target patterns
    #Store mapping from hash value to index of the pattern 
    #as there may be multiple patterns with the same hash value
    for j in range(1, l - m + 1):
        hp = (Base * hp - int(Z[j - 1]) * bm + int(Z[j - 1 + m])) % Prime
        d[hp][T[j:j + m]] = j

    #Calculate the hash values of the substrings of the sequence
    hi = heval(X[:m], Base, Prime)
    if hi in d:
        #Check for any of the patterns with the same hash value
        #Matches the substring
        if T[:m] in d[hi]:
            L[d[hi][T[:m]]].append(0)

    for i in range(1, n - m + 1):
        #Update the rolling hash
        hi = (Base * hi - int(X[i - 1]) * bm + int(X[i - 1 + m])) % Prime
        if hi in d:
            #Check for any of the patterns with the same hash value
            #Matches the substring
            if S[i:i + m] in d[hi]:
                L[d[hi][S[i:i + m]]].append(i)
    return L

if __name__=='__main__':
    #Small example for part 2
    S = 'ATCGTACTAGTTATCGT'
    T = 'ATCGT'
    m = 3
    out = part2(S,T,m)
    
    #Large gene sequence from which S and T test sequences can be constructed
    infile = open("test_sequence.txt") #file from lab 3
    sequence = infile.read()
    infile.close()