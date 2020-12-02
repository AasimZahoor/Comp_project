#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:58:01 2020

@author: aj3008
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:56:39 2020

@author: aj3008
"""

#------------------------------------importing libraries--------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp    # np.exp doesn't save smaller values but sp.exp does





# This code is divided into two parts 1) Data generation 2) Parameter estimation 

#----------------------------------------Generating DATA----------------------------------------------------
#time, mass needs to be in meters
cc=7.424*10**(-28)     # unit is m/kg. Converts mass into distance

msun=2*10**30*cc # unit is meters


#parameters of the binary system
m1=30*msun       #mass of blackhole 1
m2=35*msun       #mass of blackhole 2

r=100*10**6 #meters  distance at which we are detecting, can be anything
Ri=1000*10**3 #meters initial orbital separation 


M=m1+m2 
u=m1*m2/M


dt=100000 #meters (time step)
h=[]     #saves the strain values
R=[]     #saves the corresponding separation values
t=[]     #saves the time
ti=0
i=0     
while i<=7000:     #7000 to keep the data well within the inspiral phase. Calculated from a paper.
    print(Ri)
    noise=np.random.normal(0,1*10**(-7))
    hi=u*M/(r*Ri)+noise  #strain
    h.append(hi)
    hn.append(-hi)
    R.append(Ri)
    Ri=Ri-dt*(u*M**2)/Ri**3    #using forward difference method
    t.append(ti)
    ti=ti+dt
    i=i+1
plt.title("Data v/s time")
plt.xlabel("time in meters")
plt.ylabel("Strain")
plt.plot(t,h)

plt.savefig("Images/data")
















#%%
#----------------------------------------Parameter estimation------------------------------------------------    
#Defining the likelihood function
def Like(data,mod):
    '''
    

    Parameters
    ----------
    data : strain data.
    mod : strain values from model.

    Returns
    -------
    Likelihood of the model with given parameters.

    '''
    h=mod[0]
    R=mod[1]
    sum=0
    for i in range(len(data)):
        sum=sum-0.5*(data[i]-h[i])**2
    return(sum+np.log(1/(np.sqrt(2*3.14))))

#Definging the model
def model(y):
    '''
    

    Parameters
    ----------
    y :array of parameters: [mass, sigma of noise]

    Returns
    -------
    returns an array: [strain, orbital separation values]

    '''
    cc=7.424*10**(-28)     #(m/kg)convert mass into distance

    msun=2*10**30*cc #meters
    
    m1=y[0]*msun #mass of blackhole 1 
    m2=y[1]*msun #mass of blackhole 2
    
    r=100*10**6 #meters  distance at which we are detecting, can be anything
    Ri=1000*10**3 #meters initial orbital separation
    
    
    M=m1+m2
    u=m1*m2/M
    
    
    dt=100000 #time step
    h=[] #strain values
    i=0
    while i<=7000:
        noise=np.random.normal(0,abs(y[2]))
        hi=u*M/(r*Ri)+noise #strain
        h.append(hi)
        R.append(Ri)
        Ri=Ri-dt*(u*M**2)/Ri**3  #orbital separation  #forward difference method
        i=i+1
    return(h,R)
    
    

#-----------------------------------------------MCMC----------------------------------------------------------
i=0
iterations=1000 
xt=[40,40,2*10**(-7)]      #best initial guess
distd=[]                   #to store the parameters
y=[40,40,2*10**(-7)]       #defined twice, to store the changed values
while i<iterations:     
    print(i)      #keep track of iterations
    #based on the algorithm we studied in class
    y[0]=abs(np.random.normal(xt[0],0.5))
    num=Like(h,model(y))
    den=Like(h,model(xt))
    r=sp.exp(num-den)
    if r>=1:
        xt[0]=y[0]
    else:
        u=np.random.uniform(0,1)
        if u<=r:
            xt[0]=y[0]
            
        else:
            xt[0]=xt[0]
    y[1]=abs(np.random.normal(xt[1],0.5))
    num=Like(h,model(y))
    den=Like(h,model(xt))
    r=sp.exp(num-den)
    if r>=1:
        xt[1]=y[1]
    else:
        u=np.random.uniform(0,1)
        if u<=r:
            xt[1]=y[1]
            
        else:
            xt[1]=xt[1]
    # uncomment the next lines to make the MCMC look for sigma values
    # y[2]=abs(np.random.normal(xt[2],10**(-7)))
    # num=Like(h,model(y))
    # den=Like(h,model(xt))
    # r=sp.exp(num-den)
    # if r>=1:
    #     xt[2]=y[2]
    # else:
    #     u=np.random.uniform(0,1)
    #     if u<=r:
    #         xt[2]=y[2]
            
    #     else:
    #         xt[2]=xt[2]
    i=i+1
    xt=list(xt)
    distd.append(xt)

print("done")

#calculating probability for each parameter value
prob=[] #saves probability values
sum=0
for i in range(len(distd)):
    print(i)
    a=sp.exp(Like(h,model(distd[i])))
    sum=sum+a     #sum of all probabilities, for normalising
    prob.append(a)
prob=[prob[i]/sum for i in range(len(prob))]    #normalising  
ind=prob.index(max(prob))                       #finding max probability
print("the value of parameters is",distd[ind])

#plotting
mass1=distd[ind][0]
mass2=distd[ind][1]
sigma=distd[ind][2]
distd=np.array(distd)
x=distd[:,0]
y=distd[:,1]
plt.title("Sampling through parameter space")
plt.xlabel("m1")
plt.ylabel("m2")
plt.axvline(x=mass1)   #most probable values 
plt.axhline(y=mass2)
plt.scatter(x, y,color="red")
plt.savefig("Images/corner")
plt.show()





