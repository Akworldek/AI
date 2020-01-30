# -*- coding: utf-8 -*-
"""
Math Modeling

@author: akul
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detailed_Slope(f):
    resultant=[0,0]
    for i in range(len(f)):
        #select a point/ vertex
        x=i
        y=f[i]
        for j in range(i,len(f)):
            #select the destination points
            xj=j
            yj=f[j]
            vector_ij=[xj-x,yj-y]
            resultant[0]+=vector_ij[0]
            resultant[1]+=vector_ij[1]
    return resultant[1]/resultant[0]

num=int(input("Real Points= "))
df=pd.read_csv("D:/IPG2211A2N.csv")
f=list(df['IPG2211A2N'])[:num]
L=len(f)
error=0.07 #7%


diff1=[(f[term+1]-f[term-1])/2 for term in range(1,L-1)]

meanSlope=sum(diff1)/(L-2)
meanSlope_left, meanSlope_right=meanSlope-error*meanSlope, meanSlope+error*meanSlope



freq=[diff1[term]/f[term+1] for term in range(L-2)]

meanFreq=sum(freq)/(L-2)
meanFreq_left, meanFreq_right=meanFreq-error*meanFreq, meanFreq+error*meanFreq

stdSlope=[diff1[point]-meanSlope for point in range(L-2)]
varSlope=[std**2 for std in stdSlope]

plt.figure(figsize=(10,8))
plt.plot(diff1,'-o',  label="f'(x)")
plt.plot(list(range(L-2)),[meanSlope]*(L-2), label='mean Slope= %.4f'%meanSlope)
#plt.plot(list(range(L-2)),[meanSlope_left]*(L-2), label='mean Slope_l')
#plt.plot(list(range(L-2)),[meanSlope_right]*(L-2), label='mean Slope_r')
plt.legend(loc= 'upper right',prop={'size': 10})
plt.title("Slope")
plt.show()


plt.figure(figsize=(10,8))
plt.plot(freq,'-o', label="f'(x)/f(x)")
plt.plot(list(range(L-2)),[meanFreq]*(L-2), label='mean freq= %.5f'%meanFreq)
freqS=detailed_Slope(freq)
plt.plot([freqS*t+freq[0] for t in range(L-2)],'--', label='change in freq= %.5f'%freqS)
#plt.plot(list(range(L-2)),[meanFreq_left]*(L-2), label='mean freq_l')
#plt.plot(list(range(L-2)),[meanFreq_right]*(L-2), label='mean freq_r')
plt.legend(loc= 'upper right',prop={'size': 10})
plt.title("Frequency")
plt.show()

if abs(freqS)<0.00001:
    messageForFreq=" The frequency is constant, thus implying that the function is purely exponential: f(t)=Ae^(kt)+B"
else:
    messageForFreq=" The frequency varies with t, is hence dependant on t.\n f(t) follows a power series pattern"
    
print("Conclusion from Frequency:\n\n"+messageForFreq+"\n\n")



plt.figure(figsize=(10,8))
meanVar=sum(varSlope)/len(varSlope)
plt.plot(varSlope,'-o')
plt.plot(list(range(L-2)),[meanVar]*(L-2), label='mean Var of Slope= %.5f'%meanVar)
vSS=detailed_Slope(varSlope)
plt.plot([vSS*t+varSlope[0] for t in range(L-2)],'--', label='change in Var= %.5f'%vSS)
plt.legend(loc= 'upper right',prop={'size': 10})
plt.title("Variance of Slope")
plt.show()

if vSS>0:
    messageForVariance=" Since the variance of slope is increasing from the mean, the curve is diverging from a linear projection.\n This means the curve is either shooting up, or saturating.\n Either way, the curve is exponential."
elif vSS<0:
    messageForVariance=" Since the variance of slope is decreasing from the mean, the curve is converging to a linear projection. \n The curve eventually follows a linear pattern."
else:
    messageForVariance=" Since the variance of slope is constant, the curve is oscillating.\n It follows a hypergeometric trend, but oscillating around the mean."

print("Conclusion from Variance of Slope:\n\n"+messageForVariance+"\n\n")




plt.figure(figsize=(10,8))
plt.plot(diff1,'-o',  label="f'(x)")
diff2_scal=detailed_Slope(diff1)
plt.plot([diff2_scal*t+diff1[0] for t in range(L-2)],'--', label='second slope= %.5f'%diff2_scal)
plt.plot(list(range(L-2)),[meanSlope]*(L-2), label='mean Slope= %.2f'%meanSlope)
plt.legend(loc= 'upper right',prop={'size': 10})
plt.title("diff 1 and 2")
plt.show()

if diff2_scal>0:
    messageForDiff2=" Since the second differential is positive, the curve is time-divergent and increasing.\n This implies that the trend follows a non-linear (or exponential) projection."
elif diff2_scal<0:
    messageForDiff2=" Since the second differential is negative, the curve is time-convergent. \n The trend is non-linear, but attains saturation in the near future."
else:
    messageForDiff2=" Since the second differential is 0, the curve is linear."

print("Conclusion from Second Differential:\n\n"+messageForDiff2+"\n\n")

#
#if abs(freqS)<0.00001:
#    #f(t)=A.exp(Kt)+B
#    K=meanFreq
#    A=(freq[1]-freq[0])/(math.exp(K)-1)
#    B=freq[0]-A
#    print("f(t) = %.5f exp(%.5f t) + %.5f"%(A,K,B))

 #f(t)=A.exp(Kt)+B
K=meanFreq
A=(f[1]-f[0])/(math.exp(K)-1)
B=f[0]-A
print("f(t) = %.5f exp(%.5f t) + %.5f"%(A,K,B))

#ft=[A*math.exp(K*t)+B for t in range(L,L+50)]
ft=[A*math.exp(K*t)+B for t in range(L+50)]
plt.figure(figsize=(10,8))
plt.plot(df['IPG2211A2N'][:(num+20+50)],'o')
plt.plot(f,'-o')
#plt.plot(list(range(L,L+50)),ft,'r*')
plt.plot(ft,'r-*')