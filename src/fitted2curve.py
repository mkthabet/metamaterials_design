import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.metrics import mean_squared_error

# %%
def fitted_data_1res (param,b,t,w,w0,p):
  ans =  (param[0]**2+2*param[0]*((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2)) + 
   (((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2)))**2 +
    (((b[0]*t[0]*(np.sin(p[0])*(w-w0[0])-np.cos(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2)))**2)**(1/2) 
  return ans

# %%
def fitted_data_2res (param,b,t,w,w0,p):
  ans =  (param[0]**2+2*param[0]*((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2)+ 
     (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1])))/((w-w0[1])**2+(param[2]+t[1])**2)) + 
   (((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+
    (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2)))**2 +
    (((b[0]*t[0]*(np.sin(p[0])*(w-w0[0])-np.cos(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+ 
    (b[1]*t[1]*(np.sin(p[1])*(w-w0[1])-np.cos(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2)))**2)**(1/2) 
  return ans

# %%
def fitted_data_3res (param,b,t,w,w0,p):
  ans = (param[0]**2+2*param[0]*((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2)+
     (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1])))/((w-w0[1])**2+(param[2]+t[1])**2)+
     (b[2]*t[2]*(np.cos(p[2])*(w-w0[2])+np.sin(p[2])*(param[3]+t[2])))/((w-w0[2])**2+(param[3]+t[2])**2))+
    (((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(0)*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+
     (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2))+
     (b[2]*t[2]*(np.cos(p[2])*(w-w0[2])+np.sin(p[2])*(param[3]+t[2]))/((w-w0[2])**2+(param[3]+t[2])**2)))**2+
    (((b[0]*t[0]*(np.sin(p[0])*(w-w0[0])-np.cos(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+
     (b[1]*t[1]*(np.sin(p[1])*(w-w0[1])-np.cos(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2))+
     (b[2]*t[2]*(np.sin(p[2])*(w-w0[2])-np.cos(p[2])*(param[3]+t[2]))/((w-w0[2])**2+(param[3]+t[2])**2)))**2)**(1/2)
  return ans

# %%
def fitted_data_4res (param,b,t,w,w0,p):
  ans = (param[0]**2+2*param[0]*((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2)+
     (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1])))/((w-w0[1])**2+(param[2]+t[1])**2)+
     (b[2]*t[2]*(np.cos(p[2])*(w-w0[2])+np.sin(p[2])*(param[3]+t[2])))/((w-w0[2])**2+(param[3]+t[2])**2)+
     (b[3]*t[3]*(np.cos(p[3])*(w-w0[3])+np.sin(p[3])*(param[4]+t[3])))/((w-w0[3])**2+(param[4]+t[3])**2))+
    (((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(0)*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+
     (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2))+
     (b[2]*t[2]*(np.cos(p[2])*(w-w0[2])+np.sin(p[2])*(param[3]+t[2]))/((w-w0[2])**2+(param[3]+t[2])**2))+
     (b[3]*t[3]*(np.cos(p[3])*(w-w0[3])+np.sin(p[3])*(param[4]+t[3]))/((w-w0[3])**2+(param[4]+t[3])**2)))**2+
    (((b[0]*t[0]*(np.sin(p[0])*(w-w0[0])-np.cos(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+
     (b[1]*t[1]*(np.sin(p[1])*(w-w0[1])-np.cos(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2))+
     (b[2]*t[2]*(np.sin(p[2])*(w-w0[2])-np.cos(p[2])*(param[3]+t[2]))/((w-w0[2])**2+(param[3]+t[2])**2))+
     (b[3]*t[3]*(np.sin(p[3])*(w-w0[3])-np.cos(p[3])*(param[4]+t[3]))/((w-w0[3])**2+(param[4]+t[3])**2)))**2)**(1/2)
  return ans

# %%
def fitted_data_5res (param,b,t,w,w0,p):
  ans = (param[0]**2+2*param[0]*((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2)+
     (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1])))/((w-w0[1])**2+(param[2]+t[1])**2)+
     (b[2]*t[2]*(np.cos(p[2])*(w-w0[2])+np.sin(p[2])*(param[3]+t[2])))/((w-w0[2])**2+(param[3]+t[2])**2)+
     (b[3]*t[3]*(np.cos(p[3])*(w-w0[3])+np.sin(p[3])*(param[4]+t[3])))/((w-w0[3])**2+(param[4]+t[3])**2)+
     (b[4]*t[4]*(np.cos(p[4])*(w-w0[4])+np.sin(p[4])*(param[5]+t[4])))/((w-w0[4])**2+(param[5]+t[4])**2))+
    (((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(0)*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+
     (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2))+
     (b[2]*t[2]*(np.cos(p[2])*(w-w0[2])+np.sin(p[2])*(param[3]+t[2]))/((w-w0[2])**2+(param[3]+t[2])**2))+
     (b[3]*t[3]*(np.cos(p[3])*(w-w0[3])+np.sin(p[3])*(param[4]+t[3]))/((w-w0[3])**2+(param[4]+t[3])**2))+
     (b[4]*t[4]*(np.cos(p[4])*(w-w0[4])+np.sin(p[4])*(param[5]+t[4]))/((w-w0[4])**2+(param[5]+t[4])**2)))**2+
    (((b[0]*t[0]*(np.sin(p[0])*(w-w0[0])-np.cos(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+
     (b[1]*t[1]*(np.sin(p[1])*(w-w0[1])-np.cos(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2))+
     (b[2]*t[2]*(np.sin(p[2])*(w-w0[2])-np.cos(p[2])*(param[3]+t[2]))/((w-w0[2])**2+(param[3]+t[2])**2))+
     (b[3]*t[3]*(np.sin(p[3])*(w-w0[3])-np.cos(p[3])*(param[4]+t[3]))/((w-w0[3])**2+(param[4]+t[3])**2))+
     (b[4]*t[4]*(np.sin(p[4])*(w-w0[4])-np.cos(p[4])*(param[5]+t[4]))/((w-w0[4])**2+(param[5]+t[4])**2)))**2)**(1/2)
  return ans

# %%
def fitted_data_6res (param,b,t,w,w0,p):
  ans = (param[0]**2+2*param[0]*((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2)+
     (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1])))/((w-w0[1])**2+(param[2]+t[1])**2)+
     (b[2]*t[2]*(np.cos(p[2])*(w-w0[2])+np.sin(p[2])*(param[3]+t[2])))/((w-w0[2])**2+(param[3]+t[2])**2)+
     (b[3]*t[3]*(np.cos(p[3])*(w-w0[3])+np.sin(p[3])*(param[4]+t[3])))/((w-w0[3])**2+(param[4]+t[3])**2)+
     (b[4]*t[4]*(np.cos(p[4])*(w-w0[4])+np.sin(p[4])*(param[5]+t[4])))/((w-w0[4])**2+(param[5]+t[4])**2)+
     (b[5]*t[5]*(np.cos(p[5])*(w-w0[5])+np.sin(p[5])*(param[6]+t[5])))/((w-w0[5])**2+(param[6]+t[5])**2))+
    (((b[0]*t[0]*(np.cos(p[0])*(w-w0[0])+np.sin(0)*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+
     (b[1]*t[1]*(np.cos(p[1])*(w-w0[1])+np.sin(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2))+
     (b[2]*t[2]*(np.cos(p[2])*(w-w0[2])+np.sin(p[2])*(param[3]+t[2]))/((w-w0[2])**2+(param[3]+t[2])**2))+
     (b[3]*t[3]*(np.cos(p[3])*(w-w0[3])+np.sin(p[3])*(param[4]+t[3]))/((w-w0[3])**2+(param[4]+t[3])**2))+
     (b[4]*t[4]*(np.cos(p[4])*(w-w0[4])+np.sin(p[4])*(param[5]+t[4]))/((w-w0[4])**2+(param[5]+t[4])**2))+
     (b[5]*t[5]*(np.cos(p[5])*(w-w0[5])+np.sin(p[5])*(param[6]+t[5]))/((w-w0[5])**2+(param[6]+t[5])**2)))**2+
    (((b[0]*t[0]*(np.sin(p[0])*(w-w0[0])-np.cos(p[0])*(param[1]+t[0])))/((w-w0[0])**2+(param[1]+t[0])**2))+
     (b[1]*t[1]*(np.sin(p[1])*(w-w0[1])-np.cos(p[1])*(param[2]+t[1]))/((w-w0[1])**2+(param[2]+t[1])**2))+
     (b[2]*t[2]*(np.sin(p[2])*(w-w0[2])-np.cos(p[2])*(param[3]+t[2]))/((w-w0[2])**2+(param[3]+t[2])**2))+
     (b[3]*t[3]*(np.sin(p[3])*(w-w0[3])-np.cos(p[3])*(param[4]+t[3]))/((w-w0[3])**2+(param[4]+t[3])**2))+
     (b[4]*t[4]*(np.sin(p[4])*(w-w0[4])-np.cos(p[4])*(param[5]+t[4]))/((w-w0[4])**2+(param[5]+t[4])**2))+
     (b[5]*t[5]*(np.sin(p[5])*(w-w0[5])-np.cos(p[5])*(param[6]+t[5]))/((w-w0[5])**2+(param[6]+t[5])**2)))**2)**(1/2)
  return ans

# %%
#Function to Assign S parameter values to the given fitted values
#inputs are the fitted paramters given as a single array
#The input array is arranged the same it is present in the file of fitted data with thier corresponding structural parmaters
# array = [n_res, a, amp1, phase1, freq1, lwidth1, coeff1, amp2, phase2, fre2, ...........]

def FittedDataToCurve (Fittedarray):

  n_res= Fittedarray[0]
  w = np.arange(4,7.003,.003) #can be adjusted according to given data range
  if n_res==1:
      b=Fittedarray[2]
      p=Fittedarray[3]
      w0=Fittedarray[4]
      t=Fittedarray[5]
      param = np.take(Fittedarray, [1,6])
      ans=fitted_data_1res (param,b,t,w,w0,p)
  elif n_res==2:
      b=np.take(Fittedarray, [2,7])
      p=np.take(Fittedarray, [3,8])
      w0=np.take(Fittedarray, [4,9])
      t=np.take(Fittedarray, [5,10])
      param = np.take(Fittedarray, [1,6,11])
      ans=fitted_data_2res (param,b,t,w,w0,p)
  elif n_res==3:
      b=np.take(Fittedarray, [2,7,12])
      p=np.take(Fittedarray, [3,8,13])
      w0=np.take(Fittedarray, [4,9,14])
      t=np.take(Fittedarray, [5,10,15])
      param = np.take(Fittedarray, [1,6,11,16])
      ans=fitted_data_3res (param,b,t,w,w0,p)  
  elif n_res==4:
      b=np.take(Fittedarray, [2,7,12,17])
      p=np.take(Fittedarray, [3,8,13,18])
      w0=np.take(Fittedarray, [4,9,14,19])
      t=np.take(Fittedarray, [5,10,15,20])
      param = np.take(Fittedarray, [1,6,11,16,21])
      ans=fitted_data_4res (param,b,t,w,w0,p)
  elif n_res==5:
      b=np.take(Fittedarray, [2,7,12,17,22])
      p=np.take(Fittedarray, [3,8,13,18,23])
      w0=np.take(Fittedarray, [4,9,14,19,24])
      t=np.take(Fittedarray, [5,10,15,20,25])
      param = np.take(Fittedarray, [1,6,11,16,21,26])
      ans=fitted_data_5res (param,b,t,w,w0,p)  
  elif n_res==6:
      b=np.take(Fittedarray, [2,7,12,17,22,27])
      p=np.take(Fittedarray, [3,8,13,18,23,28])
      w0=np.take(Fittedarray, [4,9,14,19,24,29])
      t=np.take(Fittedarray, [5,10,15,20,25,30])
      param = np.take(Fittedarray, [1,6,11,16,21,26,31])        
      ans=fitted_data_6res (param,b,t,w,w0,p)  


  return ans,w

if __name__ == "__main__":
  # %%
  #import an example of  fitted parameters
  data = pd.read_excel('all_fitted_data.csv')
  data_arr= np.asarray(data)
  Fittedarray=data_arr[965,4:-1]

  #plot scurve of fitted data
  scurve,w= FittedDataToCurve (Fittedarray)
  plt.plot(w, scurve, '--', color ='blue', label ='Fitted data')
  plt.legend()
  plt.show()


