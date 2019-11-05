from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

a=np.array([[455,877,10,7],[8,11,8,2],[1,11,8,7],[8,11,8,8],[8,11,8,3]])
df=DataFrame(a, columns=['DE3','DE10','SMOTUNED','Swift-\u03B5',], index=["chromium","Wicket","Ambari","Camel","Derby"])

df.plot(kind='bar')
# Turn on the grid
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()