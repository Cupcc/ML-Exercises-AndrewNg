import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x = np.linspace(0,1,100)
y = [min(0,i*i+i-1) for i in x]
index=0
for i in x:
    if(y[index]==0):
        x_table=i
        break
    index+=1

plt.figure(figsize=(15,8),dpi=100)
plt.xlabel('x')
plt.xticks(np.linspace(min(x),max(x),21))
plt.ylabel('y')
plt.yticks(np.linspace(min(y),max(y),21))
plt.text(x_table,0,(x_table,0))
plt.grid(1)
plt.plot(x,y)

plt.show()