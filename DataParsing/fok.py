import matplotlib.pyplot as plt
import  numpy as np
import scipy.optimize as sp
x = np.linspace(0,100,50)
mu = 50  # mean of distribution
sigma = 20
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (x - mu))**2))
y2 = np.abs(np.random.randn(len(x)))
y2 = max(y)/y2
fig,ax = plt.subplots()
ax.bar(x,y,width=1)
ax.bar(x,y2,width=1,color="purple")
ax.plot(x,y,"--",color = "red")
plt.show()