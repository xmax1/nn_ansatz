# implicit none
# integer ix,iy,n
# real x,y,r,xp,yp,rp,eps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eps = 0.00001
def eq100(x):
    if(x < -0.25):
        xp=-1/(8*(1+2*x))
    elif(x < 0.25):
        xp=x
    else:
        xp=1/(8*(1-2*x))
    return xp

Xs = np.linspace(-0.5, 0.5, 10) + eps
Ys = Xs

datas = []
for x in Xs:
    xp = eq100(x)
    for y in Ys:
        yp = eq100(y)

        r = np.linalg.norm(np.array([x, y]))
        rp = np.linalg.norm(np.array([xp, yp]))

        data = [np.squeeze(i) for i in [x, y, xp, yp, r, np.exp(-3*r), np.exp(-3*rp)]]
        datas.append(data)


data = pd.DataFrame(datas, columns=['x', 'y', 'xp', 'yp', 'r', 'exp_r', 'exp_rp'])

plot2d_data = data.pivot(index='x', columns='y', values='exp_rp')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

print(plot2d_data)

axs[0].contourf(plot2d_data)

plt.show()
