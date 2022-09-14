import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
import get_prototype as gp
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
plt.style.use('ggplot')

data = make_moons(n_samples=400 ,noise=0.15, random_state=100, shuffle=True)
cmap = plt.cm.Spectral
for i in range(cmap.N):
   rgba = cmap(i)
   print("Hexadecimal representation of rgba:{} is {}".format(rgba, matplotlib.colors.rgb2hex(rgba)))
X, y = data
color_map = plt.cm.get_cmap('viridis')
x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.3)
proto = gp.getpro (x_train, y_train)
ps_prototypes = proto.prototypes
ps_prototype_label = proto.prototype_label
zip_proto = np.column_stack ((ps_prototypes, ps_prototype_label))

#figure, axes = plt.subplots()
fig = plt.figure(figsize=(5, 3))
axes = fig.add_subplot()
axes.axis([-2.29,3.4,-1.82,2.25])
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors="k", cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(ps_prototypes[:, 0], ps_prototypes[:, 1], c=ps_prototype_label, cmap=plt.cm.Spectral, marker='X', s=150, edgecolors="k")
prototypes_l0, prototypes_l1 = [],[]
for i, j in zip(ps_prototypes, ps_prototype_label):
    if j == 0:
        draw_circle = plt.Circle((i[0], i[1]), proto.epsilon_, fill=False, edgecolor='#9e0142', linewidth=0.8)
        axes.set_aspect(1)
        axes.add_artist(draw_circle)
    else:
        draw_circle1 = plt.Circle ((i[0], i[1]), proto.epsilon_, fill=False, edgecolor='#5b53a4', linewidth=0.8)
        axes.set_aspect (1)
        axes.add_artist (draw_circle1)
#plt.tight_layout()
plt.title('Epsilon =' + str(proto.epsilon_))
axes.xaxis.set_visible (False)
axes.yaxis.set_visible (False)
plt.savefig('EpsilonBall1.png', dpi=200)
plt.show()