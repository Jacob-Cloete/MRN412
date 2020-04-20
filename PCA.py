%notebook inline
X1 = time_series
fault_colors = (1)
import sklearn.decomposition as skdecomp
from matplotlib import pyplot as plt
pca = skdecomp.PCA(n_components=2)

X = np.vstack([X1])
index = np.hstack([np.ones(X1.shape[0])*0])

Y = pca.fit_transform(X)

print(index.shape)
print(Y.shape)
plt.figure(figsize=(12,8))
plt.title("Time Series Length {}, PCA Feature Space = {}".format(X.shape[1],Y.shape[1]),size=25)
plt.scatter(Y[:,0],Y[:,1],c=Class,s=8)
plt.ylim(-4,6)
plt.xlim(-25,50)
#plt.ylim(-20,20)
#plt.xlim(-20,17)
plt.ylabel("PCA 2",size=20)
plt.xlabel("PCA 1",size=20)
unique_classes = list(set(Class))
cmap = plt.cm.get_cmap("viridis",13)
Faults = ['BASE','D1A','D1B','D2A','D2B','D3A','D3B','D4A','D4B','D5A','D5B','D6A','D6B']
the_table = plt.table(cellText=[[x] for x in Faults], loc='lower right',
          colWidths=[0.05],colLabels=['FAULT'],rowColours=cmap(np.array(unique_classes)/13))
the_table.auto_set_font_size(False)
the_table.set_fontsize(14)
the_table.scale(1.7, 1.7)
plt.show()
    
print("We can automatically separate the features with PCA!")
