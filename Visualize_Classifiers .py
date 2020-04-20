%notebook inline
h=0.01
plt.figure(figsize=(8,6))
plt.ylim(-0.75,0.75)
plt.xlim(-1.8,2.5)
a = np.arange(-1.8, 2.5, h)
b = np.arange(-0.75, 0.75, h)
xx, yy = np.meshgrid(a,b)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.title("AdaBoost",size=20)
#plt.scatter(2,2,c='r')

#Add table
unique_classes = list(set(y_train))
cmap = plt.cm.get_cmap("viridis",14)
Faults = ['BASE','D1A','D1B','D2A','D2B','D3A','D3B','D4A','D4B','D5A','D5B','D6A','D6B','Noise']
the_table = plt.table(cellText=[[x] for x in Faults], bbox = [1.2, 0.1, 0.1, 0.8],
          colWidths=[0.05],colLabels=['FAULT'],rowColours=cmap(np.array(unique_classes)/14))
the_table.auto_set_font_size(False)
the_table.set_fontsize(14)
the_table.scale(1.7, 1.7)

plt.show()
