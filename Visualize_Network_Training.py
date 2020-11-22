from matplotlib import pyplot as plt

N = np.arange(0, nb_epochs)[:]
plt.style.use("ggplot")
plt.figure()
plt.figure(figsize=(15,8))
plt.plot(N, hist.history['val_accuracy'][:], label="testing_accuracy")
plt.plot(N, hist.history["val_loss"][:], label="testing_loss")
plt.plot(N, hist.history["loss"][:], label="train_loss")
plt.plot(N, hist.history["accuracy"][:], label="train_accuracy")
plt.title("Training Loss and Accuracy of testing data",size=20)
plt.xlabel("Epoch #",size=18)
plt.ylabel("Loss/Accuracy",size=18)
plt.legend(loc=1, prop={'size': 15})
plt.ylim(0,1.2)

plt.show()
