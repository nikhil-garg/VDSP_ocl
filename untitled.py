np.savez('qmnist_test.npz',image_test_filtered=image_test_filtered,label_test_filtered=label_test_filtered)

np.save('fnist.npz',image_train_filtered=image_train_filtered,label_train_filtered=label_train_filtered,image_test_filtered=image_test_filtered,label_test_filtered=label_test_filtered)


data = np.load('fmnist.npz', allow_pickle=True)

image_train_filtered = data['image_train_filtered']
label_train_filtered = data['label_train_filtered']
image_test_filtered = data['image_test_filtered']
label_test_filtered = data['label_test_filtered']


data = np.load('qmnist_test.npz', allow_pickle=True)
image_test_filtered = data['image_test_filtered']
label_test_filtered = data['label_test_filtered']

mnist_trainset = datasets.QMNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.QMNIST(root='./data', train=False, download=True, transform=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=60000, shuffle=True)