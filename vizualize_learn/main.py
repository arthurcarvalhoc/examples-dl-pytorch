import sklearn.datasets
import matplotlib.pyplot as plt
import torch
from FeedForward import FeedForward

x,y = sklearn.datasets.make_moons(200, noise=0.20)
print( x[0])
print( x[0][0])
print( x[0][1])
plt.scatter(x[:,0], x[:,1],s=40, c=y, cmap=plt.cm.PuOr)
plt.show()


x = torch.FloatTensor(x)
y = torch.LongTensor(y)

network = FeedForward(input_neurons=2, hidden_neurons=50, output_neurons=2)
optmizer = torch.optim.Adam(network.parameters())
loss_function = torch.nn.CrossEntropyLoss()

#plt.ion
for epoch in range(10000):
    out = network(x)
    loss = loss_function(out, y)
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()

    if epoch % 9000 == 0:
        max_value, prediction = torch.max( out, 1)
        predicted_y = prediction.data.numpy()
        target_y = y.data.numpy()
        #plt.scatter( x.data.numpy()[:,0], x.data.numpy()[:,1], s = 40, c = predicted_y, lw = 0 )
        accuracy = (predicted_y == target_y ).sum() / target_y.size
        print( predicted_y.shape )
        print( (predicted_y == target_y ).sum() )
        print( (predicted_y == target_y ) )
        print(  "Accuracy is {:.10f}".format(accuracy) ) 
        #plt.text( 3, -1, "Accuracy is {:.2f}".format(accuracy), fontdict=14)
        #plt.pause(0.1)
#plt.ioff
#plt.show



