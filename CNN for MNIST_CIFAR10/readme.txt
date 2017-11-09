HOW TO RUN THE PROGRAM:

MNIST dataset:
1.	Make sure that the file 'mnist.pkl.gz' is in the directory of the scripts.
	You can download it from: http://deeplearning.net/data/mnist/mnist.pkl.gz

2.	Run 'python <path>/le_net.py mnist' from the command line, <path> is the path to the folder that contains the scripts

CIFAR-10 dataset:
1.	Download and extract in the directory of the scripts the CIFAR-10 dataset for python
	You can dowload it from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

2.	Run 'python <path>/le_net.py cifar10' from the command line, <path> is the path to the folder that contains the scripts
	You can pass a third argument to specify the number of image classes, e.g., 'python <path>/le_net.py cifar10 5' will use
	just 5 image classes


NOTES

1.	Layers used:
	1st: Convolutional layer with 20 (5, 5) kernels and (2,2) max pooling, ReLu activation
	2nd: Convolutional layer with 50 (5, 5) kernels and (2,2) max pooling, ReLu activation
	3rd: Convolutional layer with 50 (3, 3) kernels, (1,1) zero padding and no max pooling, ReLu activation
	4rd: Fully connected layer with 500 neurons, ReLu activation
	On top of that I use a LogisticRegression layer with cross-entropy loss

2.	I used "He initialization" for the Convolutional Layers with ReLu activation as suggest in the paper that can be found at
	https://arxiv.org/pdf/1502.01852v1.pdf.
	It is also possible to use tanh or sigmoid activation function by changing the code of "le_net.py" accordingly, in this case
	the initializiation is as suggested in the paper that can be founda at:
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.207.2059&rep=rep1&type=pdf

3.	For the third convolutional layer I used a (3, 3) filter with (1, 1) zero padding to keep the size of each feature map

4.	By default I do not use any regularization, but my code allows its usage by passing the L1_reg and L2_reg arguments to the function
	train_and_predict.

4.	I used SGD with Nesterov momentum update, but I also implement standard SGD and standard SGD with momentum in the script
	'updates.py'. Feel free to use the one you prefer by changing the code of 'le_net.py' accordingly.

5.	For data augmentation, I thought that horizontal/vertical flipping would be a good idea.

6.	Running the code on a CPU with batch_size=100, n_epochs=200 and no regularization I achieved ~1% error on MNIST dataset 


