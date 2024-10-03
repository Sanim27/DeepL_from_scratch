import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_mnist_as_h5():
    # Load the MNIST dataset from the local directory
    path = '/Users/sanimpandey/.keras/datasets/mnist.npz'
    with np.load(path) as mnist_data:
        # Extract training and test sets
        x_train = mnist_data['x_train']
        y_train = mnist_data['y_train']
        x_test = mnist_data['x_test']
        y_test = mnist_data['y_test']

    # Save MNIST dataset in .h5 format for easier loading like catvnoncat
    with h5py.File('mnist_dataset.h5', 'w') as h5f:
        h5f.create_dataset('train_set_x', data=x_train)
        h5f.create_dataset('train_set_y', data=y_train)
        h5f.create_dataset('test_set_x', data=x_test)
        h5f.create_dataset('test_set_y', data=y_test)
        h5f.create_dataset('list_classes', data=np.arange(10))  # Digits 0-9 as classes

def load_dataset():
    # Load the saved .h5 MNIST dataset
    dataset_path = 'mnist_dataset.h5'
    with h5py.File(dataset_path, 'r') as dataset:
        train_set_x_orig = np.array(dataset['train_set_x'][:])  # Training features
        train_set_y_orig = np.array(dataset['train_set_y'][:])  # Training labels
        test_set_x_orig = np.array(dataset['test_set_x'][:])    # Test features
        test_set_y_orig = np.array(dataset['test_set_y'][:])    # Test labels
        classes = np.array(dataset['list_classes'][:])          # List of classes (0-9)

    # Reshape the labels to match the format used earlier
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Create the .h5 MNIST dataset file (only needs to be run once)
load_mnist_as_h5()

# Load the data
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

print("Training set shape:", train_set_x_orig.shape)
print("Training labels shape:", train_set_y_orig.shape)
print("Test set shape:", test_set_x_orig.shape)
print("Test labels shape:", test_set_y_orig.shape)
print("Classes:", classes)

def initialize_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Subtract max for numerical stability , to make the highest exponential value as 0 and all others less than 0.
    return expZ / np.sum(expZ, axis=0, keepdims=True), Z

def relu(Z):
  return np.maximum(0, Z),Z

def linear_forward(A,W,b):
  Z=np.dot(W,A)+b
  cache=(A,W,b)
  return Z,cache

def linear_activation_forward(A_prev, W, b, activation):
  if activation=="softmax":
    Z,linear_cache=linear_forward(A_prev,W,b)
    A,activation_cache=softmax(Z)
  elif activation=="relu":
    Z,linear_cache=linear_forward(A_prev,W,b)
    A,activation_cache=relu(Z)
  cache=(linear_cache,activation_cache)
  return A,cache

def L_model_forward(parameters,X):
  caches=[]
  A=X
  L=len(parameters)//2   #// is used to ensure L is integer as we have to loop through it.
  for l in range(1,L):
    A_prev=A
    A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
    caches.append(cache)

  AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"softmax")
  caches.append(cache)
  return AL,caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    # Use categorical cross-entropy for multi-class classification
    cost = -np.sum(Y * np.log(AL)) / m
    cost = np.squeeze(cost)  # Ensures cost is a scalar
    return cost

def linear_backward(dZ,cache):
  A_prev,W,b=cache
  m=A_prev.shape[1]
  dW=(1/m)*np.dot(dZ,A_prev.T)
  db=(1/m)*np.sum(dZ,axis=1,keepdims=True)
  dA_prev=np.dot(W.T,dZ)

  return dA_prev,dW,db


def one_hot_encode(Y, num_classes=10):
    one_hot_Y = np.eye(num_classes)[Y.reshape(-1)].T   #This is not dot product/multiplication. But it is advanced numpy indexing where np.eye(10) makes an identity matrix of size(10,10) and suppose the first element is 3 then first row of result before transposing would be [0001000000]
    return one_hot_Y

train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2],train_set_x_orig.shape[0])
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[1]*test_set_x_orig.shape[2],test_set_x_orig.shape[0])

train_set_y = one_hot_encode(train_set_y_orig, 10)
test_set_y = one_hot_encode(test_set_y_orig, 10)
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

def softmax_backward(dA, activation_cache):
    Z = activation_cache
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Numerically stable softmax
    A = expZ / np.sum(expZ, axis=0, keepdims=True)  # A[l] is the output of softmax
    
    # Instead of using a loop, compute the gradient for the softmax function directly
    dZ = A * (dA - np.sum(dA * A, axis=0, keepdims=True))  # Element-wise gradient
    return dZ

def relu_backward(dA,activation_cache):
  Z=activation_cache
  return dA*(Z>0)

def linear_activation_backward(dA,cache,activation):
  linear_cache,activation_cache=cache
  if activation=="softmax":
    dZ=softmax_backward(dA,activation_cache)
  elif activation=="relu":
    dZ=relu_backward(dA,activation_cache)
  dA_prev,dW,db=linear_backward(dZ,linear_cache)
  return dA_prev,dW,db

def L_model_backward(AL,Y,caches):
  grads={}
  m=AL.shape[1]
  L=len(caches)   
  dAL=-Y/AL
  current_cache=caches[L-1]    #cache for layer L-1
  grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(dAL,current_cache,"softmax")  #dA4,dW5,db5
  for l in reversed(range(L-1)):
    current_cache=caches[l]
    grads["dA"+str(l)],grads["dW"+str(l+1)],grads["db"+str(l+1)]=linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu") #dA3,dW4,db4
  
  return grads

def update_parameters(parameters,grads,learning_rate):
  L=len(parameters)//2
  for l in range(L):
    parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
    parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]
  return parameters


def L_layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
  parameters=initialize_deep(layer_dims)
  costs=[]
  m=X.shape[1]
  for i in range(0,num_iterations):
    AL,caches=L_model_forward(parameters,X)
    cost=compute_cost(AL,Y)
    grads=L_model_backward(AL,Y,caches)
    parameters = update_parameters(parameters, grads, learning_rate)
    if print_cost and i % 100 == 0:
      print ("Cost after iteration %i: %f" %(i, cost))
    if i % 100 == 0:
      costs.append(cost)
  plt.plot(np.squeeze(costs))
  plt.ylabel('cost')
  plt.xlabel('iterations (per hundreds)')
  plt.title("Learning rate =" + str(learning_rate))
  plt.show()
    
  return parameters

layer_dims = [784, 20, 7, 5, 10]

para = L_layer_model(train_set_x, train_set_y, layer_dims, num_iterations = 3000, print_cost = True)

def predict(X, parameters):
    AL, _ = L_model_forward(parameters, X)
    predictions = np.argmax(AL, axis=0)  # Class with the highest probability
    return predictions

predictions_train=predict(train_set_x,para)
predictions_test=predict(test_set_x,para)

train_accuracy = np.mean(predictions_train == np.argmax(train_set_y, axis=0)) * 100
test_accuracy = np.mean(predictions_test == np.argmax(test_set_y, axis=0)) * 100

print(f"training accuracy is {train_accuracy}")
print(f"test accuracy is {test_accuracy}")
