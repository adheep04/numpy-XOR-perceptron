
import numpy as np
import random
import matplotlib.pyplot as plt

class NeuralLayer():
    def __init__(self, in_size, out_size):
        # attribute to hold the last passed in input for backprop purposes
        self.last_in = None
        
        # initialize weights so they are in range (0.5, 1)
        self.weight = np.random.rand(out_size, in_size) * 0.5 + 0.5
        self.bias = np.zeros((out_size))
        
        # contains gradients during backward pass
        # updated during backward
        self.grad = {
            # gradient of the loss  with respect
            # to this layer's input components
            'in' : {
                # (out, in)
                'weight' : None,
                # (out)
                'bias' : None,
            },
            # gradient of the loss with respect to this
            # layer's output
            'out' : None,  
        }
    
    def __call__(self, x):
        self.last_in = x
        return self.weight @ x + self.bias
    
    def backward(self):
        '''
        x: (in,)
        w: (out,in)
        b: (in,)
        propagate_grad: (out,)

        '''
        # gets the gradient of the weight with respect to the loss, shape: (out, in)
        self.grad['in']['weight'] = np.outer(self.grad['out'], self.last_in)
        # gets gradient of bias (out)
        self.grad['in']['bias'] =  self.grad['out']
        # value of gradient for the next layer in backward pass 
        propagate_grad = self.grad['out'] @ self.weight
        return propagate_grad
        
class Sigmoid():
    def __init__(self):
        self.last_in = None
        self.grad = {
            'in' : 0,
            'out' : 0
        }
    
    def __call__(self, x):
        # cache last input
        self.last_in = x
        # return sigmoid 
        return 1/(1 + np.exp(-x))
    
    def backward(self):
        x = self.last_in
        self.grad['in'] = np.exp(-x) / (1 + np.exp(-x))**2 * self.grad['out']
        # multiply the derivative of this output with respect to its
        # input and the derivative of the loss with respect to
        # this output
        propagate_grad = self.grad['in']
        return propagate_grad
             
class BinaryCrossEntropy():
    def __init__(self, e=1e-10,):
        # cache both last input and label fed into loss
        self.last_in = None
        self.last_label = None
        self.grad = {
            'in' : 0,
        }
        self.e = e
    
    def __call__(self, y_pred, y_label):
        # retrieve last label
        self.last_label = y_label
        # retrieve last prediction
        self.last_in = y_pred
        # calculate binary cross entropy loss
        return -(y_label*np.log(y_pred) + (1-y_label)*np.log(self.e + 1-y_pred))
    
    def backward(self):
        # get scalar value from vector of size 1
        pred = self.last_in.item()
        # redefine for ease
        label = self.last_label
        
        # calculate gradient of loss with respect to prediction
        # small epsilon e value for numerical stability (avoiding division by 0)
        self.grad['in'] = (-label/(self.e + pred) + (1-label)/(self.e + 1-pred))
        return self.grad['in']
           
if __name__ == "__main__":
    
    # learning rate (it's pretty high but it works!)
    lr = 0.5
    epsilon = 1e-10
    num_steps = 12000

    # initialize the layers
    in_layer = NeuralLayer(2, 2)
    sigmoid1 = Sigmoid()
    h_layer = NeuralLayer(2, 1)
    sigmoid2 = Sigmoid()
    loss = BinaryCrossEntropy(e=epsilon)
    
    # get model parameters that have gradients
    def get_params():
        return [
            [in_layer.weight, in_layer.grad['in']['weight']],
            [in_layer.bias, in_layer.grad['in']['bias']],
            [h_layer.weight, h_layer.grad['in']['weight']],
            [h_layer.bias, h_layer.grad['in']['bias']],
            ]
    
    # all samples in XOR problem
    samples = [
            ([0, 0], 0),
            ([1, 0], 1),
            ([0, 1], 1),
            ([1, 1], 0),
        ]
    
    # accumulating loss to average when returning
    loss_accum = 0
    
    # collecting losses for plotting purposes
    losses = []
    
    # training loop
    for step in range(num_steps):
        
        # randomly select a sample from valid XOR samples
        sample = random.choice(samples)
        data = sample[0]
        label = sample[1]
        
        ''' forward pass '''
        data = in_layer(data)
        data = sigmoid1(data)
        data = h_layer(data)
        pred = sigmoid2(data)
        
        # calculate loss/error and update accumulators for final statistics
        loss_value = loss(pred, label)
        losses.append(loss_value)
        loss_accum += loss_value
        error = abs(label - pred)
        
        ''' backward pass '''
        sigmoid2.grad['out'] = loss.backward()
        h_layer.grad['out'] = sigmoid2.backward()
        sigmoid1.grad['out'] = h_layer.backward()
        in_layer.grad['out'] = sigmoid1.backward()
        in_layer.backward()

        # collect all learnable parameters (weights and biases on input and hidden layer)
        updates = [(grad, param) for param, grad in get_params()]
        
        # Update all params at once
        for grad, param in updates:
            param -= lr * grad
            
            
            
        ''' all code below this line is only for printing/displaying statistics '''
        
        
        if step%(num_steps//10) == 0:
            print(f'___step {step}___')
            print(f'loss {loss_value}')
            print(f'error {error}\n')
        
    print(f'avg_loss {loss_accum/num_steps}\n')
    
    print('__________FINAL PREDICTIONS________')
    for sample in samples:
        data = sample[0]
        data = in_layer(data)
        data = sigmoid1(data)
        data = h_layer(data)
        pred = sigmoid2(data)
        print()
        print(f'pred: {pred.item():.3f}, label: {sample[1]}')
        print(f'sample: {sample}')
    
    print()
    print(f'_______PARAMETER VALUES________\n')
    print(f"input layer weights:\n")
    print(f'{in_layer.weight}\n')
    print(f"input layer bias:\n")
    print(f'{in_layer.bias}\n')
    print(f"hidden layer weights:")
    print(f'{h_layer.weight}\n')
    print(f"hidden layer bias:\n")
    print(f'{h_layer.bias}\n')
    
    window = 100  
    losses = [loss.item() for loss in losses]
    rolling_mean = np.convolve(losses, np.ones(window)/window, mode='valid')
        
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(losses)), losses, alpha=0.3, label='Raw Loss')
    plt.plot(range(len(rolling_mean)), rolling_mean, 'r-', linewidth=2, label='Moving Average')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.title('Training Loss over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        

    