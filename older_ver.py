import numpy as np
import random
from matplotlib.pyplot import plot as plt

class NeuralLayer():
    def __init__(self, in_size, out_size, weights=None, bias=None):
        self.last_in = None
        
        self.weight = weights if weights is not None else np.random.rand(out_size, in_size) * 0.5 + 0.5 

        self.bias = bias if bias is not None else np.zeros((out_size))
        
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
        g_out: (out,)

        '''
        # (out, in)
        self.grad['in']['weight'] = np.outer(self.grad['out'], self.last_in)
        # (out)
        self.grad['in']['bias'] =  self.grad['out']
        # (out,)
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
        self.last_in = x
        return 1/(1 + np.exp(-x))
    
    def backward(self):
        x = self.last_in
        self.grad['in'] = np.exp(-x) / (1 + np.exp(-x))**2 * self.grad['out']
        # multiply the derivative of this output with respect to its
        # input and the derivative of the loss with respect to
        # this output
        propagate_grad = self.grad['in']
        return propagate_grad
             
class Loss():
    def __init__(self, e=1e-10):
        # temp variables to store data for backprop
        self.last_in = None
        self.last_label = None
        self.grad = {
            'in' : 0,
        }
        self.e = e
    
    def __call__(self, y_pred, y_label):
        # last sample label temp variable
        self.last_label = y_label
        # last prediction temp variable
        self.last_in = y_pred
        # calculate binary cross entropy loss
        bce = -(y_label*np.log(y_pred) + (1-y_label)*np.log(self.e + 1-y_pred))
        return bce
    
    def backward(self):
        # get scalar value from vector of size 1
        pred = self.last_in.item()
        # redefine for ease
        label = self.last_label
        # calculate gradient of loss with respect to
        # prediction
        self.grad['in'] = (-label/(self.e + pred) + (1-label)/(self.e + 1-pred))
        return self.grad['in']
            
if __name__ == "__main__":
    
    e = 1e-5
    lr = 1

    in_layer = NeuralLayer(2, 2)
    sigmoid1 = Sigmoid()
    h_layer = NeuralLayer(2, 1)
    sigmoid2 = Sigmoid()
    loss = Loss(e=e)
    
    def update_params():
        return [
            [in_layer.weight, in_layer.grad['in']['weight']],
            [in_layer.bias, in_layer.grad['in']['bias']],
            [h_layer.weight, h_layer.grad['in']['weight']],
            [h_layer.bias, h_layer.grad['in']['bias']],
            ]
        
    samples = [
            ([0, 0], 0),
            ([1, 0], 1),
            ([0, 1], 1),
            ([1, 1], 0),
        ]
    
    errors = [0, 0, 0, 0]
    
    loss_accum = 0
    losses = []
    
    for i in range(4000):
        sample = random.choice(samples)
        data = sample[0]
        label = sample[1]
        
        data = in_layer(data)
        data = sigmoid1(data)
        data = h_layer(data)
        pred = sigmoid2(data)
        
        loss_value = loss(pred, label)
        losses.append(loss_value)
        loss_accum += loss_value
        
        error = abs(label - pred)
        
        sigmoid2.grad['out'] = loss.backward()
        h_layer.grad['out'] = sigmoid2.backward()
        sigmoid1.grad['out'] = h_layer.backward()
        in_layer.grad['out'] = sigmoid1.backward()
        in_layer.backward()

        params = update_params()
        
        updates = [(grad, param) for param, grad in params]
        
        # Update all params at once
        for grad, param in updates:
            param -= lr * grad
        
        print('loss', loss_value)
        print('error', error)
        print()
    print(f'avg_loss {loss_accum/3000}')
    
    for sample in samples:
        data = sample[0]
        data = in_layer(data)
        data = sigmoid1(data)
        data = h_layer(data)
        pred = sigmoid2(data)
        print(f'pred: {pred.item():.3f}, label: {sample[1]}')
        print(f'sample: {sample}')
    
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # lr = 1e-5
    # steps = 10000
    # checkpoint = 50
    # e = 1e-10
    
    # hidden_sizes = [8]
    
    # avg_loss = 0
    
    # total_dist = 0
    
    # xor = XORClassifier(hidden_sizes=hidden_sizes, e=e)

    
    # for step in range(steps):
    #     sample = random.choice(xor_samples)
    #     data = sample[0]
    #     label = sample[1]
    #     pred, loss = xor(data, label)
    #     xor.backward()
        
    #     # Collect all gradients first
    #     updates = [(grad, param) for param, grad in xor.params]
        
    #     # Update all params at once
    #     for grad, param in updates:
    #         param -= lr * grad
                
        
    #     avg_loss += loss
        
    #     total_dist += abs(pred - label)
        
    #     if step%checkpoint == 0:
    #         print()
    #         print('step: ', step) 
    #         print('loss: ',loss)
    #         print(f'pred: {pred.item():.1f} label: {label}')
    #         print(f"sample: {sample}")
        
    #     for i, pair in enumerate(xor.params):
    #         grad = pair[1]
    #         param = pair[0]
    #         pair[0] -= lr * pair[1]
    # print()
    # print('stats')
    # print(f'avg_loss: {avg_loss.item()/steps:.3f}, average_dist: {total_dist/steps}')
        