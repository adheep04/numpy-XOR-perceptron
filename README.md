![image](https://github.com/user-attachments/assets/adeda2ec-8db0-4dc6-a33c-b74de5bdac7f)

classifies the following boolean expressions into either (1) -> XOR expression, (0) -> NOT XOR expression:
1) (0, 0) -> 0
2) (1, 0) -> 1
3) (0, 1) -> 1
4) (1, 1) -> 1

this is a linearly inseperable problem which makes a hidden layer and non-linear activation functions (like sigmoid) necessary to classify. The source code only uses numpy and matplotlib for the graph you see above. All backpropagation logic written from scratch.



before manual gradient descent:

![image](https://github.com/user-attachments/assets/590e11be-331c-43c9-aa23-e4ef236e9c80)



after:

![image](https://github.com/user-attachments/assets/488211cb-f3bb-4bb5-8619-87670aa3d080)
