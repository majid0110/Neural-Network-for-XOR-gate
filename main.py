import math
import random


class NeuralNetwork:
    def __init__(self):
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)
        self.b1 = random.uniform(-1, 1)
        self.w3 = random.uniform(-1, 1)
        self.w4 = random.uniform(-1, 1)
        self.b2 = random.uniform(-1, 1)
        self.w5 = random.uniform(-1, 1)
        self.w6 = random.uniform(-1, 1)
        self.b3 = random.uniform(-1, 1)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def derivative_sigmoid(self, x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    def feedforward(self, x):
        h1 = self.sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = self.sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = self.sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, labels, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(data, labels):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = self.sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = self.sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = self.sigmoid(sum_o1)
                y_pred = o1

                loss = (y_true - y_pred) ** 2
                total_loss += loss

                d_L_d_ypred = -2 * (y_true - y_pred)

                d_ypred_d_w5 = h1 * self.derivative_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * self.derivative_sigmoid(sum_o1)
                d_ypred_d_b3 = self.derivative_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * self.derivative_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * self.derivative_sigmoid(sum_o1)

                d_h1_d_w1 = x[0] * self.derivative_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * self.derivative_sigmoid(sum_h1)
                d_h1_d_b1 = self.derivative_sigmoid(sum_h1)

                d_h2_d_w3 = x[0] * self.derivative_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * self.derivative_sigmoid(sum_h2)
                d_h2_d_b2 = self.derivative_sigmoid(sum_h2)

                self.w1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.w3 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                self.w5 -= learning_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learning_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learning_rate * d_L_d_ypred * d_ypred_d_b3
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(data)}')
network = NeuralNetwork()

data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 1, 1, 0]

network.train(data, labels, epochs=10000, learning_rate=0.1)
print('Deep Learning Assigmnet 1')
print('Submitted by Majid khan')
input1 = int(input("Enter the first binary input: "))
input2 = int(input("Enter the second binary input: "))

print(f"XOR({input1}, {input2}) = {round(network.feedforward([input1, input2]))}")
