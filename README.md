## Steps:

Created Class names as 'NeuralNetwork' and initialized the weights

defined an activation function i.e sigmoid function i.e σ(x) = 1 / (1 + e ^ -x)

Then I used derivative of Sigmoid function that is neccessory for backword propegation for reducing error loss. i.e dy/dx (σ(x)) => σ ′(x)=σ(x)⋅(1−σ(x))

Then I defined another function for feed farward, that will calculating the outputs of the neurons and theri outputs. as a total I have 5 neurons . 2 for inputs, 2 hidden and 1 output neuron.

General Formuls = x' = W1.input1 + W2.input2 + Bias

Then I put the value of x' in the sigmoid function for output of hidden leyer i.e h = σ(x') = 1 / (1 + e ^ -x`)
For out put, I considered values of hidden neuron and and applied same technique on it, applied same technique, i.e multiplied each by W and added bias to it to get my resulted output .
I used MSE (Mean Square Error) to calculate loss function i.e MSE = sum (actual -predicted)^2
Now I started training my model by declearing a function for it as train it took data (dataset; in XOR case we have 4 possible combinations => [ [0, 0], [0, 1], [1, 0], [1, 1] ] ), their labels (supervised Leaning, In XOR case [0, 1, 1, 0] ), epochos i.e the number of time I want to pass my data from model, learning rate -> any small value in my case I take it as 0.1.

for backword propegation weights are updated, i.e W`= W − η⋅ ∂L/∂y(predictd) . ∂y(predictd)/ ∂h1 . ∂h1/∂W

Model is trained and result will be assigned to a veriable network.
Now using above defined fnction, I will train my model on XOR data, passing other parameters as well, I chooose epoch as 1000 rounds for better trained and learing rate as 0.1 for fast learning (generally we take it as 0.01).

I started with error : 0.0009426991191646666 in first stage, which keep on reducing by updating weights through backword propegation.
Final Error: 0.0008247080119330441
improvement: ~12%
