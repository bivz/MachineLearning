function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Solution Part 1
 a_1 = [ones(rows(X), 1) X];
 z_2 = a_1 * Theta1';
 a_2 = sigmoid(z_2);
 a_2 = [ones(rows(a_2), 1) a_2];
 z_3 = a_2 * Theta2';
 a_3 = sigmoid(z_3);
 h_x = a_3;
 
 % coding y to Y 
 I = eye(num_labels);
 Y = zeros(m, num_labels);
 for i = 1:m
   Y(i, :) = I(y(i), :);
 endfor


% J = sum(sum((-Y).*log(h_x) - (1-Y).*log(1-h_x), 2))/m;

reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); % Making sure we done include bias

J = (1/m) * sum(sum((-Y).*log(h_x)-(1-Y).*log(1-h_x)))+reg;


% Part 2 Backpropagation algorithm
%Step 1 values of a_x and z_x already calculated above

%Step 2
del_3 = a_3 .- Y;

% Step 3
del_2 = (del_3 * Theta2) .* sigmoidGradient([ones(size(z_2, 1), 1) z_2]);
del_2 = del_2(:, 2:end);  % Removing Bias terms

% Step 4
DELTA_1 = del_2' * a_1;
DELTA_2 = del_3' * a_2;

%Step 5
% Calculating Regularized terms. 
T_1 = [zeros(rows(Theta1), 1) Theta1(:, (2:end))];  % Removing the first column for reg comutationa as its not added for j =0
T_2 = [zeros(rows(Theta2), 1) Theta2(:, (2:end))];  % Removing the first column for reg comutationa as its not added for j =0

P1 = (lambda/m)* T_1;  % Penalty terms
P2 = (lambda/m)* T_2;

Theta1_grad = DELTA_1./m + P1;
Theta2_grad = DELTA_2./m + P2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
