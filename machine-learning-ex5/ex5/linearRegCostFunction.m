function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

penalty = (lambda/(2*m)) * (sum(theta.^2)- theta(1).^2);  % Excluding the theta(1) term 
h_x = (X * theta); 
J = (1/(2*m)) * sum((h_x - y).^2) + penalty;  % Cost Function Calculation

% Calculating Regularized Linear Regression Gradient 
penalty_grad = (lambda/m) * theta;
penalty_grad(1) = 0;

grad = (1/m)* (X' * (h_x - y)) + penalty_grad;




% =========================================================================

grad = grad(:);

end
