function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

options = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
results = eye(64,3);
errorRow = 0;

for i = 1:size(options,2)
  for j = 1:size(options,2)
    C = options(i); 
	sigma = options(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    new_error = mean(double(predictions ~= yval))
    errorRow=errorRow+1
    
    results(errorRow,:) = [C, sigma, new_error]; 
    
  end
end  

sorted_result = sortrows(results, 3); % find the min of error

C = sorted_result(1,1);
sigma = sorted_result(1,2);


% =========================================================================

end
