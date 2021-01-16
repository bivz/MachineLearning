%% Initialization
clear ; close all; clc;
addpath('./bivfiles');
file_contents = [];

fprintf('\nPreprocessing sample emails (Email(x).txt)\n');
##
##
##for i = 1 : 2796
##  %%num2str (i)
##fprintf('Processing Email: ');
##  name = ["Email_(" num2str(i) ")" ".txt"]
##  file_contents = readFile(name);
##  word_indices  = processEmail(file_contents);
##  features(i,:)   = emailFeatures(word_indices);
##  % Print Stats
##  
##%%fprintf('Length of feature vector: %d\n', length(features(i,:)));
##%%fprintf('Number of non-zero entries: %d\n', sum(features(i,:) > 0));
##%%fprintf('Program paused. Press enter to continue.\n');
##%%pause;
##endfor

%%% Dataset received now, divided into training and test sets
%% =========== Part : Train Linear SVM for Spam Classification ========
%  In this section, you will train a linear classifier to determine if an
%  email is Spam or Not-Spam.
load('biv_train.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(X_train, Y_train, C, @linearKernel);

p = svmPredict(model, X_train);

fprintf('Training Accuracy: %f\n', mean(double(p == Y_train)) * 100);

%% =================== Part 4: Test Spam Classification ================
%  After training the classifier, we can evaluate it on a test set. We have
%  included a test set in spamTest.mat

% Load the test dataset
% You will have Xtest, ytest in your environment
##load('biv_test.mat');
##
##fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')
##
##p = svmPredict(model, X_test);
##
##fprintf('Test Accuracy: %f\n', mean(double(p == Y_test)) * 100);
##pause;

%% =================== Part 6: Try Your Own Emails =====================
%  Now that you've trained the spam classifier, you can use it on your own
%  emails! In the starter code, we have included spamSample1.txt,
%  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
%  The following code reads in one of these emails and then uses your 
%  learned SVM classifier to determine whether the email is Spam or 
%  Not Spam

% Set the file to be read in (change this to spamSample2.txt,
% emailSample1.txt or emailSample2.txt to see different predictions on
% different emails types). Try your own emails as well!
filename = 'TestMail.txt';

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');


