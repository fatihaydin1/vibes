%% Authors: Fatih Aydin & Zafer Aslan
%% This program is a demo for the vibes algorithm
clear
clc
close all


%% Example 1
load Cardiotocography3.mat
% The base learners you can select are: knn, nb, svm, disc, tree, rf, nn.
% You should modify the functions regarding the base learners in
% order to set their parameters.
TrainingOptions.baseLearner = 'knn';
% isDependent : true, false
TrainingOptions.isDependent = true;
% searchMethod : OFS (Optimized Forward Search), GA (Genetic Algorithm)
TrainingOptions.searchMethod = 'OFS';
% featureRank : IG (Information Gain), RF (ReliefF)
TrainingOptions.featureRank = 'IG';
TrainingOptions.fold = 10;
[testIndices1, trainIndices1] = split(Cardiotocography3, 0.1);
% Construct a model or hypothesis by means of cross validation method
[model1] = vibes(Cardiotocography3(trainIndices1,:), TrainingOptions);
% The last argument is for a test set. And, the experiment is more
% meaningful and reasonable if the training set and the test set are not
% the same and all the data are not used for just the training.
[outcome1Test] = vibes(model1, Cardiotocography3(testIndices1,:));


%% Example 2
load BreastCancer.mat
TrainingOptions2.baseLearner = 'nb';
TrainingOptions2.isDependent = true;
TrainingOptions2.searchMethod = 'OFS';
TrainingOptions2.featureRank = 'RF';
TrainingOptions2.K = 2;
TrainingOptions2.fold = 10;
[model2] = vibes(BreastCancer, TrainingOptions2);


%% Example 3
load LetterRecognition.mat
TrainingOptions3.baseLearner = 'tree';
TrainingOptions3.isDependent = true;
TrainingOptions3.searchMethod = 'OFS';
TrainingOptions3.featureRank = 'IG';
TrainingOptions3.fold = 5;
[model3] = vibes(LetterRecognition, TrainingOptions3);


%% Example 4
load VertebralColumn.mat
TrainingOptions4.baseLearner = 'nn';
TrainingOptions4.isDependent = true;
TrainingOptions4.searchMethod = 'OFS';
TrainingOptions4.featureRank = 'IG';
% All the data are allocated for just the training if the below value is not
% set, otherwise, the data are divided into three groups such as
% training, validation, and test. The ratios of the division are 70%, 15%,
% and 15% respectively.
TrainingOptions4.trainRatio = 70;
TrainingOptions4.valRatio = 15;
TrainingOptions4.testRatio = 15;
[testIndices4, trainIndices4] = split(VertebralColumn, 0.1);
[model4] = vibes(VertebralColumn(trainIndices4,:), TrainingOptions4);
[outcome4Test] = vibes(model4, VertebralColumn(testIndices4,:));


%% Example 5
load glass.mat
TrainingOptions5.baseLearner = 'svm';
TrainingOptions5.isDependent = false;
TrainingOptions5.searchMethod = 'OFS';
TrainingOptions5.featureRank = 'IG';
[model5] = vibes(glass, TrainingOptions5);
