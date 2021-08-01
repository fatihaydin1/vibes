%% Authors: Fatih Aydin & Zafer Aslan
% The vibes algorithm was updated on 01.11.2020
% Version (1.1.6)
%% INPUTS:
% dataset      -   It is data used to create the ensemble. 
               %   The dataset contains the actual values.
               %   It is a table with size NxM,
               %   where N is the number of observations (rows) and
               %   M is the number of features (columns).
               %   the last one of features is the true labels of each
               %   observation.
        
% baseLearner  -   You can choice the six base learners which are k-Nearest
               %   Neighbour (knn), Naive Bayes (nb), Support Vector
               %   Machine (svm), Discriminant Analysis (disc), Decision
               %   Tree (tree), and Random Forest (rf).
               %   Its values are: knn, nb, svm, disc, tree, rf.
            
% isDependent  -   It is a boolean value regarding whether features are
               %   interdependent, i.e. dependent on each other, or
               %   inter-independent, i.e. independent of each other.
               
% searchMethod -   It is the method of searching the fit base learners
               %   while combining the most appropriate ensemble.
               %   Its values can taken as 'OFS' or 'GA'.
            
% featureRank  -   Ranking the features through the functions such as
               %   ReliefF and Shannon entropy according to their amount of
               %   information... Its values are as follows: 'shannon' or
               %   'relief'.
            
% K            -   The number of the nearest neighbor per class. By
               %   default, the value 'K' is 1.
% fold         -   This is the number of folds for cross-validation. The
               %   models are constructed without the cross validation
               %   if the field "fold" is not defined in the structure
               %   "Options". That is, all the data are used for the
               %   training.
% Parameters used for the test process of the algorithm
%--------------------------------------------------------------------------
% testSet      -   This is a test data to put to test by the VIBES
               %   algorithm.
% model        -   This is the output of the VIBES algorithm trained by
               %   the training set 'dataset', and includes the performance
               %   summary and models belonging to base learners.
%% OUTPUTS: If in training mode, the classifier's performance summary and hypothesis are returned. If test mode, the classifier's performance summary is returned.

%% VIBES algorithm
function [ varargout ] = vibes( varargin )
    if istable(varargin{1})
        [X, Y] = divideTable(varargin{1});
    
        if ~isstruct(varargin{2})
            error('The parameter "Options" must be structure, not a %s.', class(varargin{2}));
        end
        
        % Return the performance summary and hypothesis of the VIBES
        % algorithm.
        varargout{1} = trainEnsemble(X, Y, varargin{2});
    elseif isstruct(varargin{1})
        [testX, testY] = divideTable(varargin{2});
        
        % Return the performance summary on a test set of the VIBES
        % algorithm.
        varargout{1} = testEnsemble(varargin{1}, testX, testY, varargin{1}.Categories);
    end
end


%% Distinguish between the input matrix and the output vector
function [ X, Y ] = divideTable( dataset )
    if istable(dataset)
        X = table2array(dataset(:,1:end-1));
        
        % The name of the last column of the 'dataset' table must be
        % 'Class'.
        Y = categorical(dataset.Class);
    else
        error('The dataset must be a table, not a %s.', class(dataset));
    end
end


%% If in training mode, the classifier's performance summary and hypothesis are returned
function [ trainingOutput ] = trainEnsemble( X, Y, Options )
    % Rank features according to a feature-ranking method that you choice.
    if strcmpi(Options.featureRank, 'RF')
        % Set the value 'K' to 1 if the field 'K' is not defined in the
        % parameter 'Options'.
        if ~isfield(Options, 'K')
            Options.K = 1;
            warning('The value "K" is assigned 1 by default because of the fact that the field "K" is not defined in the structure "Options".');
        end
        [ranked, ~] = relieff(X, Y, Options.K);
    elseif strcmpi(Options.featureRank, 'IG')
        ranked = IG_Rank(X, Y);
    else
        error('It is an unknown feature-ranking function.');
    end
    
    if ~strcmpi(Options.baseLearner, 'nn')
        % Construct models without the cross validation if the field 'fold'
        % is not defined in the parameter 'Options'.
        if ~isfield(Options, 'fold')
            % Models have been constructed in accordance with the parameters.
            [Mdl, predictions] = constructModels(X, Y, ranked, Options);
            warning('The models are constructed without the cross validation if the field "fold" is not defined in the structure "Options". That is, all the data are used for the training.');
        else
            % Models have been constructed in accordance with the
            % given parameters.
            [Mdl, ~] = constructModels(X, Y, ranked, Options); 
            
            % Run k-fold cross-validation for each model.
            [predictions] = getCrossValModel(Mdl, X, Y, Options);
        end
    else
        % Construct models without the validtion and test if the field
        % 'trainRatio' is not defined in the parameter 'Options'.        
        if ~isfield(Options, 'trainRatio')
            warning('Now all the data are allocated for just the training. Remember be able to set the ratios of training, validation and test data (by default, 70%, 15%, 15%) as you wish for the Neural Network. Notice that this operation is different than k-fold cross-validation.');
        end
        % Models have been constructed in accordance with the
        % given parameters.
        [Mdl, predictions] = constructModels(X, Y, ranked, Options);        
    end
    
    % According to the search method (OFS, GA), make up an ensemble.
    if strcmpi(Options.searchMethod, 'OFS')
        % Make up an ensemble by fusing base learners which are selected 
        % for obtaining the highest accuracy rate by using the Optimized
        % Forward Search Algorithm.
        [hPredictions, indices, accuracyValue] = makeEnsemble_OFS(Y, predictions);
    elseif strcmpi(Options.searchMethod, 'GA')
        % Create options structure for Genetic Algorithm.
        % You can set the options of the GA using 'gaoptimset' as you wish.
        options = gaoptimset('PopulationType', 'bitstring');
        
        % Make up an ensemble by fusing base learners which are selected
        % for obtaining the highest accuracy rate by using Genetic
        % Algorithm.
        FitnessFcn = @(x)makeEnsemble_GA(x, Y, predictions);
        [~, col] = size(X);
        
        % Constrained optimization by using genetic algorithm.
        [x, accuracyValue, ~, ~, ~, ~] = ga(FitnessFcn, col+1, options);
        
        indices = find(x);
        hPredictions = finalHypotesis(predictions(:, indices), categories(Y));
    else
        error('It is an unknown search method.');
    end
    
    % Form the model, and computing its statistics (performance summary).
    trainingOutput = modelStatistics(Y, hPredictions);
    
    % Feature Indices.
    trainingOutput.Indices = indices;
    
    % Models belonging to base learners.
    trainingOutput.BaseLearnerModels = Mdl(indices);
    
    % The accuracy rate of the ensemble.
    trainingOutput.AccuracyValues = accuracyValue;
    
    % The order of features as their ranks.
    trainingOutput.RankedFeatures = ranked;
    
    % The name of the selected base learner
    trainingOutput.BaseLearner = Options.baseLearner;
    
    % The classes of the training set
    trainingOutput.Categories = categories(Y);
end


%% If in test mode, the classifier's performance summary is returned
function [ testOutput ] = testEnsemble( Model, testX, testY, classes )
    [~, col] = size(testX);    
    Indices = Model.Indices;
    RankedFeatures = Model.RankedFeatures;
    BaseLearnerModels = Model.BaseLearnerModels;
    len = length(Indices);
    
    % Creates a large matrix 'prediction' consisting of an (1)-by-(len)
    % tiling of copies of 'testY'.
    prediction = repmat(testY, 1, len);
    
    Ranges = cell(len, 1);
    
    handle = ProgressBar(0, len, 'The ensemble are making its predictions on the test data');
    
    for i = 1 : len
        if Indices(i) ~= (col + 1)
            % Assign the selected features to the cell 'Ranges{i}'
            Ranges{i} = num2cell(RankedFeatures(1:Indices(i)));            
            if ~strcmpi(Model.BaseLearner, 'nn')
                % Base Learners make predictions over a given test set.
                prediction(:, i) = predict(BaseLearnerModels{i}, testX(:, RankedFeatures(1:Indices(i))));
            else
                % Get the predictions of the Pattern Neural Network model
                prediction(:, i) = PredictionsOfNetwork(Model.BaseLearnerModels{i}, testX(:, RankedFeatures(1:Indices(i))), testY, classes);
            end
        else
            % Assign the selected features to the cell 'Ranges{i}'
            Ranges{i} = [RankedFeatures(1) RankedFeatures(col)];
            if ~strcmpi(Model.BaseLearner, 'nn')
                % Base Learners make predictions over a given test set.
                prediction(:, i) = predict(BaseLearnerModels{i}, testX(:, [RankedFeatures(1) RankedFeatures(col)]));
            else
                % Get the predictions of the Pattern Neural Network model
                prediction(:, i) = PredictionsOfNetwork(Model.BaseLearnerModels{i}, testX(:, [RankedFeatures(1) RankedFeatures(col)]), testY, classes);
            end
        end
        ProgressBar(i, len, 'The ensemble are making its predictions on the test data', handle);
    end
    
    testOutput.SelectedFeaturesPerDataset = Ranges;
    testOutput.PredictionsOfBaseLearners = prediction;
    ePrediction = finalHypotesis( prediction, classes );
    testOutput.PredictionsOfFinalHypothesis = ePrediction;
    testOutput.Actual = testY;
    testOutput.AccuracyRate = sum(testY == ePrediction)*100/size(testY,1);
end


%% Return a KNN classification model and its predictions for predictors X and response Y
function [ Mdl, Predictions ] = NearestNeighbor( X, Y )
    % Set the 'k' value for the nearest neighbor classifier. By
    % default, the value of 'k' is 1. You can alter it as you wish.
    % Besides, you can set the other parameters as you wish.
    k = 1;
    Mdl = fitcknn(X, Y, 'NumNeighbors', k, 'Distance', 'cityblock');
    
    % Predict the output of an identified model
    Predictions = predict(Mdl, X);
end


%% Return a naive Bayes model and its predictions for predictors X and response Y
function [ Mdl, Predictions ] = NaiveBayes( X, Y )
    % The kernels that you can choice for Naive Bayes are:
    % "normal", "mn", "kernel", "mvmn".
    % Besides, you can set the other parameters as you wish.
    Mdl = fitcnb(X, Y, 'DistributionNames', 'kernel');
    
    % Predict the output of an identified model
    Predictions = predict(Mdl, X);    
end


%% Fit a multiclass model for Support Vector Machine for predictors X and response Y, and Return it and its predictions
function [ Mdl, Predictions ] = SupportVectorMachine( X, Y )
    % The kernels that you can choice for SVM are as follows:
    % "linear", "gaussian", "rbf", "polynomial".
    % Besides, you can set the other parameters as you wish.
    t = templateSVM('Standardize', true, 'KernelFunction', 'linear');
    % Fit a multiclass model for Support Vector Machine
    Mdl = fitcecoc(X, Y, 'Learners', t);
    
    % Predict the output of an identified model
    Predictions = predict(Mdl, X);
end


%% Return a discriminant analysis model and its predictions for predictors X and response Y
function [ Mdl, Predictions ] = DiscriminantAnalysis( X, Y )
    % The discriminant types that you can choice for Discriminant
    % Analysis are as follows:
    % "linear", "pseudolinear", "diaglinear", "quadratic",
    % "diagquadratic", "pseudoquadratic".
    % Besides, you can set the other parameters as you wish.
    Mdl = fitcdiscr(X, Y, 'DiscrimType', 'linear');
    
    % Predict the output of an identified model
    Predictions = predict(Mdl, X);    
end


%% Return a classification decision tree and its predictions for predictors X and response Y
function [ Mdl, Predictions ] = DecisionTree( X, Y )
    Mdl = fitctree(X, Y);
    
    % Predict the output of an identified model
    Predictions = predict(Mdl, X);    
end


%% Return Bootstrap aggregation for an ensemble of decision trees and its predictions for predictors X and response Y
function [ Mdl, Predictions ] = RandomForest( X, Y )
    % The 'NumTrees' value is the number of decision trees in the
    % ensemble making up by Random Forest algorithm.
    % You can change it as you wish.
    % Besides, you can set the other parameters as you wish.
    NumTrees = 100;
    Mdl = TreeBagger(NumTrees, X, Y, 'OOBPrediction', 'on');
    
    % Predict the output of an identified model
    Predictions = predict(Mdl, X);    
end


%% Return a Neural Network model and its predictions (the backpropagation for now) for predictors X and response Y
function [ net, Predictions ] = NeuralNetwork( X, Y, Options )
    % The number of hidden layers in the network. you can specify a
    % network with 3 hidden layers, where the first hidden layer
    % size is 10, the second is 8, and the third is 5; the statement is :
    % [10, 8, 5].
    hiddenSizes = 10;
    
    % The following parameter, that is, trainFcn is a variable for
    % specifying the functions that you will be able to choose to
    % train the network. The list of the functions is as follows:
    % 'trainb'   : Batch training with weight & bias learning rules
    % 'trainbfg' : BFGS quasi-Newton backpropagation
    % 'trainlm'  : Levenberg-Marquardt backpropagation
    % 'trainbr'	 : Bayesian Regulation backpropagation
    % 'trainbu'  : Unsupervised batch training with weight & bias learning rules
    % 'trainbuwb': Unsupervised batch training with weight & bias learning rules
    % 'trainc'   : Cyclical order weight/bias training
    % 'trainbfg' : BFGS Quasi-Newton
    % 'trainrp'	 : RPROP backpropagation
    % 'trainscg' : Scaled conjugate gradient backpropagation
    % 'traincgb' : Conjugate gradient backpropagation with Powell-Beale restarts
    % 'traincgf' : Conjugate gradient backpropagation with Fletcher-Reeves updates
    % 'traincgp' : Conjugate gradient backpropagation with Polak-Ribiere updates
    % 'trainoss' : One Step Secant
    % 'traingdx' : Gradient descent w/momentum & adaptive lr backpropagation
    % 'traingdm' : Gradient Descent with Momentum
    % 'traingd'  : Gradient descent backpropagation
    % 'traingda' : Gradient descent with adaptive lr backpropagation
    % 'trainoss' : One step secant backpropagation
    % 'trainr'   : Random order weight/bias training
    % 'trainru'  : Unsupervised random order weight/bias training
    % 'trains'   : Sequential order weight/bias training
    trainFcn = 'trainbr';
    
    % This property defines the function used to measure the
    % network's performance. The available Performance Functions
    % are as follows:
    % 'mae'         : Mean absolute error performance function
    % 'mse'         : Mean squared error performance function
    % 'sae'         : Sum absolute error performance function
    % 'sse'         : Sum squared error performance function
    % 'crossentropy': Cross-entropy performance (default)
    % 'msesparse'   : Mean squared error performance function with
    % L2 weight and sparsity regularizers.
    performFcn = 'mse';
    % It returns a function Pattern Recognition neural network with a
    % hidden layer size of hiddenSizes, training function specified by
    % trainFcn, and performance function specified by performFcn.
    net = patternnet(hiddenSizes, trainFcn, performFcn);
    
    % Choose Input and Output Pre/Post-Processing Functions
    % The list of all processing functions are as follows:
    % 'fixunknowns'        : Processes matrix rows with unknown values
    % 'mapminmax'          : Map matrix row minimum and maximum values to [-1 1]
    % 'mapstd'             : Map matrix row means and deviations to standard values
    % 'processpca'         : Processes rows of matrix with principal component analysis
    % 'removeconstantrows' : Remove matrix rows with constant values
    % 'removerows'         : Remove matrix rows with specified indices
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    % This property defines the data division function to be used
    % when the network is trained using a supervised algorithm,
    % such as backpropagation. You can set this property to the
    % name of a division function. There are four functions
    % provided for dividing data into training, validation and test
    % sets. They are as follows:
    % 'dividerand' : Divide the data randomly (default)
    % 'divideblock': Divide the data into contiguous blocks
    % 'divideint'  : Divide the data using an interleaved selection
    % 'divideind'  : Divide the data by index
    net.divideFcn = 'dividerand';
    % This property defines the target data dimensions which to
    % divide up when the data division function is called. Its
    % default value is 'sample' for static networks and 'time' for
    % dynamic networks. It may also be set to 'sampletime' to
    % divide targets by both sample and timestep, 'all' to divide
    % up targets by every scalar value, or 'none' to not divide up
    % data at all (in which case all data is used for training,
    % none for validation or testing).
    if isfield(Options, 'trainRatio')
        net.divideMode = 'sample';
    else
        % All the data is allocated for just the training
        net.divideMode = 'none';
    end
    % This property defines the parameters and values of the
    % current data-division function.
    if isfield(Options, 'trainRatio')
        net.divideParam.trainRatio = Options.trainRatio/100;
        net.divideParam.valRatio   = Options.valRatio/100;
        net.divideParam.testRatio  = Options.testRatio/100;
    end
    
    % Maximum number of epochs to train
    net.trainParam.epochs = 7000;
    
    % Do not show training GUI
    net.trainParam.showWindow = 0;
    % The cause of using the function below is that the train function does
    % disapprove the categorical data. Therefore, we convert data into
    % numerical type. The explanation of the function used for this is
    % right below.
    % Create index vector from a grouping variable.
    % For [G, GN, GL] = grp2idx(S);
    % It returns a column vector GL representing the group levels. The set
    % of groups and their order in GL and GN are the same, except that GL
    % has the same type as S. If S is a character matrix, GL(G,:)
    % reproduces S, otherwise GL(G) reproduces S.
    [T, ~, GL] = grp2idx(Y);
    
    % Transpose the input and the output to train the network
    t = T';
    x = X';
    
    % Train the neural network that you specified its properties.
    [net, ~] = train(net, x, t);
    
    % Predict the output of an identified model
    YY = net(x);
    YY = round(YY, 0);
    Predictions = GL(YY);
end


%% Predict the output of an identified model
function [ Predictions ] = PredictionsOfNetwork( net, X, Y, classes )
    
    %[~, ~, GL] = grp2idx(Y);
    GL = classes;
    x = X';
    YY = net(x);
    YY = round(YY, 0);
    Predictions = GL(YY);
end


%% Construct the models in accordance with the parameters
function [ Mdl, Predictions ] = constructModels( X, Y, ranked, Options )
    [~, col] = size(X);
    
    % Creates a large matrix 'prediction' consisting of an (1)-by-(col+1)
    % tiling of copies of 'Y'.
    Predictions = repmat(Y, 1, col+1); 
    
    % Create cell array.
    Mdl = cell(col+1,1);
    
    handle = ProgressBar(0, col+1, 'The models are constructed...');
    
    for i=1 : col+1
        if Options.isDependent
            if i~=col+1
                indices = ranked(1:i);
            else
                indices = [ranked(1); ranked(end)];
            end  
        else
            if i~=col+1
                indices = i;
            else
                indices = 1:col;
            end                      
        end
        
        if strcmpi(Options.baseLearner, 'knn')
            [Mdl{i}, Predictions(:,i)] = NearestNeighbor(X(:,indices), Y);
        elseif strcmpi(Options.baseLearner, 'nb')
            [Mdl{i}, Predictions(:,i)] = NaiveBayes(X(:,indices), Y);
        elseif strcmpi(Options.baseLearner, 'svm')
            [Mdl{i}, Predictions(:,i)] = SupportVectorMachine(X(:,indices), Y);
        elseif strcmpi(Options.baseLearner, 'disc')
            [Mdl{i}, Predictions(:,i)] = DiscriminantAnalysis(X(:,indices), Y);
        elseif strcmpi(Options.baseLearner, 'tree')
            [Mdl{i}, Predictions(:,i)] = DecisionTree(X(:,indices), Y);
        elseif strcmpi(Options.baseLearner, 'rf')
            [Mdl{i}, Predictions(:,i)] = RandomForest(X(:,indices), Y);
        elseif strcmpi(Options.baseLearner, 'nn')
            [Mdl{i}, Predictions(:,i)] = NeuralNetwork(X(:,indices), Y, Options);
        else
            error('It is an unknown base learner.');
        end
        ProgressBar(i, col+1, 'The models are constructed...', handle);
    end
    ProgressBar(handle);
end


%% Run k-fold cross-validation for each model
function [ prediction ] = getCrossValModel( Mdl, X, Y, Options )
    [row, col] = size(X);
    
    % Creates a large matrix 'prediction' consisting of an (1)-by-(col+1)
    % tiling of copies of 'Y'.
    prediction = repmat(Y, 1, col+1);
    
    if Options.fold < 1
        error('The number of fold cannot be small than 1.');
    end
    
    handle = ProgressBar(0, col+1, 'The models are making their predictions');
    
    % Stratified cross-validation
    %cp = cvpartition(Y,'k',10);
    for i=1 : col+1
        if ~strcmpi(Options.baseLearner, 'rf')
            %CVMdl = crossval(Mdl{i}, X, cellstr(Y), 'partition', cp);
            if Options.fold < row
                % Perform k-fold cross-validation for the model 'Mdl{i}'.
                CVMdl = crossval(Mdl{i}, 'kfold', Options.fold);
            elseif Options.fold == row
                % Leave-one-out cross validation.
                CVMdl = crossval(Mdl{i}, 'Leaveout', 'on');
            else
                error('The number of fold cannot be big than the row count in dataset.');
            end
            % Return cross-validated predicted responses by the
            % cross-validated kernel regression model 'CVMdl'.
            [YHat, ~] = kfoldPredict(CVMdl);
            prediction(:,i) = YHat;
        else
            % Return the predicted responses for the out-of-bag data in
            % the ensemble.
            prediction(:,i) = oobPredict(Mdl{i});
        end
        ProgressBar(i, col+1, 'The models are making their predictions', handle);
    end
    ProgressBar(handle);
end


%% Generate the predictions of a final hypothesis, i.e. a model, for an ensemble
function [ ePrediction ] = finalHypotesis( predictions, classes )
    % Count occurrences of categories in the elements of the categorical
    % array 'prediction'.
    predictedClassCount = countcats(predictions,2);
    
    % Find the indices of a column vector containing the maximum value of
    % each row. If the maximum value occurs more than once, then max
    % returns the index corresponding to the first occurrence.
    [~, I] = max(predictedClassCount,[],2);
    
    % Convert the cell array 'class' into the categorical array
    % 'ePrediction'.
    ePrediction = categorical(classes(I(1:end)));
end


%% Make up an ensemble by fusing base learners which are selected so as to obtain the highest accuracy rate by using Optimized Forward Search Algorithm
function [ prediction, index, accuracyValue ] = makeEnsemble_OFS( actual, hPredictions )
    
    [~, b] = size(hPredictions);
    index = zeros(1, b);
    accuracyValue = zeros(1, b);
    waitCount = 0;
    
    % Set the constant 'cycle', ending the search up while combining the
    % fittest ensemble in the course of the optimization.
    cycle = log(b);
    
    handle = ProgressBar(0, b*b, 'The models are combined');
    
    % The process of putting together base learners.
    for i=1 : b
        res = -1;
        for j=1 : b
            if sum(index == j) > 0
                continue;
            end
            prediction = finalHypotesis([hPredictions(:, index(1:(i-1))) hPredictions(:, j)], categories(actual));
            acc =  sum(prediction == actual);
            if res < acc
                res = acc;
                accuracyValue(i) = acc;
                index(i) = j;
            end
            ProgressBar((i-1)*b+j, b*b, 'The models are combined', handle);
        end
        
        % The necessary codes for the optimization.
        if max(accuracyValue) > accuracyValue(i)
            waitCount = waitCount + 1;
            % If the 'waitCount' is bigger than the 'cycle', the global
            % optimum point was reached.
            if waitCount > cycle
                break;
            end
        else
            waitCount = 0;
        end
    end
    
    % Get the index of the combination that the accuracy rate is the
    % highest.
    [~, Ind] = max(accuracyValue);
    
    % Only get the combination with the highest accuracy rate.
    index = index(1:Ind);
    
    % Get the predictions of a final hypothesis.
    prediction = finalHypotesis(hPredictions(:, index), categories(actual));
    
    ProgressBar(handle);
end


%% Make up an ensemble by fusing base learners which are selected so as to obtain the highest accuracy rate by using Genetic Algorithm
function [ accuracyValue, prediction, index ] = makeEnsemble_GA( index, actual, hPredictions )
    if sum(index) == 0
        accuracyValue = 0;
    else
        prediction = finalHypotesis(hPredictions(:, (index>0)), categories(actual));
        accuracyValue = sum(prediction == actual);
    end
end


%% Form the model, and Compute its statistics (performance summary)
function [ output ] = modelStatistics( actual, ePrediction )
    Class = categories(actual);
    NumberOfInstanceAsClass = countcats(actual);
    NumberOfClass = length(Class);
    ROC = zeros(NumberOfClass, 1);
    
    % Number of Instances.
    NumberOfInstance = length(actual);
    output.NOI = NumberOfInstance;
    
    % Set Confusion Matrix.
    ConfusionMatrix = confusionmat(actual, ePrediction);
    
    % Compute True Positive.
    TP = diag(ConfusionMatrix) ./ sum(ConfusionMatrix,2);
    
    % Compute False Positive.
    FP = (sum(ConfusionMatrix,1)' - diag(ConfusionMatrix)) ./ (length(actual) - sum(ConfusionMatrix,2));
    
    % Compute Precision.
    Precision = diag(ConfusionMatrix) ./ sum(ConfusionMatrix,1)';
    
    % Compute Recall.
    Recall = TP;
    
    % Compute F-Measure.
    F_Measure = (2*Precision.*Recall)./(Precision+Recall);
    
    % Compute Cohen's Kappa Statistic.
    pA = trace(ConfusionMatrix) / NumberOfInstance;
    pE = sum((sum(ConfusionMatrix,2).*sum(ConfusionMatrix,1)') ./ (NumberOfInstance*NumberOfInstance));
    output.Kappa = (pA-pE)/(1-pE);
    
    predict = (actual == ePrediction);
    meanPredict = mean(predict);
    sumPredict = sum(predict);
    
    % Compute Mean Absolute Error.
    output.MAE = 1-meanPredict;
    
    % Compute Root Mean Squared Error.
    output.RMSE = sqrt(1-meanPredict);
    
    % Compute Relative Absolute Error.
    output.RAE = sumPredict / sum(abs(meanPredict-double(predict)));
    
    % Compute Root Relative Squared Error.
    output.RRSE = sqrt(sumPredict / sum((meanPredict-double(predict)).^2));
    
    % Compute 'TPR' as each Class.
    last = 0;
    TPR = zeros(length(actual), NumberOfClass);
    for i = 1 : NumberOfClass
        first = last + 1;
        last = last + NumberOfInstanceAsClass(i);
        row = 1;
        for j = first : last
            TPR(row, i) = sum(ePrediction(first:j) == Class(i)) / NumberOfInstanceAsClass(i);
            row = row + 1;
        end
        TPR(row:end, i) = TPR(row-1, i);
    end
    
    % Compute Area Under ROC.
    for i = 1 : NumberOfClass
        ROC(i) = TP(i)*(1-FP(i));
    end
    % Create the detailed accuracy table by class.
    Class{NumberOfClass+1} = '';
    RowNames = cell(NumberOfClass+1,1);
    for i = 1 : NumberOfClass
        RowNames(i) = cellstr(int2str(i));
    end
    RowNames(end) = cellstr('Weighted Avg.');
    
    output.DetailedAccuracyByClass = table([TP; mean(TP)], [FP; mean(FP)], [Precision; mean(Precision)], ...
                                    [Recall; mean(Recall)], [F_Measure; mean(F_Measure)], [ROC; mean(ROC)], ...
                                    cellstr(Class), ...
                                    'VariableNames', {'TP', 'FP', 'Precision', 'Recall', 'FMeasure', 'ROC', ... 
                                    'Class'}, 'RowNames', RowNames);
    
    % The predicted output.
    output.Prediction = ePrediction;
    
    % Correctly Classified Instances.
    output.CCI = trace(ConfusionMatrix);
    
    % Incorrectly Classified Instances.
    output.ICI = length(actual) - trace(ConfusionMatrix);
    
    % Accuracy Rate
    output.AccuracyRate = output.CCI * 100 / output.NOI;
    
    % Confusion Matrix.
    CM = table;
    for i = 1 : NumberOfClass
        CM(:,i) = num2cell(ConfusionMatrix(:,i));
    end
    CM.Properties.VariableNames = Class(1:end-1);
    CM.Properties.RowNames = Class(1:end-1);
    output.ConfusionMatrix = CM;
end


%% Compute the ranks of the features according to the information gains
function [ ranked ] = Ranking( gains )
    
    % Sort the elements of gains in ascending order.
    [ngains, I] = sort(gains);
    
    % Reverse the order of elements.
    featureOrder = I(end:-1:1);
    ngainsOrder = ngains(end:-1:1);
    
    % Sort rows in ascending order.
    A = sortrows([ngainsOrder featureOrder]);
    
    % Returns a vector ranked with the same dimensions as 'A', but with
    % the order of elements flipped.
    ranked = flip(A(:,2));
end


%% Compute the entropy of the class, i.e. Compute H(Y)
function [ H ] = H_Theorem( p )
    % Create a frequency table
    % The first column in 'T' contains the unique string values in 'p'.
    % The second is the number of instances of each value.
    % The last column contains the percentage of each value.
    T = tabulate(p);
    % Get a vector including the number of instances per class.
    m = cell2mat(T(:,2));
    % Compute probabilities per class.
    probPerClass = m ./ sum(m);
    % Ignore bins with 0 probability such as 0*log(0)=0.
    probPerClass = probPerClass(probPerClass > 0);
    % Compute Boltzmann's H Theorem.
    H = -sum(probPerClass .* log2(probPerClass));
end


%% Compute the entropy per attributes, i.e. Compute H(Y|X)
function [ eoa ] = entropyOfAttribute( X, outcome )
    eoa = 0;
    % It partitions the values in 'X' into bins, and is an array of the
    % same size as 'X' whose elements are the 'BIN' indices for the
    % corresponding elements in 'X'.
    [~, ~, BIN] = histcounts(X, 'BinMethod', 'sturges');
    
    for c = unique(BIN)'
        idx = (BIN == c);
        p = sum(idx) / length(X);
        if p > 0
            eoa = eoa + p * H_Theorem(outcome(idx));
        end
    end  
end


%% Compute the Information Gain per features, i.e. Compute IG = H(Y) - H(Y|X)
function [ gains ] = InformationGain( features, outcome )
    
    % Compute the entropy of the class 
    outcomeEntropy = H_Theorem(outcome);
    
    d = size(features, 2);
    
    % Compute entropy per features.
    gains = zeros(d, 1);
    for i = 1 : d
        gains(i) = outcomeEntropy - entropyOfAttribute(features(:,i), outcome);
    end
end


%% Sort the features according to their ranks
function [ ranked ] = IG_Rank( features, outcome )
    
    gains = InformationGain(features, outcome);
    ranked = Ranking(gains);
end


%% Progress Bar
function [ handle ] = ProgressBar( varargin )
    if (nargin == 1)
        close(varargin{1});
    elseif nargin == 3
        ratio = varargin{1}/varargin{2};
        handle = waitbar(ratio, strcat(num2str(round(ratio*100, 0)), '%'), 'Name', varargin{3});
    elseif nargin == 4
        ratio = varargin{1}/varargin{2};
        waitbar(ratio, varargin{4}, strcat(num2str(round(ratio*100, 0)), '%'), 'Name', varargin{3});
    else
        error('There are too many parameters.');
    end
end
