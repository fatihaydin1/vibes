%% Authors: Fatih Aydin & Zafer Aslan
function [ testIndices, trainIndices ] = split( dataset, ratio )
    if ratio >= 1
        error('The parameter "ratio" cannot be equal or bigger than 1.');
    end
    row = size(dataset, 1);
    testIndices = randperm(row, floor(row*ratio));
    trainIndices = setdiff(1:row, testIndices);
end