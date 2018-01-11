%Prevent Octave from thinking this is a function file_in_loadpath
%1;
%load poker-hand-testing-tiny.data

%Feedforward network training.
%Very very slow for some reason despite
%small data sizes. Why? Dunno.

%[WEIGHTS, ERRCURVE, TRAINACC, TESTACC] = TrainFF_GPU(TR, TE, 100, 100);
%optional args: learn_rate, momentum, num_layers, minibatch_size,
%scaleFeatures



function [WEIGHTS, ERRCURVE, TRAINACC, TESTACC, MINWEIGHTS, MAXWEIGHTS, MINUPDATES, MAXUPDATES] = TrainFF_GPU(TRAIN, ...
    TEST, num_hidden1, iterations, varargin)
matlab = false;
if(matlab)
    rng(42);
end


learn_rate = 0.1;
momentum = 0.0;
num_layers = 3;
minibatch_size = 1;
scaleFeatures = false;
WEIGHTS = {};
init_weights = true;
%tanh and relu did not work for mnist,
%not sure why.
%don't seem to work for poker either.
%This was because my momentum is broken. Actually tanh works fine.

type = 'relu';

if(length(varargin) > 0)
    learn_rate = varargin{1};
end
if(length(varargin) > 1)
    momentum = varargin{2};
end
if(length(varargin) > 2)
    num_layers = varargin{3};
end
if(length(varargin) > 3)
    minibatch_size = varargin{4};
end
if(length(varargin) > 4)
    scaleFeatures = varargin{5};
end
if(length(varargin) > 5)
    WEIGHTS = varargin{6};
    init_weights = false;
end

%Assumption: final row contains y, the target
TRAIN = GpuOrLocal(TRAIN);
TEST = GpuOrLocal(TEST);
trainRowCnt = size(TRAIN, 1);
testRowCnt = size(TEST, 1);
colCnt = size(TRAIN, 2);
featCnt = colCnt;

SCALERATIO=-100;
MINUNSCALED=-100;

TR_X = GetFeatures(TRAIN);
TE_X = GetFeatures(TEST);

%Scale features for faster convergence?

if(scaleFeatures)
    minScaled = -1;
    maxScaled = 1;
    
    minUnscaled = min(min([TR_X; TE_X]));
    MINUNSCALED=minUnscaled;
    maxUnscaled = max(max([TR_X; TE_X]));
    ratio = (maxUnscaled - minUnscaled) / (maxScaled - minScaled);
    SCALERATIO=ratio;
    
    TR_X = Rescale(TR_X, ratio, minUnscaled, minScaled);
    %TR_X = minScaled + ((TR_X - minUnscaled) / ratio);
    %TE_X = minScaled + ((TE_X - minUnscaled) / ratio);
    TE_X = Rescale(TE_X, ratio, minUnscaled, minScaled);
end


TR_X = [ones(trainRowCnt, 1) TR_X];
TR_Y = GetTarget(TRAIN);

TE_X = [ones(testRowCnt, 1) TE_X];
TE_Y = GetTarget(TEST);

TR_X = GpuOrLocal(TR_X);
TR_Y = GpuOrLocal(TR_Y);
TE_X = GpuOrLocal(TE_X);
TE_Y = GpuOrLocal(TE_Y);
ERRCURVE = [];
TRAINACC = [];
TESTACC = [];
MINWEIGHTS=[];
MAXWEIGHTS=[];
MINUPDATES=[];
MAXUPDATES=[];

switch(type)
case 'sigmoid'
  range_min = -.05;
  range_max = .05;
case 'tanh'
  range_min = -.2;
  range_max = .2;
 case 'relu'
  %range_min = -.05;
  %range_max = .05;
  range_max = sqrt(2.0/num_hidden1);
  range_min = -range_max;
end

num_output = FromGpuOrLocal(max(TR_Y) + 1);


if(init_weights)
    WEIGHTS{1} = GpuOrLocal(RandMatrix(num_hidden1, featCnt, range_min, range_max));
    currIntermediate = 2;
    %TODO: Make hidden layer sizes configurable
    %TODO: Test me!
    while(currIntermediate < num_layers)
        WEIGHTS{currIntermediate}=GpuOrLocal(RandMatrix(num_hidden1, num_hidden1, ...
            range_min, range_max));
        currIntermediate = currIntermediate + 1;
    end
    WEIGHTS{currIntermediate} = GpuOrLocal(RandMatrix(num_output, num_hidden1, range_min, range_max));
end

UPDATES_PREV={};
%for adagrad
UPDATES_SQUARED_SUM={};
currLayer = 1;
while(currLayer <= num_layers)
    currRows = size(WEIGHTS{currLayer}, 1);
    currCols = size(WEIGHTS{currLayer}, 2); 
    UPDATES_PREV{currLayer} = GpuOrLocal(zeros(currRows, currCols));
    UPDATES_SQUARED_SUM{currLayer} = GpuOrLocal(ones(currRows, currCols));
    currLayer = currLayer + 1;
end

target = GpuOrLocal(zeros(num_output, trainRowCnt));
row = 1;
while(row <= trainRowCnt)
    target(TR_Y(row,1) + 1, row) = 1;
    row = row + 1;
end

curr_iter = 0;

PLOT_ITER = [];

best_test_acc = -1;
if(minibatch_size ~= 1)
  learn_rate = learn_rate / minibatch_size;
end
errfig = figure;
accfig = figure;
while curr_iter < iterations
    %no cellfun on gpu
    %minweight = min(cellfun(@(C) min(min(C)), WEIGHTS));
    minweight = 0.0;
    %minweight = min(min(WEIGHTS));
    MINWEIGHTS = [MINWEIGHTS minweight];
    %maxweight = max(max(WEIGHTS));
    %maxweight = max(cellfun(@(C) max(max(C)), WEIGHTS));
    maxweight = 0.0;
    MAXWEIGHTS = [MAXWEIGHTS maxweight];
    PLOT_ITER = [PLOT_ITER curr_iter];
    minupdate = 10000;
    maxupdate = -10000;
    right = 0;
    wrong = 0;
    row = 1;
    total_error = 0;
    while row + minibatch_size - 1 <= trainRowCnt
        xBatch = TR_X(row:row+minibatch_size - 1,:);
        yBatch = target(:,row:row+minibatch_size - 1);
        %Compute output
        %Multilayer: This will just be a vector.
        [outputs] = Forward(WEIGHTS, xBatch', type);
        %Update error
        [curr_error, correct, incorrect] = ComputeError(outputs{num_layers}, yBatch);
        right = right + correct;
        wrong = wrong + incorrect;
        total_error = total_error + curr_error;
        %Compute derivatives
        [UPDATES] = ComputeUpdates(outputs, xBatch, yBatch,  ...
            WEIGHTS, UPDATES_PREV, UPDATES_SQUARED_SUM, learn_rate, momentum, type);
        %curr_minupdate = min(cellfun(@(C) min(min(C)), UPDATES));
        curr_minupdate = 0.0;
        if(curr_minupdate < minupdate)
          minupdate = curr_minupdate;
        end
        %curr_maxupdate = max(cellfun(@(C) max(max(C)), UPDATES));
        curr_maxupdate=0.0;
        if(curr_maxupdate > maxupdate)
          maxupdate = curr_maxupdate;
        end
	      currLayer = 1;
        while(currLayer <= num_layers)
            WEIGHTS{currLayer} = WEIGHTS{currLayer} + UPDATES{currLayer};
            UPDATES_PREV{currLayer} = UPDATES{currLayer};
            thing1 = UPDATES{currLayer}.^2;
            thing2 = UPDATES_SQUARED_SUM{currLayer} + thing1;
            UPDATES_SQUARED_SUM{currLayer} = thing2;
            UPDATES_SQUARED_SUM{currLayer} = UPDATES_SQUARED_SUM{currLayer} + (UPDATES{currLayer}.^2);
            currLayer = currLayer + 1;
        end
        row = row + minibatch_size;
    end
    MINUPDATES = [MINUPDATES minupdate];
    MAXUPDATES = [MAXUPDATES maxupdate];
    
    
    
    total_error = total_error / 2;
    ERRCURVE = [ERRCURVE total_error];
    accTrain = ComputeAccuracy(TR_X, TR_Y, WEIGHTS, type);
    TRAINACC = [TRAINACC accTrain];
    accTest = ComputeAccuracy(TE_X, TE_Y, WEIGHTS, type);
    if(accTest > best_test_acc)
        best_test_acc = accTest;
        try
            save('bestweights', 'WEIGHTS');
        catch ME
        end
    end
    TESTACC = [TESTACC accTest];
    set(0, 'CurrentFigure', errfig);
    plot(PLOT_ITER, ERRCURVE);
    legend('err', 'Location', 'northeast');
    set(0, 'CurrentFigure', accfig);
    plot(PLOT_ITER, TRAINACC, PLOT_ITER, TESTACC);
    legend('acctrain', 'actest', 'Location', 'southeast');
    %plot(PLOT_ITER, ERRCURVE);
    drawnow();
    curr_iter = curr_iter + 1;
    %shuffle
    permuted = randperm(trainRowCnt);
    TR_X = TR_X(permuted,:);
    target = target(:,permuted);
    TR_Y = TR_Y(permuted,:);
end
end

function trn = GetFeatures(getTrainFromMe)
    cols = size(getTrainFromMe, 2);
    trn = getTrainFromMe(:,1:cols - 1);
end

function tgt = GetTarget(getTargetFromMe)
    cols = size(getTargetFromMe, 2);
    tgt = getTargetFromMe(:,cols);    
end

function rescaled = Rescale(rescaleMe, ratio, minUnscaled, minScaled)
    rescaled = minScaled + ((rescaleMe - minUnscaled) / ratio);
end

function casted = GpuOrLocal(X)
run_on_gpu = false;
single_precision = true;

casted = X;
if(single_precision)
    casted = single(casted);
end
if(run_on_gpu)
    casted = gpuArray(casted);
end
end

function gathered = FromGpuOrLocal(X)
run_on_gpu = false;
gathered = X;
if(run_on_gpu)
    gathered = gather(gathered);
end
end



function M = RandMatrix(rows, cols, min, max)
rr = round(rows);
rc = round(cols);
M = rand(rr, rc);
M = M .* (max - min);
M = M - ((max - min) / 2);
end



function df = SquishDeriv(needDeriv, type)
switch type
    case 'sigmoid'
        df = needDeriv .* (1-needDeriv);
    case 'relu'
        df = double(needDeriv > 0);
    case 'tanh'
        %df = 1 - (needDeriv.^2);
        %alternative formulation from https://www.csee.umbc.edu/~dpatte3/nn/res/bbackprop.m
        %This is the same thing, duh.
        df = (1 + needDeriv) .* (1 - needDeriv);
end
end




function [error, correct, incorrect] = ComputeError(out, target)
[~, maxidx] = max(target);
%Classes start at zero but Octave 1-based
ref = maxidx;
%assert(dontcare == 1);
[~, prediction] = max(out);
correct = sum(ref - prediction == 0);
incorrect = size(out, 2) - correct;
squared_error_row = (target - out) .* (target - out);
error = sum(sum(squared_error_row));
end


function [UPDATES] = ComputeUpdates(outputs, x, ...
    y, WEIGHTS, UPDATES_PREV, UPDATES_SQUARED_SUM, learn_rate, momentum, type)

UPDATES={};
num_layers = size(outputs,2);
%dk = outputs{num_layers} .* (1 - outputs{num_layers}) .* (y - outputs{num_layers});
dk = SquishDeriv(outputs{num_layers}, type) .* (y - outputs{num_layers});
%Update weights
%BUGBUG: For layers with a bias, need to revert x_0 to the bias here!!!
tmp = outputs{num_layers - 1};
tmp(1,:) = ones(1, size(tmp, 2));
%UPDATES{num_layers} = dk * outputs{num_layers - 1}';
UPDATES{num_layers} = dk * tmp';
UPDATES{num_layers} = UPDATES{num_layers}.* learn_rate;
UPDATES{num_layers} = UPDATES{num_layers} + (momentum .* UPDATES_PREV{num_layers});

dLast = dk;

%Update weights for all layers after output
curr_layer = num_layers - 1;
while(curr_layer > 0)
    %currD = outputs{curr_layer} .* (1- outputs{curr_layer}) .* ...
    currD = SquishDeriv(outputs{curr_layer}, type) .* ...
        (WEIGHTS{curr_layer + 1}' * dLast);
    if(curr_layer == 1)
	    UPDATES{1} = currD * x;
    else
	    tmp = outputs{curr_layer - 1};
	    tmp(1,:) = ones(1, size(tmp, 2));
	    UPDATES{curr_layer} = currD * tmp';
    end
    
    %adagrad
    UPDATES{curr_layer} = UPDATES{curr_layer} ./ (1e-6 + sqrt(UPDATES_SQUARED_SUM{curr_layer}));
    UPDATES{curr_layer} = UPDATES{curr_layer} .* learn_rate;
    %UPDATES{curr_layer} = UPDATES{curr_layer} + (momentum .* UPDATES_PREV{curr_layer});
    dLast = currD;
    curr_layer = curr_layer - 1;
end

end
