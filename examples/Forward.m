%Multilayer: weight inputs become cell arrays
%Return becomes matrix.
%Replace with loop over cell array of layers.
function [outputs] = Forward(WEIGHTS, x, type)
dropouts = [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0];
outputs={};
outputs{1} = WEIGHTS{1} * x;
outputs{1} = Squish(outputs{1}, type);
currOut = 2;
while(currOut <= size(WEIGHTS,2))
    %bias
    tmp = outputs{currOut - 1};
    tmp(1,:) = 1;
    %outputs{currOut - 1}(1,:) = 1;
    outputs{currOut} = WEIGHTS{currOut} * tmp;
    outputs{currOut} = Squish(outputs{currOut}, type);
    dropout = dropouts(currOut);
    if(dropout ~= 1.0)
        mask = (rand(size(outputs{currOut})) < dropout) / dropout;
        outputs{currOut} = outputs{currOut} .* mask;
    end
    currOut = currOut + 1;
end
end

function [outputs] = Squish(squishMe, type)
switch type
    case 'sigmoid'
        outputs = Sigmoid(squishMe);
    case 'relu'
        outputs = Rectifier(squishMe);
    case 'tanh'
        outputs = tanh(squishMe);
       
end
end

function rect = Rectifier(rectifyMe)
mask = rectifyMe > 0;
rect = mask .* rectifyMe;
end

function sig = Sigmoid(sigmoidMe)
sig = 1 ./ (1 + (exp(-sigmoidMe)));
end
