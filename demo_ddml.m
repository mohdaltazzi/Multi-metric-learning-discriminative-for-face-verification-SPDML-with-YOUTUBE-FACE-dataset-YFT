%% DDML demo

clc;
clear all;

addpath('activations');

%% LFW data: sparse sift (ssift)

%load('data/lfw_sift.mat');
load('E:\Phd Trip\Test new dataset\data\YTFstart77_uxx.mat');

rand('state', 0);

dim_layer = [300 200 150];  % [300 200 150]   %[250 150 100]

nL = length(dim_layer);     % number of layers

ux = ux(1:dim_layer(1),:);

n_sam = size(ux, 2);
x_mean =  mean(ux, 2);
ux = ux - repmat(x_mean , 1, n_sam);

%% initializing
[Wo, bo] = initialize_Weights(dim_layer, 1);


matches = logical(pairs(:,4)); %1*6000
un = unique(pairs(:,3)); 

%nfold = 10;
nfold = length(un);
loss_f= zeros(1,nfold);

t_acc = zeros(nfold, 1);
for c = 1:nfold
    
    disp ('*************************************************************************************');
    disp ('*************************************************************************************');
    disp(['fold:  ' num2str(c)]);
    
    
      trainMask = pairs(:,3) ~= c; % 9 folds except fold ?= c
      testMask = pairs(:,3) == c;  % one fold only == c
      
    tr_idxa = idxa(trainMask); % get elements of indxa at the index = trainMask
    tr_idxb = idxb(trainMask); % get elements of indxa at the index = trainMask
    tr_matches = matches(trainMask); 
    
    [W, b ] = ddml_bp(tr_idxa, tr_idxb, tr_matches, ux, Wo, bo, nL);
    
    
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    ts_matches = matches(testMask);
    
    Xa = ux(:, ts_idxa);
    Xb = ux(:, ts_idxb);
    Xa = ddml_fp(Xa, W, b, nL);
    Xb = ddml_fp(Xb, W, b, nL);
    
    % cosine similarity
    sim = cos_sim(Xa, Xb);
    % accuracy
    [~, ~, acc] = ROCcurve(sim, ts_matches);
    t_acc(c) = acc;
end

disp([mean(t_acc) std(t_acc)]); % show mean accuracy
stderror= std( t_acc ) / sqrt( length( t_acc ));
disp ('SE= ');
disp (stderror);