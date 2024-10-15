%% Initialization
warning off; close all; clear; clc

%%  Data input
res = xlsread('ExperimentalData.xlsx'); % Input data to be analyzed

%% SSA decomposation if the preprocessing results are not included in the data input
%res = xlsread('NorthLineData.xlsx');
%res = res(:,1);
%[LT,ST,R] = trenddecomp(res,"ssa",7*24);
Lag = 71; %% 48h-ahead is 48 hours, and the input variables are x, x-1, ...,x-23 h. Lag is 71 in total
for i = 1:Lag+1
    for ii = 1:21
        eval(['resA',num2str(ii),'(:,i)=res(i:end-Lag-1+i,',num2str(ii),');']); %resA1(:,i)=res(i:end-Lag-1+i,1);
    end
end
clear res; 

save_net = [];
for jj = 1:21
    eval(['res(:,1:24)=resA',num2str(jj),'(:,1:24);']); %res(:,1:Lag)=resA1(:,1:Lag);
    eval(['res(:,25)=resA',num2str(jj),'(:,end);']); %res(:,Lag+1)=resA1(:,end);

%%  Data analysis
    num_size = 0.8;                              % Proportion of training set in the dataset
    outdim = 1;                                  % Output at the last column
    num_samples = size(res, 1);                  % Number of samples
    resForTrain = res;
    num_train_s = round(num_size * num_samples); % Nuber of training samples
    f_ = size(res, 2) - outdim;                  % Output dimension

    %%  Split into training set and test set
    P_train = resForTrain(1: num_train_s, 1: f_)';
    T_train(jj,:) = resForTrain(1: num_train_s, f_ + 1: end)';
    M = size(P_train, 2);

    P_test = res(num_train_s + 1: end, 1: f_)';
    T_test(jj,:) = res(num_train_s + 1: end, f_ + 1: end)';
    N = size(P_test, 2);

    %%  Normalization
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test = mapminmax('apply', P_test, ps_input);

    [t_train, ps_output] = mapminmax(T_train(jj,:), 0, 1);
    t_test = mapminmax('apply', T_test(jj,:), ps_output);

    %%  Format conversion
    for i = 1 : M 
        vp_train{i, 1} = p_train(:, i);
        vt_train{i, 1} = t_train(:, i);
    end

    for i = 1 : N 
        vp_test{i, 1} = p_test(:, i);
        vt_test{i, 1} = t_test(:, i);
    end

    %%  Network construction
    numFilters = 32; %Number of filters
    filterSize = 3; % Filter size
    dropoutFactor = 0.2; %Dropout factor
    numBlocks = 3; %Number of blocks

    %%  Model development
    lgraph = layerGraph();
    layer = sequenceInputLayer(f_,Normalization="rescale-symmetric",Name="input");
    lgraph = layerGraph(layer);
    
    outputName = layer.Name;
    
    for i = 1:numBlocks
        dilationFactor = 2^(i-1);                                   % Dilation factor
        
        layers = [
            convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
            layerNormalizationLayer
            dropoutLayer(dropoutFactor) 
            convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
            layerNormalizationLayer
            reluLayer
            dropoutLayer(dropoutFactor) 
            additionLayer(2,Name="add_"+i)];
    
        % Add and connect layers.
        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"conv1_"+i);
    
        % Skip connection.
        if i == 1
            % Include convolution in first skip connection.
            layer = convolution1dLayer(1,numFilters,Name="convSkip");
    
            lgraph = addLayers(lgraph,layer);
            lgraph = connectLayers(lgraph,outputName,"convSkip");
            lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
        else
            lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
        end
        
        % Update layer output name.
        outputName = "add_" + i;
    end
    
    tempLayers = [
        flattenLayer("Name","flatten")
        fullyConnectedLayer(outdim,"Name","fc_1");
        gruLayer(6,"Name","gru1")];                              % Forward GRU layer 
    lgraph = addLayers(lgraph,tempLayers);
    
    lgraph = connectLayers(lgraph,outputName,"flatten");
    
    
    tempLayers = [                                               % Backward GRU layer
        FlipLayer("flip")
        gruLayer(6,"Name","gru2")];
    lgraph = addLayers(lgraph,tempLayers);
    
    tempLayers = [
        concatenationLayer(1,2,"Name","concat")
        selfAttentionLayer(2,20,"Name","selfattention")          % Multi-head attention
        fullyConnectedLayer(outdim,"Name","fc")
        QRegressionLayer('out', 0.5)];
    lgraph = addLayers(lgraph,tempLayers);
    
    % Connect all branches of the network to create a network graph
    lgraph = connectLayers(lgraph,"fc_1","flip");
    lgraph = connectLayers(lgraph,"gru1","concat/in1");
    lgraph = connectLayers(lgraph,"gru2","concat/in2");    
        
    %%  Network parameter setting
    options = trainingOptions('adam', ...      % Adam
        'MaxEpochs', 10, ...                   % Max traing epochs
        'InitialLearnRate', 1e-2, ...          % Initial learning rate
        'LearnRateSchedule', 'piecewise', ...  % Learning rate drop
        'LearnRateDropFactor', 0.1, ...        % Drop factor
        'LearnRateDropPeriod', 70, ...         % Learning rate drop period
        'Shuffle', 'every-epoch', ...          % Shuffle dataset
        'ValidationPatience', Inf, ...         % Validation disables
        'ExecutionEnvironment','cpu',...       % Execution environment
        'Verbose', true);
    
    %%  Network training
    net = trainNetwork(vp_train, vt_train, lgraph, options);
    save_net = [save_net, net];

    %%  Different forecasting models
    %%  Simulation forecasts
    t1_sim1(jj, :) = predict(save_net(jj), vp_train); 
    t1_sim2(jj, :) = predict(save_net(jj), vp_test ); 

    %%  Format conversion 
    t_sim1(jj, :) = cell2mat(t1_sim1(jj, :));
    t_sim2(jj, :) = cell2mat(t1_sim2(jj, :));
    
    %%  Inverse normalization
    L_sim1{jj} = mapminmax('reverse', t_sim1(jj, :), ps_output);
    L_sim2{jj} = mapminmax('reverse', t_sim2(jj, :), ps_output);
    
    T_sim1(jj, :) = mapminmax('reverse', t_sim1(jj, :), ps_output);
    T_sim2(jj, :) = mapminmax('reverse', t_sim2(jj, :), ps_output);

    
    %%  Plot
    figure
    plot(1 : N, T_test(jj,:), 'r-', 1 : N, T_sim2(jj, :), 'b-', 'LineWidth', 1)
    legend( 'True value', 'Forecast')
    xlabel('Sample')
    ylabel('Result')
    string = {'Comparison'};
    title(string)
    xlim([1, N])
    grid
    set(gcf,'color','w')

    %%  Statistic index
    %  R2
    R2(jj) = 1 - norm(T_test(jj,:)  - T_sim2(jj,:))^2 / norm(T_test(jj,:)  - mean(T_test(jj,:) ))^2;
    disp(['R2=', num2str(R2(jj))])

    %  MAE
    mae2(jj) = sum(abs(T_sim2(jj,:) - T_test(jj,:) )) ./ N ;
    disp(['MAE=', num2str(mae2(jj))])

    % MSE
    MSE2(jj) = sum((T_test(jj,:) - T_sim2(jj,:)).^2)./N;
    disp(['MSE=', num2str(MSE2(jj))])

    %MAPE
    MAPE2(jj) = mean(abs((T_test(jj,:) - T_sim2(jj,:))./T_test(jj,:)));
    disp(['MAPE=', num2str(MAPE2(jj))])
end

m = [T_sim1(2,:); T_sim1(6,:); T_sim1(10,:); T_sim1(14,:); T_sim1(18,:); ...
    T_sim1(3,:); T_sim1(4,:); T_sim1(5,:); ...
    T_sim1(7,:); T_sim1(8,:); T_sim1(9,:); ...
    T_sim1(11,:); T_sim1(12,:); T_sim1(13,:); ...
    T_sim1(15,:); T_sim1(16,:); T_sim1(17,:); ...
    T_sim1(19,:); T_sim1(20,:); T_sim1(21,:)]';
m2 = [T_sim1(1,:); m']';
c = [T_train(3,:); T_train(4,:); T_train(5,:); ...
    T_train(7,:); T_train(8,:); T_train(9,:); ...
    T_train(11,:); T_train(12,:); T_train(13,:); ...
    T_train(15,:); T_train(16,:); T_train(17,:); ...
    T_train(19,:); T_train(20,:); T_train(21,:)]';
c2 = [T_train(2,:); T_train(6,:); T_train(10,:); T_train(14,:); T_train(18,:); c']';

%% Hierarchical reconciliation: Component-Branch
p1 = pinv(m' * m) * m' * c;
s1 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1; ...
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
T_sim2B = [T_sim2(2,:); T_sim2(6,:); T_sim2(10,:); T_sim2(14,:); T_sim2(18,:); ...
    T_sim2(3,:); T_sim2(4,:); T_sim2(5,:); ...
    T_sim2(7,:); T_sim2(8,:); T_sim2(9,:); ...
    T_sim2(11,:); T_sim2(12,:); T_sim2(13,:); ...
    T_sim2(15,:); T_sim2(16,:); T_sim2(17,:); ...
    T_sim2(19,:); T_sim2(20,:); T_sim2(21,:)];
T_sim2C = [T_sim2(2,:)+ T_sim2(6,:)+ T_sim2(10,:)+ T_sim2(14,:)+ T_sim2(18,:); T_sim2B];
yt1B = s1*p1'*T_sim2B;
cB = s1*p1'*m';
yt1 = [yt1B(1,:)+yt1B(2,:)+yt1B(3,:)+yt1B(4,:)+yt1B(5,:); yt1B];

%% Hierarchical reconciliation: Branch-Mainline
p2 = pinv(m2' * m2) * m2' * cB';
s2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
%yt= s2*p2'*yt1;
yt= s2*p2'*T_sim2C;

p = pinv(m2' * m2) * m2' * c;
s = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1; ...
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1; ...
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0; ...
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
yt_1 = s*p'*T_sim2C;

resB18= T_sim2(19, :)+T_sim2(20, :)+T_sim2(21, :);
resB14= T_sim2(15, :)+T_sim2(16, :)+T_sim2(17, :);
resB10= T_sim2(11, :)+T_sim2(12, :)+T_sim2(13, :);
resB6= T_sim2(9, :)+T_sim2(8, :)+T_sim2(7, :);
resB2= T_sim2(3, :)+T_sim2(4, :)+T_sim2(5, :);
resB1= resB2+resB6+resB10+resB14+resB18;

%% A1-Mainline forecasting without reconciliation
%% B1-Mainline forecasting with BU
%% C1-Mainline forecasting with HR
%% D1-Mainline forecasting with HR of Component-Mainline
R2A1 = 1 - norm(T_test(1,:) - T_sim2(1,:))^2 / norm(T_test(1,:)  - mean(T_test(1,:) ))^2
R2B1 = 1 - norm(T_test(1,:) - resB1)^2 / norm(T_test(1,:) - mean(T_test(1,:)))^2
R2C1 = 1 - norm(T_test(1,:) - yt(1, :))^2 / norm(T_test(1,:) - mean(T_test(1,:)))^2
R2D1 = 1 - norm(T_test(1,:) - yt_1(1, :))^2 / norm(T_test(1,:) - mean(T_test(1,:)))^2
maeA1 = sum(abs(T_sim2(1,:) - T_test(1,:) )) ./ N
maeB1 = sum(abs(resB1 - T_test(1,:))) ./ N 
maeC1 = sum(abs(yt(1, :) - T_test(1,:))) ./ N
maeD1 = sum(abs(yt_1(1, :) - T_test(1,:))) ./ N 
MSEA1 = sum((T_test(1,:) - T_sim2(1,:)).^2)./N
MSEB1 = sum((T_test(1,:) - resB1).^2)./N
MSEC1 = sum((T_test(1,:) - yt(1, :)).^2)./N
MSED1 = sum((T_test(1,:) - yt_1(1, :)).^2)./N
MAPEA1 = mean(abs((T_test(1,:) - T_sim2(1,:))./T_sim2(1,:)))
MAPEB1 = mean(abs((T_test(1,:) - resB1)./T_sim2(1,:)))
MAPEC1 = mean(abs((T_test(1,:) - yt(1, :))./T_sim2(1,:)))
MAPED1 = mean(abs((T_test(1,:) - yt_1(1, :))./T_sim2(1,:)))

%% Forecasting results for Branch #1
R2A2 = 1 - norm(T_test(2,:) - T_sim2(2,:))^2 / norm(T_test(2,:)  - mean(T_test(2,:) ))^2
R2B2 = 1 - norm(T_test(2,:) - resB2)^2 / norm(T_test(2,:) - mean(T_test(2,:)))^2
R2C2 = 1 - norm(T_test(2,:) - yt(2, :))^2 / norm(T_test(2,:) - mean(T_test(2,:)))^2
R2D2 = 1 - norm(T_test(2,:) - yt_1(2, :))^2 / norm(T_test(2,:) - mean(T_test(2,:)))^2
maeA2 = sum(abs(T_sim2(2,:) - T_test(2,:) )) ./ N
maeB2 = sum(abs(resB2 - T_test(2,:))) ./ N 
maeC2 = sum(abs(yt(2, :) - T_test(2,:))) ./ N
maeD2 = sum(abs(yt_1(2, :) - T_test(2,:))) ./ N 
MSEA2 = sum((T_test(2,:) - T_sim2(2,:)).^2)./N
MSEB2 = sum((T_test(2,:) - resB2).^2)./N
MSEC2 = sum((T_test(2,:) - yt(2, :)).^2)./N
MSED2 = sum((T_test(2,:) - yt_1(2, :)).^2)./N
MAPEA2 = mean(abs((T_test(2,:) - T_sim2(2,:))./T_sim2(2,:)))
MAPEB2 = mean(abs((T_test(2,:) - resB2)./T_sim2(2,:)))
MAPEC2 = mean(abs((T_test(2,:) - yt(2, :))./T_sim2(2,:)))
MAPED2 = mean(abs((T_test(2,:) - yt_1(2, :))./T_sim2(2,:)))

%% Forecasting results for Branch #2
R2A3 = 1 - norm(T_test(6,:) - T_sim2(6,:))^2 / norm(T_test(6,:)  - mean(T_test(6,:) ))^2
R2B3 = 1 - norm(T_test(6,:) - resB6)^2 / norm(T_test(6,:) - mean(T_test(6,:)))^2
R2C3 = 1 - norm(T_test(6,:) - yt(3, :))^2 / norm(T_test(6,:) - mean(T_test(6,:)))^2
R2D3 = 1 - norm(T_test(6,:) - yt_1(3, :))^2 / norm(T_test(6,:) - mean(T_test(6,:)))^2
maeA3 = sum(abs(T_sim2(6,:) - T_test(6,:) )) ./ N
maeB3 = sum(abs(resB6 - T_test(6,:))) ./ N 
maeC3 = sum(abs(yt(3, :) - T_test(6,:))) ./ N
maeD3 = sum(abs(yt_1(3, :) - T_test(6,:))) ./ N 
MSEA3 = sum((T_test(6,:) - T_sim2(6,:)).^2)./N
MSEB3 = sum((T_test(6,:) - resB6).^2)./N
MSEC3 = sum((T_test(6,:) - yt(3, :)).^2)./N
MSED3 = sum((T_test(6,:) - yt_1(3, :)).^2)./N
MAPEA3 = mean(abs((T_test(6,:) - T_sim2(6,:))./T_sim2(6,:)))
MAPEB3 = mean(abs((T_test(6,:) - resB6)./T_sim2(6,:)))
MAPEC3 = mean(abs((T_test(6,:) - yt(3, :))./T_sim2(6,:)))
MAPED3 = mean(abs((T_test(6,:) - yt_1(3, :))./T_sim2(6,:)))

%% Forecasting results for Branch #3
R2A4 = 1 - norm(T_test(10,:) - T_sim2(10,:))^2 / norm(T_test(10,:)  - mean(T_test(10,:) ))^2
R2B4 = 1 - norm(T_test(10,:) - resB10)^2 / norm(T_test(10,:) - mean(T_test(10,:)))^2
R2C4 = 1 - norm(T_test(10,:) - yt(4, :))^2 / norm(T_test(10,:) - mean(T_test(10,:)))^2
R2D4 = 1 - norm(T_test(10,:) - yt_1(4, :))^2 / norm(T_test(10,:) - mean(T_test(10,:)))^2
maeA4 = sum(abs(T_sim2(10,:) - T_test(10,:) )) ./ N
maeB4 = sum(abs(resB10 - T_test(10,:))) ./ N 
maeC4 = sum(abs(yt(4, :) - T_test(10,:))) ./ N
maeD4 = sum(abs(yt_1(4, :) - T_test(10,:))) ./ N 
MSEA4 = sum((T_test(10,:) - T_sim2(10,:)).^2)./N
MSEB4 = sum((T_test(10,:) - resB10).^2)./N
MSEC4 = sum((T_test(10,:) - yt(4, :)).^2)./N
MSED4 = sum((T_test(10,:) - yt_1(4, :)).^2)./N
MAPEA4 = mean(abs((T_test(10,:) - T_sim2(10,:))./T_sim2(10,:)))
MAPEB4 = mean(abs((T_test(10,:) - resB10)./T_sim2(10,:)))
MAPEC4 = mean(abs((T_test(10,:) - yt(4, :))./T_sim2(10,:)))
MAPED4 = mean(abs((T_test(10,:) - yt_1(4, :))./T_sim2(10,:)))

%% Forecasting results for Branch #4
R2A5 = 1 - norm(T_test(14,:) - T_sim2(14,:))^2 / norm(T_test(14,:)  - mean(T_test(14,:) ))^2
R2B5 = 1 - norm(T_test(14,:) - resB14)^2 / norm(T_test(14,:) - mean(T_test(14,:)))^2
R2C5 = 1 - norm(T_test(14,:) - yt(5, :))^2 / norm(T_test(14,:) - mean(T_test(14,:)))^2
R2D5 = 1 - norm(T_test(14,:) - yt_1(5, :))^2 / norm(T_test(14,:) - mean(T_test(14,:)))^2
maeA5 = sum(abs(T_sim2(14,:) - T_test(14,:) )) ./ N
maeB5 = sum(abs(resB14 - T_test(14,:))) ./ N 
maeC5 = sum(abs(yt(5, :) - T_test(14,:))) ./ N
maeD5 = sum(abs(yt_1(5, :) - T_test(14,:))) ./ N 
MSEA5 = sum((T_test(14,:) - T_sim2(14,:)).^2)./N
MSEB5 = sum((T_test(14,:) - resB14).^2)./N
MSEC5 = sum((T_test(14,:) - yt(5, :)).^2)./N
MSED5 = sum((T_test(14,:) - yt_1(5, :)).^2)./N
MAPEA5 = mean(abs((T_test(14,:) - T_sim2(14,:))./T_sim2(14,:)))
MAPEB5 = mean(abs((T_test(14,:) - resB14)./T_sim2(14,:)))
MAPEC5 = mean(abs((T_test(14,:) - yt(5, :))./T_sim2(14,:)))
MAPED5 = mean(abs((T_test(14,:) - yt_1(5, :))./T_sim2(14,:)))

%% Forecasting results for Branch #5
R2A6 = 1 - norm(T_test(18,:) - T_sim2(18,:))^2 / norm(T_test(18,:)  - mean(T_test(18,:) ))^2
R2B6 = 1 - norm(T_test(18,:) - resB18)^2 / norm(T_test(18,:) - mean(T_test(18,:)))^2
R2C6 = 1 - norm(T_test(18,:) - yt(6, :))^2 / norm(T_test(18,:) - mean(T_test(18,:)))^2
R2D6 = 1 - norm(T_test(18,:) - yt_1(6, :))^2 / norm(T_test(18,:) - mean(T_test(18,:)))^2
maeA6 = sum(abs(T_sim2(18,:) - T_test(18,:) )) ./ N
maeB6 = sum(abs(resB18 - T_test(18,:))) ./ N 
maeC6 = sum(abs(yt(6, :) - T_test(18,:))) ./ N
maeD6 = sum(abs(yt_1(6, :) - T_test(18,:))) ./ N 
MSEA6 = sum((T_test(18,:) - T_sim2(18,:)).^2)./N
MSEB6 = sum((T_test(18,:) - resB18).^2)./N
MSEC6 = sum((T_test(18,:) - yt(6, :)).^2)./N
MSED6 = sum((T_test(18,:) - yt_1(6, :)).^2)./N
MAPEA6 = mean(abs((T_test(18,:) - T_sim2(18,:))./T_sim2(18,:)))
MAPEB6 = mean(abs((T_test(18,:) - resB18)./T_sim2(18,:)))
MAPEC6 = mean(abs((T_test(18,:) - yt(6, :))./T_sim2(18,:)))
MAPED6 = mean(abs((T_test(18,:) - yt_1(6, :))./T_sim2(18,:)))

%% Forecasting errors between levels without reconciliation
aa1_1 = T_sim2(1,:);
aa2_1 = T_sim2(2,:) + T_sim2(6,:) + T_sim2(10,:) + T_sim2(14,:) + T_sim2(18,:);
aa3_1 = T_sim2(3,:) + T_sim2(4,:) + T_sim2(5,:) + ...
        T_sim2(7,:) + T_sim2(8,:) + T_sim2(9,:) + ...
        T_sim2(11,:) + T_sim2(12,:) + T_sim2(13,:) + ...
        T_sim2(15,:) + T_sim2(16,:) + T_sim2(17,:) + ...
        T_sim2(19,:) + T_sim2(20,:) + T_sim2(21,:);
aa2_2 = [T_sim2(2,:), T_sim2(6,:), T_sim2(10,:), T_sim2(14,:), T_sim2(18,:)];
aa3_2 = [T_sim2(3,:) + T_sim2(4,:) + T_sim2(5,:),...
         T_sim2(7,:) + T_sim2(8,:) + T_sim2(9,:), ...
         T_sim2(11,:) + T_sim2(12,:) + T_sim2(13,:), ...
         T_sim2(15,:) + T_sim2(16,:) + T_sim2(17,:), ...
         T_sim2(19,:) + T_sim2(20,:) + T_sim2(21,:)];
R21_2 = 1 - norm(aa1_1 - aa2_1)^2 / norm(aa1_1 - mean(aa1_1))^2
R21_3 = 1 - norm(aa1_1 - aa3_1)^2 / norm(aa1_1 - mean(aa1_1))^2
R22_3 = 1 - norm(aa2_2 - aa3_2)^2 / norm(aa2_2 - mean(aa3_2))^2
mae1_2 = sum(abs(aa1_1 - aa2_1)) ./ N
mae1_3 = sum(abs(aa1_1 - aa3_1)) ./ N 
mae2_3 = sum(abs(aa2_2 - aa3_2)) ./ N
MSE1_2 = sum((aa1_1 - aa2_1).^2)./N
MSE1_3 = sum((aa1_1 - aa3_1).^2)./N
MSE2_3 = sum((aa2_2 - aa3_2).^2)./N
MAPE1_2 = mean(abs((aa1_1 - aa2_1)./aa2_1))
MAPE1_3 = mean(abs((aa1_1 - aa3_1)./aa3_1))
MAPE2_3 = mean(abs((aa2_2 - aa3_2)./aa3_2))