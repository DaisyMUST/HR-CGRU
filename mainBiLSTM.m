%% Initialization
warning off; close all; clear; clc

%%  Data input
res = xlsread('ExperimentalData.xlsx');
%% SSA decomposation if the preprocessing results are not included in the data input
%res = xlsread('NorthLineData.xlsx');
%res = res(:,1);
%[LT,ST,R] = trenddecomp(res,"ssa",7*24);
Lag = 47; %% Day-ahead is 24 hours, and the input variables are x, x-1, ...,x-23 h. Lag is 47 in total
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
    P_train = res(1: num_train_s, 1: f_)';
    T_train(jj,:) = res(1: num_train_s, f_ + 1: end)';
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
    layers = [
        sequenceInputLayer(f_,"Name","sequence");
        bilstmLayer(16,"Name","bilstm");
        fullyConnectedLayer(outdim,"Name","fc")
        QRegressionLayer('out', 0.5)];
    
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
    net = trainNetwork(vp_train, vt_train, layers, options);
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
    %for jj=1:21
    MAPE2(jj) = mean(abs((T_test(jj,:) - T_sim2(jj,:))./T_sim2(jj,:)));
    %end
    disp(['MAPE=', num2str(MAPE2(jj))])
end

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
