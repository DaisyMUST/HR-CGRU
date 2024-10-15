
classdef QRegressionLayer < nnet.layer.RegressionLayer
    
    % Regression layer
    properties
        tau
    end

    methods

        function layer = QRegressionLayer(name, tau)
            
            % Regression layer construction
            layer.tau = tau;                       % Parameter
            layer.Name = name;                     % Name
            layer.Description = 'quantile error';  % Description

        end
        
        function loss = forwardLoss(layer, Y, T)
            % True T and Forecast Y
            % MAPE calculation
            R = size(Y, 1);
            quantileError = sum(max(layer.tau * (T - Y), (1 - layer.tau) * (Y - T))) / R;  
            N = size(Y, 3);
            loss = sum(quantileError) / N;
        end

        % Loass function
        function dLdY = backwardLoss(layer, Y, T)
           
            dLdY =  single(-layer.tau * (T - Y >= 0) + (1 - layer.tau) * (Y - T >= 0));
                    
        end

    end

end