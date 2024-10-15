function pimwp = PIMWP(T_sim, T_train)

%%  Matrix transpose
if size(T_sim, 1) ~= size(T_train, 1)
    T_sim = T_sim';
end

%%  Interval mean width percentage
pimwp = 1 / length(T_train) * sum((T_sim(:, end) - T_sim(:, 1))...
                                ./ T_train) ;

end