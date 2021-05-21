clear
close all
clc

%% Loading SPM12 Path
Path = ""; % the path to SPM12 folder should be written here
addpath(Path)

for n_set = 1:3
    %% Load Data
    load(['./nset_', num2str(n_set), '/LogL_Matrix.mat'])
    % MFNS, HybS+U, HybN, SurNoR, Random

    %% Selecting
    Data = sum(sum(LogL_Matrix,2),3);
    Data = permute(Data,[1,4,2,3]);

    %% Save Narrow prior
    [Benchmark_alpha,Benchmark_exp_r,Benchmark_xp,Benchmark_pxp,Benchmark_bor] = spm_BMS(Data(:,1:5),[],[],[],[],ones(1,5)/5);

    BMS = struct();
    BMS.alpha = Benchmark_alpha;
    BMS.exp_r = Benchmark_exp_r;
    BMS.xp = Benchmark_xp;
    BMS.pxp = Benchmark_pxp;
    BMS.bor = Benchmark_bor;

    save(strcat(['./nset_', num2str(n_set), '/BMS_Data.mat']),'BMS')
end