clear
close all
clc

%% Loading SPM12 Path
Path = ""; % the path to SPM12 folder should be written here
addpath(Path)

%% Load Data
load('LogL_Matrix.mat')
% MBS+OI, MBS+U, MB_Leak, MBS, MFQ0, MFS+U, MFN, MFNS, HybS+OI, HybS+U, HybN, SurNoR, Random

%% Selecting
Data = sum(sum(LogL_Matrix,2),3);
Data = permute(Data,[1,4,2,3]);

%% BMS and saving data
[Benchmark_alpha,Benchmark_exp_r,Benchmark_xp,Benchmark_pxp,Benchmark_bor] = spm_BMS(Data(:,1:13),[],[],[],[],ones(1,13)/13);

BMS = struct();
BMS.alpha = Benchmark_alpha;
BMS.exp_r = Benchmark_exp_r;
BMS.xp = Benchmark_xp;
BMS.pxp = Benchmark_pxp;
BMS.bor = Benchmark_bor;

save('BMS_Data.mat','BMS')