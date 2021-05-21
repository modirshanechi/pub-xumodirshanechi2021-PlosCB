% Code for generating Fig 10, S6, and S7: EEG regression analyses
clc
clear
close all
warning('off')

% WARNING: to see the results for Fig 10 and S6, put Raw_analysis_S7Fig =
% 0, and to see the results for S7 Fig, put Raw_analysis_S7Fig = 1
Raw_analysis_S7Fig = 0;

%% Adding the necessary path
addpath('../src/MATLAB_EEG_funcs/')

%% Setting
Signal_list = ["S","Novelty","NPE","RPE","Goal"];
Sensor_region = 'Frontal';

% Window size for moving average
MA_window = round(50/1e3*256);
% Downsampling
DownSample = round(10/1e3*256);
% Range of EEG to analyze (-100 to 650 ms)
Beg = round((-98 - (-200))/1000*256);
End = round((650 - (-200))/1000*256);

if Raw_analysis_S7Fig == 0
    Goal_orthogonal = 1;
    RPE_orthogonal = 1;
    Reward_PCA = 1;
else
    Goal_orthogonal = 0;
    RPE_orthogonal = 0;
    Reward_PCA = 0;
end
Block_Set_Setting = 1:2;
Epi_Set_Setting = 1:5;


%% Load model variables
load('../data/ModelData_for_EEG/Trained_Data3.mat')
Model = Data;
clear Data

%% Load EEG data
EEGData = read_EEGData(Sensor_region);

%% Removing subject 1 and 10 from both model-variables and EEG data
Model = Model([2,3,4,5,6,7,8,9,10,12],:,:);
EEGData = EEGData([2,3,4,5,6,7,8,9,11,12],:,:);

%% Extracting the relevant signals from the model
Sub_Set = 1:10;
Epi_Set = 1:5;
Block_Set = 1:2;

Model_pross = cell(10,2,5);
for Sub = Sub_Set
    for Block = Block_Set
        for Epi = Epi_Set
            Model_pross{Sub,Block,Epi} = struct();
            Model_pross{Sub,Block,Epi}.Sub = Model{Sub,Block,Epi}.Sub;
            Model_pross{Sub,Block,Epi}.Epi = Model{Sub,Block,Epi}.Epi;
            Model_pross{Sub,Block,Epi}.Block = Model{Sub,Block,Epi}.Block;
            Model_pross{Sub,Block,Epi}.obs = Model{Sub,Block,Epi}.Input.obs;
            
            Model_pross{Sub,Block,Epi}.Novelty = Model{Sub,Block,Epi}.Novelty_Seq;
            Model_pross{Sub,Block,Epi}.NPE = Model{Sub,Block,Epi}.delta_N_Seq;
            Model_pross{Sub,Block,Epi}.RPE = Model{Sub,Block,Epi}.delta_R_Seq;
            Model_pross{Sub,Block,Epi}.S = Model{Sub,Block,Epi}.Surprise;
            Model_pross{Sub,Block,Epi}.Goal = Model{Sub,Block,Epi}.Input.obs==0;
        end
    end
end

%% Concatenating different block and episodes for each participants 
Specified_Data = cell(10,1);
% EEG Data
for Sub = Sub_Set
    k = 1;
    for Block = Block_Set_Setting
        for Epi = Epi_Set_Setting
            if k==1
                Specified_Data{Sub} = struct();                
                Specified_Data{Sub}.Y = EEGData{Sub,Block,Epi}.EEG;
            else
                Specified_Data{Sub}.Y = [Specified_Data{Sub}.Y; EEGData{Sub,Block,Epi}.EEG];
            end
            k = k+1;
        end
    end
end

for Sub = Sub_Set
    s = 1;
    for Signal = Signal_list
        k = 1;
        for Block = Block_Set_Setting
            for Epi = Epi_Set_Setting
                if k==1
                    x = Model_pross{Sub,Block,Epi}.(Signal);
                else
                    x = [x; Model_pross{Sub,Block,Epi}.(Signal)];
                end
                k = k+1;
            end
        end
        
        if s==1
            Specified_Data{Sub}.X = x;
        else
            Specified_Data{Sub}.X = [Specified_Data{Sub}.X,x];
        end
        s = s+1;
    end
end

%% Processing X and Y
for Sub = Sub_Set
    % excluding the 1st trial of each episode (where RPE, NPE, and Surprise are -1)
    Specified_Data{Sub}.Y = Specified_Data{Sub}.Y((sum(Specified_Data{Sub}.X==-1,2)==0),:);
    Specified_Data{Sub}.X = Specified_Data{Sub}.X((sum(Specified_Data{Sub}.X==-1,2)==0),:);
end

%% Normalizing X
for Sub = Sub_Set
    for i = 1:size(Specified_Data{Sub}.X,2)
        x = Specified_Data{Sub}.X(:,i);
        Specified_Data{Sub}.X(:,i) = (x-mean(x))/std(x);
    end
    Specified_Data{Sub}.X_raw = Specified_Data{Sub}.X;
end
Signal_list_raw = Signal_list;

%% PCA over goal and RPE
inds = 1:size(Specified_Data{Sub}.X,2);
G_i = inds(Signal_list == "Goal");
RPE_i = inds(Signal_list == "RPE");
PC_var = zeros(length(Sub_Set),2);

if Reward_PCA == 1
    for Sub = Sub_Set
        X_pca = Specified_Data{Sub}.X(:,[RPE_i,G_i]);
        Specified_Data{Sub}.X(:,RPE_i) = (X_pca(:,1)+X_pca(:,2))/sqrt(2);
        Specified_Data{Sub}.X(:,G_i) = (-X_pca(:,1)+X_pca(:,2))/sqrt(2);
        PC_var(Sub,1) = var(Specified_Data{Sub}.X(:,RPE_i));
        PC_var(Sub,2) = var(Specified_Data{Sub}.X(:,G_i));
        Signal_list(RPE_i) = "Rp";
        Signal_list(G_i) = "Rn";
    end
end

%% Orthogonalizing on goal
inds = 1:size(Specified_Data{Sub}.X,2);
G_i = inds((Signal_list == "Goal")|(Signal_list == "Rp"));
RPE_i = inds((Signal_list == "Goal")|(Signal_list == "Rn"));

if Goal_orthogonal == 1
    for Sub = Sub_Set
        y = Specified_Data{Sub}.X(:,G_i);
        for i = 1:size(Specified_Data{Sub}.X,2)
            if i ~= G_i
                x = Specified_Data{Sub}.X(:,i);
                C = cov(x,y);
                Specified_Data{Sub}.X(:,i) = x - (C(1,2) / var(y) * y);
            end
        end
    end
end

if RPE_orthogonal == 1
    for Sub = Sub_Set
        y = Specified_Data{Sub}.X(:,RPE_i);
        for i = 1:size(Specified_Data{Sub}.X,2)
            if i ~= RPE_i
                x = Specified_Data{Sub}.X(:,i);
                C = cov(x,y);
                Specified_Data{Sub}.X(:,i) = x - (C(1,2) / var(y) * y);
            end
        end
    end
end

%% Re-normalizing X
for Sub = Sub_Set
    for i = 1:size(Specified_Data{Sub}.X,2)
        x = Specified_Data{Sub}.X(:,i);
        Specified_Data{Sub}.X(:,i) = (x-mean(x))/std(x);
    end
end

%% ------------------------------------------------------------------------
%% S6 Fig
%% Correlation matrix
%% ------------------------------------------------------------------------
Corr_matrix_raw = zeros(length(Signal_list),length(Signal_list),length(Sub_Set));
for Sub = Sub_Set
    Specified_Data{Sub}.Corr_matrix = corr(Specified_Data{Sub}.X_raw);
    Corr_matrix_raw(:,:,Sub) = Specified_Data{Sub}.Corr_matrix;
end

Corr_matrix_projected = zeros(length(Signal_list),length(Signal_list),length(Sub_Set));
for Sub = Sub_Set
    Specified_Data{Sub}.Corr_matrix_projected = corr(Specified_Data{Sub}.X);
    Corr_matrix_projected(:,:,Sub) = Specified_Data{Sub}.Corr_matrix_projected;
end

Corr_matrix_all = zeros(2*length(Signal_list),2*length(Signal_list),length(Sub_Set));
for Sub = Sub_Set
    X_all = [Specified_Data{Sub}.X_raw,Specified_Data{Sub}.X];
    Corr_matrix_all(:,:,Sub) = corr(X_all);
end

%% Correlation matrix plot
figure
r = round(mean(Corr_matrix_all,3),3);
imagesc(r,[-1,1]) 
colormap bone
ax = gca;
ax.YTick = 1:(2*length(Signal_list));
ax.YTickLabel = [Signal_list_raw,Signal_list];
ax.XTick = 1:(2*length(Signal_list));
ax.XTickLabel = [Signal_list_raw,Signal_list];
colorbar
for i = 1:(2*length(Signal_list))
    for j=1:(2*length(Signal_list))
        text(i-0.25,j,num2str(round(r(i,j),2)))
    end
end

title('Pearson Correlation All')

%% ------------------------------------------------------------------------
%% Fig 10 or S7 depending on Raw_analysis_S7Fig value
%% Correlation matrix
%% ------------------------------------------------------------------------
%% Moving average and downsampling
for Sub = Sub_Set
    Specified_Data{Sub}.Y = movmean(Specified_Data{Sub}.Y,MA_window,2);
    Specified_Data{Sub}.Y = Specified_Data{Sub}.Y(:,Beg:End);
    Specified_Data{Sub}.Y = downsample(Specified_Data{Sub}.Y',DownSample)';
end

%% Format of the result
T = size(Specified_Data{Sub}.Y,2);
N = 10;

Result = struct();
for Signal = Signal_list
    Result.(Signal) = struct();
    
    Result.(Signal).Beta = zeros(N,T);
    Result.(Signal).dBeta = zeros(N,T);
    Result.(Signal).T_stat = zeros(N,T);

    Result.(Signal).mean_Beta = zeros(1,T);
    Result.(Signal).std_Beta = zeros(1,T);
    
    Result.(Signal).mean_T = zeros(1,T);
    Result.(Signal).std_T = zeros(1,T);
end

Result.fit = struct();
Result.fit.R2 = zeros(N,T);
Result.fit.R2_adj = zeros(N,T);

Result.fit.mean_R2 = zeros(1,T);
Result.fit.std_R2 = zeros(1,T);

Result.fit.mean_R2_adj = zeros(1,T);
Result.fit.std_R2_adj = zeros(1,T);

%% Regression
for Sub = Sub_Set
    X = Specified_Data{Sub}.X;
    for t = 1:T
        y = Specified_Data{Sub}.Y(:,t);
        y = (y-mean(y))/std(y);
        mdl = fitlm(X,y);
        for i = 1:length(Signal_list)
            Signal = Signal_list(i);
            Result.(Signal).Beta(Sub,t) = mdl.Coefficients.Estimate(i+1);
            Result.(Signal).dBeta(Sub,t) = mdl.Coefficients.SE(i+1);
            Result.(Signal).T_stat(Sub,t) = mdl.Coefficients.tStat(i+1);
        end
        Result.fit.R2(Sub,t) = mdl.Rsquared.Ordinary;
        Result.fit.R2_adj(Sub,t) = mdl.Rsquared.Adjusted;
    end
end

%% FDR test
time = -198:(1000/256):700;
time = time(Beg:End);
time = downsample(time,DownSample);

for Signal = Signal_list
    Result.(Signal).mean_Beta = mean(Result.(Signal).Beta,1);
    Result.(Signal).std_Beta = std(Result.(Signal).Beta,1)/sqrt(10);
    
    Result.(Signal).mean_T = mean(Result.(Signal).T_stat,1);
    Result.(Signal).std_T = std(Result.(Signal).T_stat,1)/sqrt(10);
    
    Result.(Signal).Output_FDR_T = FDR_correction(Result.(Signal).T_stat(:,time>=0),0.1);
    Result.(Signal).Output_FDR_B = FDR_correction(Result.(Signal).Beta(:,time>=0),0.1);
end

Result.fit.mean_R2 = mean(Result.fit.R2,1);
Result.fit.std_R2 = std(Result.fit.R2,1)/sqrt(10);

Result.fit.mean_R2_adj = mean(Result.fit.R2_adj,1);
Result.fit.std_R2_adj = std(Result.fit.R2_adj,1)/sqrt(10);

Result.fit.Output_FDR_R2 = FDR_correction(Result.fit.R2(:,time>=0),0.1);
Result.fit.Output_FDR_R2_adj = FDR_correction(Result.fit.R2_adj(:,time>=0),0.1);

%% ------------------------------------------------------------------------
%% Panel A: R2 Adj.
%% ------------------------------------------------------------------------
time = -198:(1000/256):700;
time = time(Beg:End);
time = downsample(time,DownSample);

time_pval = time(time>=0);

FDR_pval = Result.fit.Output_FDR_R2_adj.p_thresh;
p_obs = Result.fit.Output_FDR_R2_adj.p_val_obs;

figure
subplot(3,1,1)
y = Result.fit.mean_R2_adj;
dy = Result.fit.std_R2_adj;
plot(time, y, 'LineWidth',2,'Color',[0.2,0.2,0.8])
hold on
plot(time, y+dy, 'LineWidth',0.5,'Color',[0.2,0.2,0.8])
plot(time, y-dy, 'LineWidth',0.5,'Color',[0.2,0.2,0.8])
plot([0,0], [-1,1],'--black')
plot([-200,700], [0,0],'--black')
plot(time_pval(p_obs<FDR_pval), zeros(length(time_pval(p_obs<FDR_pval))), '*', 'LineWidth',5,'Color',[0.2,0.8,0.2])
ylim([-0.01,0.03])
xlim([round((Beg/256*1000)-210),round((End/256*1000)-190)])
title(join([Sensor_region,' sensors - R2 Adj.']))

subplot(3,1,2)
y = p_obs;
plot(time_pval, log(y), 'LineWidth',2,'Color',[0.8,0.2,0.2])
hold on
plot([-200,700], [log(FDR_pval),log(FDR_pval)], '--')
plot([-200,700], [log(0.05),log(0.05)], '--')
xlim([round((Beg/256*1000)-210),round((End/256*1000)-190)])
legend('pval','Benj&Hoech threshold','0.05')
title("P-values for R2 Adj.")

Y = Result.fit.R2_adj;
subplot(3,1,3)
plot(time, Y')
hold on
plot([-200,700], [0,0],'--black')
xlim([round((Beg/256*1000)-210),round((End/256*1000)-190)])
plot(time_pval(p_obs<FDR_pval), zeros(length(time_pval(p_obs<FDR_pval))), '*', 'LineWidth',5,'Color',[0.2,0.8,0.2])
title("Participant by participant R2 Adj.")

%% ------------------------------------------------------------------------
%% Panel B
%% ------------------------------------------------------------------------
figure
for i = 1:(length(Signal_list))
    Signal = Signal_list(i);
    y = Result.(Signal).mean_Beta;
    dy = Result.(Signal).std_Beta;
    plot(time, y)
    hold on
end
plot([0,0], [-1,1],'--black')
plot([-200,700], [0,0],'--black')
ylim([-0.1,0.1])
xlim([round((Beg/256*1000)-210),round((End/256*1000)-190)])

FDR_pval = Result.fit.Output_FDR_R2_adj.p_thresh;
p_obs = Result.fit.Output_FDR_R2_adj.p_val_obs;
plot(time_pval(p_obs<FDR_pval), zeros(length(time_pval(p_obs<FDR_pval))), '*', 'LineWidth',5,'Color',[0.2,0.8,0.2])
legend(Signal_list)
title("Average regression coefficients")

disp(max(Result.fit.mean_R2_adj))

%% ------------------------------------------------------------------------
%% Panel C
%% ------------------------------------------------------------------------
%% Significant intervals
FDR_pval = Result.fit.Output_FDR_R2_adj.p_thresh;
p_obs = Result.fit.Output_FDR_R2_adj.p_val_obs;

ds = diff([0,p_obs<FDR_pval,0]);
ind = 1:(length(p_obs)+1);
Beg_ints = ind(ds==1)    + length(time) - length(time_pval);
End_ints = ind(ds==-1)-1 + length(time) - length(time_pval);

% Beg_ints_peak = Beg_ints;
% End_ints_peak = End_ints;

Beg_ints_peak = [26, 34, 43, 48, 55];  % Splitting the last window to 2 parts
End_ints_peak = [26, 37, 47, 52, 59];  % Splitting the last window to 2 parts

%% Intervals informations
T2 = length(Beg_ints_peak);
N = 10;

Results_post = struct();

for Sub = Sub_Set
    for t2 = 1:T2
        for i = 1:length(Signal_list)
            Signal = Signal_list(i);
            Results_post.(Signal).Beta(Sub,t2) = mean(Result.(Signal).Beta(Sub,Beg_ints_peak(t2):End_ints_peak(t2)));
            Results_post.(Signal).dBeta(Sub,t2) = mean(Result.(Signal).dBeta(Sub,Beg_ints_peak(t2):End_ints_peak(t2)));
            Results_post.(Signal).T_stat(Sub,t2) = mean(Result.(Signal).T_stat(Sub,Beg_ints_peak(t2):End_ints_peak(t2)));
            
            y = Result.(Signal).Beta(Sub,Beg_ints_peak(t2):End_ints_peak(t2));
            x = (Beg_ints_peak(t2):End_ints_peak(t2)) - mean(Beg_ints_peak(t2):End_ints_peak(t2));
            mdl = fitlm(x,y);
            if length(y)>1
                Results_post.(Signal).Beta_a0(Sub,t2) = mdl.Coefficients.Estimate(1);
                Results_post.(Signal).Beta_a1(Sub,t2) = mdl.Coefficients.Estimate(2);
            else
                Results_post.(Signal).Beta_a0(Sub,t2) = 0;
                Results_post.(Signal).Beta_a1(Sub,t2) = 0;
            end
        end
        Results_post.fit.R2_adj(Sub,t2) = mean(Result.fit.R2_adj(Sub,Beg_ints_peak(t2):End_ints_peak(t2)));
    end
end

for Signal = Signal_list
    Results_post.(Signal).mean_Beta = mean(Results_post.(Signal).Beta,1);
    Results_post.(Signal).std_Beta = std(Results_post.(Signal).Beta,1)/sqrt(10);
    
    Results_post.(Signal).mean_Beta_a0 = mean(Results_post.(Signal).Beta_a0,1);
    Results_post.(Signal).std_Beta_a0 = std(Results_post.(Signal).Beta_a0,1)/sqrt(10);
    Results_post.(Signal).mean_Beta_a1 = mean(Results_post.(Signal).Beta_a1,1);
    Results_post.(Signal).std_Beta_a1 = std(Results_post.(Signal).Beta_a1,1)/sqrt(10);
    
    Results_post.(Signal).mean_T = mean(Results_post.(Signal).T_stat,1);
    Results_post.(Signal).std_T = std(Results_post.(Signal).T_stat,1)/sqrt(10);
    
    Results_post.(Signal).Output_FDR_T = FDR_correction(Results_post.(Signal).T_stat,0.1);
    Results_post.(Signal).Output_FDR_B = FDR_correction(Results_post.(Signal).Beta,0.1);
    Results_post.(Signal).Output_FDR_B_a0 = FDR_correction(Results_post.(Signal).Beta_a0,0.1);
    Results_post.(Signal).Output_FDR_B_a1 = FDR_correction(Results_post.(Signal).Beta_a1,0.1);
end

%% Bar plot for intervals a0
figure
for t2 = 1:T2
    x = [];
    p = [];
    dx = [];
    for i = 1:length(Signal_list)
        Signal = Signal_list(i);
        
        x = [x, Results_post.(Signal).mean_Beta(t2)];
        dx = [dx, Results_post.(Signal).std_Beta(t2)];
        p = [p,Results_post.(Signal).Output_FDR_B.p_val_obs(t2)];
    end
    subplot(2,T2,t2)
    bar(x)
    hold on
    errorbar(x,dx,'.black')
    xticklabels(Signal_list)
    title('Regression coefficients')
    
    ylim([-0.15,0.15])
    
    subplot(2,T2,T2+t2)
    bar(log(p))
    hold on
    plot([0,7],[1,1]*log(0.1/5*5),'--black')
    plot([0,7],[1,1]*log(0.1/5*4),'--black')
    plot([0,7],[1,1]*log(0.1/5*3),'--black')
    plot([0,7],[1,1]*log(0.1/5*2),'--black')
    plot([0,7],[1,1]*log(0.1/5*1),'--black')
    xticklabels(Signal_list)
    xlim([0,6])
    ylim([-5,0])
    title('P-values with Benj&Hoech thresholds')
end




