% Code for generating Fig 9: Grand Correlation analysis for EEG
clc
clear
close all

%% Adding the necessary path
addpath('../src/MATLAB_EEG_funcs/')

%% Setting
Signal_list = ["S","Novelty","NPE","RPE","Goal"];
Sensor_region = 'Frontal';

% Window size for moving average
MA_window = round(50/1e3*256);
% Downsampling
DownSample = round(10/1e3*256);
% Range of EEG to analyze (0 to 650 ms)
Beg = round((0 - (-200))/1000*256);
End = round((650 - (-200))/1000*256);

% No ortogonalization and no PCA
Goal_orthogonal = 0;
RPE_orthogonal = 0;
Reward_PCA = 0;

% Blocks to analyze
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
    % normalizing the energy of the EEG signal for each participant
    Specified_Data{Sub}.Y = (Specified_Data{Sub}.Y)/sqrt(mean(Specified_Data{Sub}.Y(:).^2));
end

%% Moving average, downsampling, and normalizing
for Sub = Sub_Set
    Specified_Data{Sub}.Y = movmean(Specified_Data{Sub}.Y,MA_window,2);
    Specified_Data{Sub}.Y = Specified_Data{Sub}.Y(:,Beg:End);
    Specified_Data{Sub}.Y = downsample(Specified_Data{Sub}.Y',DownSample)';
end

%% Concatenating all data together for the grand correlation analysis
Y = Specified_Data{1}.Y;
X = Specified_Data{1}.X;

for Sub=2:10
    Y = [Y; Specified_Data{Sub}.Y];
    X = [X; Specified_Data{Sub}.X];
end

%% Time
time = -198:(1000/256):700;
time = time(Beg:End);
time = downsample(time,DownSample);

%% Correlations All trials
Results = struct();

figure
for Sig_n = 1:5
    Sig_name = Signal_list(Sig_n);
    Results.(Sig_name) = struct();
    
    x = X(:,Sig_n);
    EEG_amp = Y;

    Rho = zeros(size(Y,2),1);
    U_Rho = zeros(size(Y,2),1);
    L_Rho = zeros(size(Y,2),1);
    p_Rho = zeros(size(Y,2),1);
    
    % Computing the correlation for all time points
    for i = 1:size(Y,2)
        if Sig_n ~= 5   % Removing the 1st episode for RPE
            [r,p,rl,ru] = corrcoef(x(x~=0),EEG_amp(x~=0,i));
        else
            [r,p,rl,ru] = corrcoef(x,EEG_amp(:,i));
        end
        Rho(i) = r(1,2);
        U_Rho(i) = ru(1,2);
        L_Rho(i) = rl(1,2);
        p_Rho(i) = p(1,2);
    end
    
    Results.(Sig_name).Rho = Rho;
    Results.(Sig_name).U_Rho = U_Rho;
    Results.(Sig_name).L_Rho = L_Rho;
    Results.(Sig_name).p_Rho = p_Rho;
    
    % Plotting the correlation
    subplot(5,2,1+(Sig_n-1)*2)
    plot(time,U_Rho, '--k')
    hold on
    plot(time,L_Rho, '--k')
    plot(time,Rho, 'LineWidth',2,'Color','k')
    plot([-200,700], [0,0],'--black')
    plot([0,0], [-3,3],'--black')
    ylim([-0.15,0.15])
    xlim([round((Beg/256*1000)-210),round((End/256*1000)-190)])
    title(join([Signal_list(Sig_n) , "- All trials"]))
    
    % Applying Benjamini and Hochberg (1995) algorithm to FDR control
    T = size(Y,2);
    q=0.1; % FDR < 0.1
    p_val_obs = p_Rho';
    p_sort = sort(p_val_obs);
    FDR_line = (1:T)/T*q;
    ind = 1:T;
    FDR_ind = max([1,ind(p_sort<=FDR_line)]);
    p_thresh = FDR_line(FDR_ind);
    Sign_t = p_val_obs<p_thresh;

    plot(time,Sign_t*0.1)
    
    % Plotting the p-values
    subplot(5,2,2+(Sig_n-1)*2)
    y = p_Rho;
    plot(time, log(y), 'LineWidth',2,'Color',[0.8,0.2,0.2])
    hold on
    plot([-200,700], [log(0.05),log(0.05)], '--')
    plot([-200,700], [log(0.01),log(0.01)], '--')
    plot([-200,700], [log(p_thresh),log(p_thresh)], '--')
    xlim([round((Beg/256*1000)-210),round((End/256*1000)-190)])
    legend('pval','0.05','0.01','FDR lim')
    title("Pvalues")
    
    Results.(Sig_name).p_thresh = p_thresh;
    Results.(Sig_name).Sign_t = Sign_t;
end