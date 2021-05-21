function EEGData = read_EEGData(Sensor_region)

    EEG_data_frame = readtable(['../data/EEG_', Sensor_region, '.csv'],'ReadVariableNames',true);
    EEG_data_frame = table2array(EEG_data_frame);
    % Condition lists:
    % 1.iSub 
    % 2.iBLK 
    % 3. numbef of tot_trial in each block
    % 4. number of trial in each episode 
    % 5. episode                 
    % 6. state
    % 7. stategroup (we don't care)
    % 8:end. -200ms to 700ms, Fs=256Hz


    Sub_set = 1:12;
    Epi_set = 1:5;
    Block_set = 1:2;

    EEGData = cell(length(Sub_set),length(Block_set),length(Epi_set));

    for Sub = Sub_set
        for Block = Block_set
            for Epi = Epi_set
                EEGData{Sub,Block,Epi} = struct();
                EEGData{Sub,Block,Epi}.Sub = Sub;
                EEGData{Sub,Block,Epi}.Block = Block;
                EEGData{Sub,Block,Epi}.Epi = Epi;
                EEGData{Sub,Block,Epi}.Fs = 256;
                EEGData{Sub,Block,Epi}.Start_time = -0.2;
                EEGData{Sub,Block,Epi}.End_time = 0.7;

                ind_select = find(EEG_data_frame(:,1) == Sub & EEG_data_frame(:,2) == (Block+2) & EEG_data_frame(:,5) == Epi);

                EEGData{Sub,Block,Epi}.obs = EEG_data_frame(ind_select,6)-1;
                EEGData{Sub,Block,Epi}.EEG = EEG_data_frame(ind_select,8:end);

            end
        end
    end

end