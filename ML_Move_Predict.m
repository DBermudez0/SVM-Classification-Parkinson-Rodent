function [error_fit] = ML_Move_Predict(record_elec_activ_file, BORIS_file,f_low, f_high, time_inter, Fs, min_ang_vel, max_ang_vel, radii_LH, C_waist)

    % Author: Dalton H. Bermudez 
    % Collaboration with Heysol C. Bermudez, who
    % provided the data parameters for the designed algorithm. 
    % Parameters of interest were chosen from various discussions.
    %
    % This is an add on script to a previous developed software called BORIS
    % for the analysis of behavior.ex
    %
    % record_elec_activ_file - recorded files of electrical activity (must be in .mat file)
    % BORIS_file - labeled BORIS files of behavior of interest.
    % f_low - low frequency of interest.
    % f_high - high frequency of interest.
    % time_inter - time interval (in seconds) from time of labeled observed motor behavior of interest to time prior or after observed motor behavior.
    % (time interval) - is going to be the delta time used to calculate the power rate.
    % Fs - sampled frequency used to record the neural activity from electrode of the rodent.
    % min_ang_vel - minimal angular velocity (in rotations/minute) of the spining cylindrical devise use to record motion of rodent.
    % max_ang_vel - maximum angular velocity (in rotations/minute) of the spining cylindrical devise use to record motion of rodent.
    % radii_LH - measured radial distance from center of spinning cylindrical devise to the hind paw located closest to the center.
    % C_waist - measured waist of the rodent as close to the hind or front paws of interest.
    %
    %
    % Angular velocity for a subjected Parkinsonian rodent walking about a rotating cylinder with respect to its central axis.
    % Every point on the projected circular rotating disk has the same angular velocity.
    %
    % This script performs a binary classification from features of arc
    % displacement and power rate between the impaiered paw and
    % non-impaiered paw of the Parkinsonian rodent.

min_ang_vel_con = (min_ang_vel*2*pi)/60;% units converted from rotation per min to radiants per sec
max_ang_vel_con = (max_ang_vel*2*pi)/60;% units converted from rotation per min to radiants per sec

% calculate the average angular velocity 
ang_vel = mean([min_ang_vel_con max_ang_vel_con]);

% radii_LH = 5/2;% units in centimeter (with respect to the center) for distance of Hind Paw closes to the center-provided by Heysol C. Bermudez
% C_waist = 20; % units in centimeter of the waist can be approximated as the circumference of a circle - value of waist of rat provided by Heysol C. Bermudez
radii_RH = radii_LH + (C_waist/pi);% units in centimeter from the center of the spinning cylinder to the outer most Hind Paw position (with respect to the center)

filttype = 'bandpassfir';

% data were provided and label by Heysol C. Bermudez
HC4 = load(record_elec_activ_file);
f_name_Ch = fieldnames(HC4);

    for i = 1:length(f_name_Ch)
        disp(['[' num2str(i) '] ' f_name_Ch{i}])
    end
    
reply_Channel = input(['Which channel do you want to analyze (pick a number between 1 to ' num2str(length(f_name_Ch)) '):'],'s');
HC4_data = importBorisData(BORIS_file);

f_name_label = fieldnames(HC4_data);

    for i = 1:length(f_name_label)
        disp(['[' num2str(i) '] ' f_name_label{i}])
    end
    
reply_label_1 = input(['Which label did you use for the impaired Paw (pick a number between 1 to ' num2str(length(f_name_label)) '):'],'s');
disp('Need to classes for classification (Pick another label)')
reply_label_2 = input(['Which label do you use for non-impaired Paw (pick a number between 1 to ' num2str(length(f_name_label)) '):'],'s');

% first class is going to be impaired Paw
post_second_class = HC4_data.(f_name_label{str2double(reply_label_2)}).*Fs;% change the label variable of interest as necessary
post_first_class = HC4_data.(f_name_label{str2double(reply_label_1)}).*Fs;% change the label variable of interest as necessary
sample_epoch = time_inter*Fs;


design_filt = designfilt(filttype, 'FilterOrder', 100, 'CutoffFrequency1', f_low, 'CutoffFrequency2', f_high, 'SampleRate', Fs);
filt_signal = filtfilt(design_filt,double(HC4.(f_name_Ch{str2double(reply_Channel)}).values));


% filter signal segment for 250 ms prior and after to LHu: power rate calculation
for i = 1:length(post_second_class)
    
    CH4_signal_prior(1:sample_epoch) = filt_signal(int32(post_second_class(i))+2-sample_epoch:int32(post_second_class(i))+1);
    CH4_signal_after(1:sample_epoch) = filt_signal(int32(post_second_class(i))+1:int32(post_second_class(i))+sample_epoch);
    WT_prior = cwt(CH4_signal_prior, Fs);
    WT_after = cwt(CH4_signal_after, Fs);
    
    power_1_prior = sqrt(real(WT_prior(:,1)).^2 + imag(WT_prior(:,1)).^2);
    power_2_prior = sqrt(real(WT_prior(:,end)).^2 + imag(WT_prior(:,end)).^2);
    power_1_after = sqrt(real(WT_after(:,1)).^2 + imag(WT_after(:,1)).^2);
    power_2_after = sqrt(real(WT_after(:,end)).^2 + imag(WT_after(:,end)).^2);
    
    % Calculate the Power Rate of the signal from Left Hind Paw (UP) for
    % time_inter prior and after labeled Left Hind Paw (UP) behavior
    % Calculated by summing the power intensity of all the pixel allong the
    % time that is time_interval before or prior the labeled time point of top up. 
    power_rate_LHu_prior(i) = (sum(power_2_prior) - sum(power_1_prior))/time_inter;
    power_rate_LHu_after(i) = (sum(power_2_after) - sum(power_1_after))/time_inter;
end

% filter signal segment for 250 ms prior to RHu: power rate calculation
for j = 1:length(post_first_class)
    
    CH4_signal_prior(1:sample_epoch) = filt_signal(int32(post_first_class(j))+2-sample_epoch:int32(post_first_class(j))+1);
    CH4_signal_after(1:sample_epoch) = filt_signal(int32(post_first_class(j))+1:int32(post_first_class(j))+sample_epoch);
    % Compute the Wavelet Transform of filtered signal
    WT_prior = cwt(CH4_signal_prior, Fs);
    WT_after = cwt(CH4_signal_after, Fs);
    
    power_1_prior = sqrt(real(WT_prior(:,1)).^2 + imag(WT_prior(:,1)).^2);
    power_2_prior = sqrt(real(WT_prior(:,end)).^2 + imag(WT_prior(:,end)).^2);
    power_1_after = sqrt(real(WT_after(:,1)).^2 + imag(WT_after(:,1)).^2);
    power_2_after = sqrt(real(WT_after(:,end)).^2 + imag(WT_after(:,end)).^2);
    
    % Calculate the Power Rate of the signal from Right Hind Paw (UP) for
    % time_inter prior and after labeled Right Hind Paw (UP) behavior
    power_rate_RHu_prior(j) = (sum(power_2_prior) - sum(power_1_prior))/time_inter;   
    power_rate_RHu_after(j) = (sum(power_2_after) - sum(power_1_after))/time_inter;
end

% Mark location where power rate either increases or decreases for 250ms
% after RHu (Affected Limb for this experiment)
RHu_loc_dec = (power_rate_RHu_after(:)< 0);
RHu_loc_incr = (~RHu_loc_dec);

% Mark location where power rate either increases or decreases for 250ms
% after LHu (Non-Affected Limb for this experiment)
LHu_loc_dec = (power_rate_LHu_after(:)< 0);
LHu_loc_incr = (~LHu_loc_dec);


% calculate the time to go from LHu to LHd and discriminate between the
% times were the calculated power rates are decreasing from does that are
% increasing
    for i = 1:length(f_name_label)
        disp(['[' num2str(i) '] ' f_name_label{i}])
    end
    
    % impaired paw
    label_impaired_up = input(['Which label did you use for the impaired Paw Toe Up (pick a number between 1 to ' num2str(length(f_name_label)) '):'],'s');
    disp('Need both toe down and town up for impaired Paw to calculate time step duration')
    label_impaired_down = input(['Which label do you use for the impaired Paw Toe Down (pick a number between 1 to ' num2str(length(f_name_label)) '):'],'s');
    % non-impaiered paw
    label_nonimpaired_up = input(['Which label did you use for the non impaired Paw Toe Up (pick a number between 1 to ' num2str(length(f_name_label)) '):'],'s');
    disp('Need both toe down and town up for impaired Paw to calculate time step duration')
    label_nonimpaired_down = input(['Which label do you use for the non impaired Paw Toe Down (pick a number between 1 to ' num2str(length(f_name_label)) '):'],'s');
% delta times for when there is an increase in power rate
delta_time_incr_LH = HC4_data.(f_name_label{str2double(label_nonimpaired_down)})(LHu_loc_incr) - HC4_data.(f_name_label{str2double(label_nonimpaired_up)})(LHu_loc_incr);
avr_time_step_LH_incr = mean(delta_time_incr_LH);
% Arc_dist_step_incr = (max_ang_vel/radii_LH)*avr_time_step_LH_incr;

% Delta Times for when there is an decrease in power rate
delta_time_dec_LH = HC4_data.(f_name_label{str2double(label_nonimpaired_down)})(LHu_loc_dec) - HC4_data.(f_name_label{str2double(label_nonimpaired_up)})(LHu_loc_dec);
avr_time_step_LH_dec = mean(delta_time_dec_LH);

delta_time_LH = [delta_time_incr_LH; delta_time_dec_LH];

% calculate the arc distance from LHu to LHd when power rate increase
Arc_dist_step_LHincr = (ang_vel*radii_LH)*avr_time_step_LH_incr;% units in centimeters

% calculate the arc distance from RHu to RHd when power rate decrease
Arc_dist_step_LHdec = (ang_vel*radii_LH)*avr_time_step_LH_dec;% units in centimeters

% average distance of step for the LH
Arc_mean_dist_LH = mean([Arc_dist_step_LHincr Arc_dist_step_LHdec]);

% calculate the time to go from RHu to RHd and discriminate between the
% times were the calculated power rates are decreasing from does that are
% increasing

% delta times for when there is an increase in power rate
delta_time_incr_RH = HC4_data.(f_name_label{str2double(label_impaired_down)})(RHu_loc_incr) - HC4_data.(f_name_label{str2double(label_impaired_up)})(RHu_loc_incr);
avr_time_step_RH_incr = mean(delta_time_incr_RH);
% delta times for when there is an decease in power rate
delta_time_dec_RH = HC4_data.(f_name_label{str2double(label_impaired_down)})(RHu_loc_dec) - HC4_data.(f_name_label{str2double(label_impaired_up)})(RHu_loc_dec);
avr_time_step_RH_dec = mean(delta_time_dec_RH);

delta_time_RH = [delta_time_incr_RH; delta_time_dec_RH];

% calculate the arc distance from LHu to LHd when power rate increase
Arc_dist_step_RHincr = (ang_vel*radii_RH)*avr_time_step_RH_incr;% units in centimeters

% calculate the arc distance from RHu to RHd when power rate decrease
Arc_dist_step_RHdec = (ang_vel*radii_RH)*avr_time_step_RH_dec;% units in centimeters

% average distance of step for the RH used to take into account for the
% difference in linear velocities as a function of radious of the spinning
% cylindrical radius
Arc_mean_dist_RH = mean([Arc_dist_step_RHincr Arc_dist_step_RHdec]);

% features used for classification will be the power rate obtained from
% 250ms prior to RH-toff/LH-toff and displacement from time of
% RH-toff/LH-toff to RH-td/LH-td.
% classification between RH-0 and LH-1
Arc_dist_RH = (ang_vel*radii_RH).* delta_time_RH;
Arc_dist_LH = (ang_vel*radii_LH).* delta_time_LH;
% 2 different features used for the classification
features_for_RH = [power_rate_RHu_prior; Arc_dist_RH'];
features_for_LH = [power_rate_LHu_prior; Arc_dist_LH'];

features = [features_for_RH,features_for_LH];
features_f = features';
binary = [zeros([1, size(features_for_RH,2)]), ones([1,size(features_for_LH,2)])];
binary_f = binary';
Cross_Val = cvpartition(binary,'k',10);
for i=1:Cross_Val.NumTestSets
    
    Train_idx = Cross_Val.training(i);
    Test_idx = Cross_Val.test(i);
    
    SVMModel = fitcsvm(features_f(Train_idx,:),binary_f(Train_idx,:),'KernelFunction','linear','Standardize', true);% 'OptimizeHyperparameters' , 'all'
    CompactSVMModel = compact(SVMModel);
    [ScoreCSVMModel, ~] = fitPosterior(CompactSVMModel,...
    features_f(Test_idx,:),binary_f(Test_idx));
    [labels,postProbs] = predict(ScoreCSVMModel,features_f(Test_idx,:));
    table(binary_f(Test_idx),labels,postProbs(:,2),...
    'VariableNames',{'TrueLabel','PredictedLabel','PosteriorProbability'})
    error_fit(i) = sum(abs(labels-binary_f(Test_idx)))./size(labels,1);
end
