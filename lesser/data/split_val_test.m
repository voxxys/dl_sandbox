%filename = 'sm_ksenia_2';


%filename = 'nikolay_im_2';
%filename = 'alex_long_2';
filename = 'ksenia_long_2';


load(filename);

%

idxs1 = find(states_cur == 1);
idxs2 = find(states_cur == 2);
idxs6 = find(states_cur == 6);

len1 = size(idxs1,2);
len2 = size(idxs2,2);
len6 = size(idxs6,2);

len_val1 = round(len1/3);
len_val2 = round(len2/3);
len_val6 = round(len6/3);

idxs1_val = idxs1(1:len_val1);
idxs1_test = idxs1(len_val1:end);

idxs2_val = idxs2(1:len_val2);
idxs2_test = idxs2(len_val2:end);

idxs6_val = idxs6(1:len_val6);
idxs6_test = idxs6(len_val6:end);

%

data_cur_full = data_cur;
states_cur_full = states_cur;

idxs_val = [idxs1_val,idxs2_val,idxs6_val];
idxs_test = [idxs1_test,idxs2_test,idxs6_test];

data_cur = data_cur_full(:,idxs_val);
states_cur = states_cur_full(:,idxs_val);

disp(['Samples in validation set: ', num2str(size(states_cur,2))]);
figure; plot(states_cur);

save([filename,'val'],'data_cur','states_cur','srate','chan_names');

data_cur = data_cur_full(:,idxs_test);
states_cur = states_cur_full(:,idxs_test);

disp(['Samples in test set: ', num2str(size(states_cur,2))]);
figure; plot(states_cur);

save([filename,'test'],'data_cur','states_cur','srate','chan_names');

