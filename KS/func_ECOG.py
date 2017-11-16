# -*- coding: utf-8 =*=

import sys
from os.path import expanduser
from packages_and_style import *
############


#####################################################
def best_filtration_lfilter(p,y,y_true):
    #bhp,ahp = butter(3,p[0],'high');
    blp,alp = butter(1,p,'low');
    y = lfilter(blp,alp,y);
    return mean_squared_error(y,y_true)

def best_filtration_filtfilt(p,y,y_true):
    #bhp,ahp = butter(2,p[0],'high');
    blp,alp = butter(1,p,'low');
    y = filtfilt(blp,alp,y);
    return mean_squared_error(y,y_true)


def Y_filtration_and_scaling(Y_all):
    bhp,ahp = butter(3,[0.1/1000],'high');
    blp,alp = butter(3,[1/1000],'low');
    Y_all = filtfilt(bhp,ahp,Y_all);
    Y_all = filtfilt(blp,alp,Y_all);
    Y_all=2*((Y_all-min(Y_all))/(max(Y_all)-min(Y_all)))-1
    return Y_all


def X_filtration_and_scaling(X_all, filter_coef=0.006, gaus=False, filt_gaus=0.0008):
    for i in range(4):
        cA, cD = pywt.cwt(X_all[:,i], np.linspace(512, 16, 20),'morl')
        cA=np.transpose(cA)
        if i==0:
            wc_morl=np.copy(cA)
        else:
            wc_morl=np.append(wc_morl,cA, axis=1)

    X_all=abs(scale(wc_morl))
    
    if filter_coef!=False:
        blp,alp = butter(3,[filter_coef],'low');
        wc_morl_filtered=np.transpose(lfilter(blp,alp,np.transpose(abs(scale(wc_morl)))))
        X_all=scale(wc_morl_filtered)        
            
            
    if gaus==True:      
        for i in range(4):
            cA, cD = pywt.cwt(X_all[:,i], np.linspace(512, 16, 20),'gaus1')
            cA=np.transpose(cA)
            if i==0:
                wc_gaus1=np.copy(cA)
            else:
                wc_gaus1=np.append(wc_gaus1,cA, axis=1)
        blp,alp = butter(3,[filt_gaus],'low')
        wc_gaus1_filtered=np.transpose(lfilter(blp,alp,np.transpose(abs(scale(wc_gaus1)))))
        wc_gaus1_scaled=abs(scale(wc_gaus1))
        X_all=np.append(X_all, scale(wc_gaus1_filtered), axis=1)
        

    X_all=scale(X_all)
    return X_all
    
    
    
def results_show(Y_all, Y, Y_val, Y_test, Y_val_and_test, 
            Y_predicted_all, Y_predicted_train, Y_predicted_val, Y_predicted_test, Y_predicted_val_and_test,
           ntr, val, name, filter_coef=None):

    sol_lfilter=optimize.minimize(best_filtration_lfilter,0.001,bounds=((0.0001, 0.004),),
                          args=(Y_predicted_val,Y_val))
    
    if filter_coef!=None:
        sol_lfilter.x[0] =filter_coef


    print ('Lfilter high: ',round(sol_lfilter.x[0],6))
    print ('__________________________________________________________________')
    
    blp,alp = butter(1,sol_lfilter.x[0],'low');
    Y_predicted_lfilter_all = lfilter(blp,alp,Y_predicted_all);

    Y_predicted_lfilter_train=np.copy(Y_predicted_lfilter_all[:ntr])
    Y_predicted_lfilter_val=np.copy(Y_predicted_lfilter_all[ntr:val])
    Y_predicted_lfilter_test=np.copy(Y_predicted_lfilter_all[val:])
    Y_predicted_lfilter_val_and_test=np.copy(Y_predicted_lfilter_all[ntr:])

    #################TRAIN
    MSE_train=mean_squared_error(Y_predicted_train,Y)
    MSE_lfilter_train=mean_squared_error(Y_predicted_lfilter_train,Y)

    corr_train=np.corrcoef(Y_predicted_train,Y)[0,1]
    corr_lfilter_train=np.corrcoef(Y_predicted_lfilter_train,Y)[0,1]
    
    r2_train=r2_score(Y,Y_predicted_train)
    r2_lfilter_train=r2_score(Y,Y_predicted_lfilter_train)

    #################VAL
    MSE_val=mean_squared_error(Y_predicted_val,Y_val)
    MSE_lfilter_val=mean_squared_error(Y_predicted_lfilter_val,Y_val)

    corr_val=np.corrcoef(Y_predicted_val,Y_val)[0,1]
    corr_lfilter_val=np.corrcoef(Y_predicted_lfilter_val,Y_val)[0,1]
    
    r2_val=r2_score(Y_val,Y_predicted_val)
    r2_lfilter_val=r2_score(Y_val,Y_predicted_lfilter_val)

    #################TEST
    MSE_test=mean_squared_error(Y_predicted_test,Y_test)
    MSE_lfilter_test=mean_squared_error(Y_predicted_lfilter_test,Y_test)

    corr_test=np.corrcoef(Y_predicted_test,Y_test)[0,1]
    corr_lfilter_test=np.corrcoef(Y_predicted_lfilter_test,Y_test)[0,1]
    
    r2_test=r2_score(Y_test,Y_predicted_test)
    r2_lfilter_test=r2_score(Y_test, Y_predicted_lfilter_test)

    #################VAL and TEST
    MSE_val_and_test=mean_squared_error(Y_predicted_val_and_test,Y_val_and_test)
    MSE_lfilter_val_and_test=mean_squared_error(Y_predicted_lfilter_val_and_test,Y_val_and_test)

    corr_val_and_test=np.corrcoef(Y_predicted_val_and_test,Y_val_and_test)[0,1]
    corr_lfilter_val_and_test=np.corrcoef(Y_predicted_lfilter_val_and_test,Y_val_and_test)[0,1]
    
    
    r2_val_and_test=r2_score(Y_val_and_test, Y_predicted_val_and_test)
    r2_lfilter_val_and_test=r2_score(Y_val_and_test,Y_predicted_val_and_test)

    ######################
    decim=3
    decim_corr=2
    results=tabulate(

        [['MSE  на обучении', round(MSE_train,decim),  round(MSE_lfilter_train,decim) ], 
         ['Corr на обучении', round(corr_train,decim_corr),  round(corr_lfilter_train,decim_corr)],
         ['R_sq на обучении', round(r2_train,decim_corr),  round(r2_lfilter_train,decim_corr)],

         ['',''],

         ['MSE  на валидации', round(MSE_val,decim),  round(MSE_lfilter_val,decim)], 
         ['Corr на валидации', round(corr_val,decim_corr),  round(corr_lfilter_val,decim_corr)],
         ['R_sq на валидации', round(r2_val,decim_corr),  round(r2_lfilter_val,decim_corr)],

         ['',''],

         ['MSE  на тесте', round(MSE_test,decim),  round(MSE_lfilter_test,decim)], 
         ['Corr на тесте', round(corr_test,decim_corr), round(corr_lfilter_test,decim_corr)],
         ['R_sq на тесте', round(r2_test,decim_corr),  round(r2_lfilter_test,decim_corr)],

         ['',''],

         ['MSE  на вал и тесте', round(MSE_val_and_test,decim),  round(MSE_lfilter_val_and_test,decim)], 
         ['Corr на вал и тесте', round(corr_val_and_test,decim_corr),  round(corr_lfilter_val_and_test,decim_corr)],
         ['R_sq на вал и тесте', round(r2_val_and_test,decim_corr),  round(r2_lfilter_val_and_test,decim_corr)],
        ],

        headers=['', 'Raw','lfilter' ])

    print (results)
    print ('')

    fig = plt.figure()
    plt.plot(range(0, len(Y)), Y_predicted_train, linewidth=0.2, c="black", )
    plt.plot(range(0, len(Y)), Y_predicted_lfilter_train, linewidth=3, c="blue")
    plt.plot(range(0, len(Y)), Y, linewidth=4, c="red")
    plt.legend(['prediction', 'lfilter', 'real'], loc='upper left')
    plt.title('Train')
    plt.show()


    fig = plt.figure()
    plt.plot(range(0, len(Y_val)), Y_predicted_val, linewidth=0.2, c="black")
    plt.plot(range(0, len(Y_val)), Y_predicted_lfilter_val, linewidth=3, c="blue")
    plt.plot(range(0, len(Y_val)), Y_val, linewidth=4, c="red")
    plt.legend(['prediction', 'lfilter', 'real'], loc='upper left')
    plt.title('Validation')
    plt.show()


    fig = plt.figure()
    plt.plot(range(0, len(Y_test)), Y_predicted_test, linewidth=0.2, c="black")
    plt.plot(range(0, len(Y_test)), Y_predicted_lfilter_test, linewidth=3, c="blue")
    plt.plot(range(0, len(Y_test)), Y_test, linewidth=4, c="red")
    plt.legend(['prediction', 'lfilter', 'real'], loc='upper left')
    plt.title('Test')
    plt.show()


    fig = plt.figure()
    plt.plot(range(0, len(Y_all)), Y_predicted_all, linewidth=0.2, c="black")
    plt.plot(range(0, len(Y_all)), Y_predicted_lfilter_all, linewidth=3, c="blue")
    plt.plot(range(0, len(Y_all)), Y_all, linewidth=4, c="red")
    plt.legend(['prediction','lfilter', 'real'], loc='upper left')
    plt.title('All')
    plt.show()

    
random_seed=1
kf = KFold(n_splits=3, shuffle=False, random_state=random_seed)
cpu_used=1

