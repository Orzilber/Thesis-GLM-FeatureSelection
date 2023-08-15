import numpy as np
from numpy.linalg import norm
from numpy.random import multivariate_normal, poisson
from numpy.linalg import inv
from numpy.linalg import multi_dot


#####################
# General Functions #
#####################

def pois_ll(X, y, beta):
    Xb = np.dot(X,beta)
    ll = np.dot(Xb,y) - np.sum(np.exp(Xb))
    return ll

def log_0(y):
    with np.errstate(divide='ignore'):
        res = np.log(y)
    res[np.isneginf(res)]=0
    return res    

def pois_KL(X, y, beta, theta = None):
    Xb = np.dot(X,beta)
    if theta is None:
        kl_1 = np.dot(y, log_0(y)-Xb)
        kl_2 = np.sum(y - np.exp(Xb))
    else:
        kl_1 = np.dot(np.exp(theta), theta - Xb)
        kl_2 = np.sum(np.exp(theta) - np.exp(Xb))
    KL = kl_1-kl_2
    return KL

def nb_ll(X, y, beta, alpha):
    Xb = np.dot(X,beta)
    ll = np.dot(y,Xb) - np.dot(np.ones(y.shape[0]) * alpha + y , np.log(np.exp(Xb) + alpha))
    return np.sum(ll)

def nb_KL(X, y, beta, alpha ,theta = None):
    Xb = np.dot(X,beta)
    if theta is None:
        rel = y
    else:
        rel = theta
    kl_1_par1 = np.log(np.exp(rel) / (np.exp(rel) + alpha))
    kl_1_par2 = np.log(np.exp(Xb) / (np.exp(Xb) + alpha))
    kl_1 = np.dot(exp(rel), kl_1_par1-kl_1_par2)
    
    kl_2 = np.sum(np.log((np.exp(Xb) + alpha) / (np.exp(y) + alpha)) * alpha)
    
    KL = kl_1+kl_2
    return KL


########################
#### IRLS Algorithm ####
########################

def IRLS(X, y, reg_type: ['poisson','nb']
         , alpha = None, threshold = 0.01
         , just_score = True):    
    beta = np.zeros((X.shape[1]))
    #### Distribution specifics ####
    ## Poisson    
    if reg_type == 'poisson':
        ll_cur = pois_ll(X, y, beta)
        W = np.diagflat(np.exp(np.dot(X,beta)))
        D = np.diagflat(np.exp(np.dot(X,beta)))
        z = np.dot(X,beta) + np.dot(inv(D),(y-np.exp(np.dot(X,beta))))
        beta_est = multi_dot([inv(multi_dot([np.transpose(X),W,X])) , np.transpose(X), W, z])
        ll_next = pois_ll(X, y, beta_est)
        
        # IRLS part
        
        while ll_next - ll_cur > threshold:
            ll_cur = pois_ll(X, y, beta_est)
            W = np.diagflat(np.exp(np.dot(X,beta_est)))
            D = np.diagflat(np.exp(np.dot(X,beta_est)))
            z = np.dot(X,beta_est) + np.dot(inv(D),(y-np.exp(np.dot(X,beta_est))))
            beta_est = multi_dot([inv(multi_dot([np.transpose(X),W,X])) , np.transpose(X), W, z])
            ll_next = pois_ll(X, y, beta_est)
    ## NB        
    if reg_type == 'nb':
        ll_cur = nb_ll(X, y, beta, alpha)
        W = np.diagflat(np.exp(np.dot(X,beta))/(1+(alpha**-1) * np.exp(np.dot(X,beta)))) # adjust to nb
        D = np.diagflat(np.exp(np.dot(X,beta)))
        z = np.dot(X,beta) + np.dot(inv(D),(y-np.exp(np.dot(X,beta))))
        beta_est = np.dot(multi_dot([inv(multi_dot([np.transpose(X),W,X])) , np.transpose(X), W]), z)
        ll_next = nb_ll(X, y, beta_est, alpha)
        
        # IRLS part
        
        while ll_next - ll_cur > threshold:
            ll_cur = nb_ll(X, y, beta_est, alpha)
            W = np.diagflat(np.exp(np.dot(X,beta_est))/(1+(alpha**-1) * np.exp(np.dot(X,beta_est)))) # adjust to nb
            D = np.diagflat((np.exp(np.dot(X,beta_est))))
            z = np.dot(X,beta_est) + np.dot(inv(D),(y-np.exp(np.dot(X,beta_est))))
            beta_est = np.dot(multi_dot([inv(multi_dot([np.transpose(X),W,X])) , np.transpose(X), W]), z)
            ll_next = nb_ll(X, y, beta_est, alpha)
    
    if just_score:
        return ll_next
    else:
        return beta_est, ll_next, np.exp(np.dot(X,beta_est))

###########################
#### Forward Algorithm ####
###########################

def fwd(X, y, 
        reg_type: ['poisson','nb'], criteria: ['AIC','BIC','RIC']
        , sel_feat = None, sel_dict = None, alpha = None):
    """
    X, y, 
        reg_type: ['poisson','nb'], criteria: ['AIC','BIC','RIC']
        , sel_feat = None, sel_dict = None, alpha = None
    """
    if sel_feat is None:
        sel_feat = []
    if sel_dict is None:
        sel_dict = {'features':[],'score': np.inf}
#         feat_score =
    for feature in [i for i in range(X.shape[1]) if i not in sel_feat]:
#         print(sel_feat)
#         print(feature)
        in_feat = sel_feat+[feature]
#         print(in_feat)
#         print('currently testing:')
#         print(in_feat)
        # Criteria definition:
        if criteria == 'AIC':
            pen = len(in_feat)
        if criteria == 'BIC':
            pen = 0.5 * np.log(X.shape[0])* len(in_feat)
        if criteria == 'RIC':
            pen = np.log(X.shape[1])* len(in_feat)
        if criteria == 'NLP': #NLP - Non-Linear penalty
            pen = len(in_feat) * np.log(X.shape[1] * np.exp(1) / len(in_feat))
        feat_score_next = -IRLS(X[:,in_feat],y,reg_type,alpha)
        if feat_score_next < sel_dict['score']:
            feat_score = feat_score_next + pen # taking the ll score
            sel_dict['features'] = in_feat
            sel_dict['score'] = feat_score
            sel_dict['-ll']  = feat_score_next
    return sel_dict

######################
### FISTA ###
######################

from numpy import linalg as LA

def pois_nll_grad(X,y,beta):
    Xb = np.dot(X,beta).astype(np.float64)
#     exp_ob = 
    nll_grad = np.dot(np.transpose(X),np.exp(Xb) - y)
    return nll_grad

def nb_nll_grad(X,y,beta,alpha):
    Xb = np.dot(X,beta).astype(np.float64)
    sec_factor = (np.exp(Xb) - y)/(np.exp(Xb) + alpha)
    nll_grad = alpha * np.dot(np.transpose(X),sec_factor)
    return nll_grad

def L_pois(X,y):
    eigs = LA.eigh(np.dot(np.transpose(X),X))[0]
    idx = eigs.argsort()[::-1][0]  
    eig_max = eigs[idx]
    return np.mean(y) * eig_max
    
def L_nb(X,y,alpha):
    eigs = LA.eigh(np.dot(np.transpose(X),X))[0]
    idx = eigs.argsort()[::-1][0]  
    eig_max = eigs[idx]
    return (alpha + np.mean(y))/alpha * eig_max

def prox(grad, beta, L, pen_vec):
    prox_inp = beta - 1/L * grad
    prox_out = np.maximum(np.abs(prox_inp) - pen_vec,0) * np.sign(prox_inp)
    return prox_out

def FISTA(X, y, pen_vec, 
          type: ['poisson', 'nb'],
          iterations = 25, is_ordered = True
          ,alpha = None
          ):
    """
    X - Design matrix
    y - Dependent variables
    pen_vec - The penalty vector
    type - The regression type - Poisson or NB
    Iterations - Number of iterations 
    is_ordered - True
    alpha - Only relevant if we use NB
    """
    beta_start = np.zeros(X.shape[1], dtype = np.float64)
    w_start = np.zeros(X.shape[1], dtype = np.float64)
    delta_start = 1
    for k in range(iterations):
        if type == 'poisson':
            grad = pois_nll_grad(X, y, beta_start)
            L = L_pois(X,y)
        elif type == 'nb':
            grad = nb_nll_grad(X, y, beta_start,alpha)
            L = L_nb(X,y,alpha)
        ### Ordering for the thresholding ###
        
        if is_ordered:

            indx = np.argsort(beta_start)[::-1]

            beta_start = beta_start[indx]
            ind_dict = {i: j for i,j in zip([*range(beta_start.shape[0])], indx)}
        ### Starting the FISTA. Pay attention that we must sort grad according to indx
        
        w_next = prox(grad[indx], beta_start, L, pen_vec)
        delta_next = (1 + np.sqrt(1 + 4*delta_start**2))/2
        beta_next = w_next + ((delta_start - 1)/delta_next)*(w_next - w_start)
        
        ### Re-ordering again for calculating the gradient
        
        beta_start = np.zeros(beta_next.shape[0])
        for origin_ind in ind_dict:
            beta_start[origin_ind] = beta_next[ind_dict[origin_ind]]
        delta_start = delta_next
        w_start = w_next
    
#         if k % 100 == 0:
#             print(k)
#             print('beta_start: {}'.format(beta_start))
#             print('beta_next: {}'.format(beta_next))
#             if type == 'poisson':
#                 print('nll :{}'.format(-pois_ll(X,y,beta_next)))
#             elif type == 'nb':
#                 print('nll :{}'.format(-nb_ll(X,y,beta_next)))
    return beta_next, ind_dict

#####################
# Simulation_runner #
#####################


###############################
# train and simulation runner #
###############################

def main_runner(# runner parameters
                d, d0
                , rho
                , reg_type: ['poisson','nb']
                # matrix simulator parameter
               , sim_num = 1
               # repetitions parameter
               , reps = 13 ):
    results_dict = {(d,d0,rho):[]}
    for i in range(reps):
        print('{} round'.format(i))
        X, y, theta = matrix_simulator(d, d0
                         , rho
                         , beta_set = [0.5, -0.5, 0.6, -0.6]
                         , n = 300
                         , sim_num = 1)
        print('running')
        results_dict[(d,d0,rho)].append(runner(X, y, theta, reg_type))
    return results_dict
        

###########################
# Main Training Function #
###########################

def runner(X, y, theta
           , reg_type: ['poisson','nb']
#            , model_type: ['fwd','LASSO','SLOPE']
           , pen_coef = np.exp(np.arange(-40,40,0.3))):
    
    # Creating train and test sets
    d = X.shape[1]
    (X_train, y_train, theta_train), (X_test, y_test, theta_test) = train_test_allocator(X, y, theta)
    print(X_train.shape)
    print(y_train.shape)
    
    model_scores_dict = {'FWD':{'nll':0,'KL':0, 'KL_theta_train':0, 'KL_theta_test':0, 'size':0}
                         ,'LASSO':{'nll':0,'KL':0, 'KL_theta_train':0, 'KL_theta_test':0, 'size':0}
                         ,'SLOPE':{'nll':0,'KL':0, 'KL_theta_train':0, 'KL_theta_test':0, 'size':0}}
    # Creating k-folds #
    indices = np.array([*range(X_train.shape[0])])
    k_folds_inds = np.split(indices,5)
    
    #####################
    # Forward selection #
    #####################
    print('Starting Forward Selection')
    
#     final_scores = {'AIC': {}, 'BIC': {} ,'RIC': {}}
    folds_score = {'AIC': 0, 'BIC': 0 ,'RIC': 0, 'NLP':0}
    # K-fold part

    for crit in ['AIC','BIC','RIC','NLP']:
        print('using penalty:{}'.format(crit))
        score_start = np.inf
#         folds_score = {'AIC': 0, 'BIC': 0 ,'RIC': 0}
        for oos_fold in k_folds_inds:
            fit_folds = [i for i in indices if i not in oos_fold]
            # val set
            val_set = X_train[oos_fold,:]
            val_y = y_train[oos_fold]
            # fit set
            fit_set = X_train[fit_folds,:]        
            fit_y = y_train[fit_folds]
            sel_dict = fwd(X = fit_set,y = fit_y,reg_type = reg_type, criteria = crit)
            while score_start > sel_dict['score']:
#                 counter +=1
                score_start = sel_dict['score']
                sel_dict = fwd(fit_set,fit_y, reg_type,crit
                               , sel_feat = sel_dict['features'], sel_dict = sel_dict)
            
            selected_features = sel_dict['features']
            beta_est_vals = IRLS(fit_set[:,selected_features], fit_y, reg_type, just_score = False)[0]
            beta_est = np.zeros(d)
            beta_est[selected_features] = beta_est_vals
            print(beta_est)
            if reg_type == 'poisson':
                ll = pois_ll(val_set, val_y, beta_est)
            elif reg_type == 'nb':
                ll = nb_ll(val_set, val_y
                             , beta_est, alpha)
            folds_score[crit] += ll
            print(folds_score)
        folds_score[crit] = folds_score[crit]/5
#         final_scores[crit] = folds_score[crit]
    
    # Taking the best option:
    
    print(sorted(folds_score.items(), key=lambda item: item))
    best_score_crit = sorted(folds_score.items(), key=lambda item: item[1])[0][0]
    print('cv selection: {}'.format(best_score_crit))
    # Fitting over entire train set:
    
    score_start = np.inf
    sel_dict = fwd(X_train, y_train, reg_type, best_score_crit)
#     counter = 0
    while score_start > sel_dict['score']:
#         counter +=1
        score_start = sel_dict['score']
        sel_dict = fwd(X_train, y_train, reg_type, best_score_crit
                       , sel_feat = sel_dict['features'], sel_dict = sel_dict)
    final_feautre_set = sel_dict['features']
    
    final_beta_vals = IRLS(X_train[:,final_feautre_set], y_train, reg_type, just_score = False)[0]
    final_beta_est = np.zeros(d)
    final_beta_est[final_feautre_set] = final_beta_vals
    # Testing over the test set:
    
    if reg_type == 'poisson':
        nll = -pois_ll( X_test, y_test, final_beta_est)
        KL = pois_KL( X_test, y_test, final_beta_est, theta = None)
        KL_theta_train = pois_KL( X_train, y_train, final_beta_est, theta = theta_train)
        KL_theta_test = pois_KL( X_test, y_test, final_beta_est, theta = theta_test)
    elif reg_type == 'nb':
        nll = -nb_ll( X_test, y_test
                     , final_beta_est, alpha)
        KL = nb_KL( X_test, y_test
                     , final_beta_est, alpha, theta = None)
        KL_theta_train = nb_KL( X_train, y_train
                     , final_beta_est, alpha, theta = theta_train)
        KL_theta_test = nb_KL( X_test, y_test
                     , final_beta_est, alpha, theta = theta_test)
    
    model_scores_dict['FWD']['nll'] += nll
    model_scores_dict['FWD']['KL'] += KL
    model_scores_dict['FWD']['KL_theta_train'] += KL_theta_train
    model_scores_dict['FWD']['KL_theta_test'] += KL_theta_test
    model_scores_dict['FWD']['size'] += len(final_feautre_set)
   
    #####################
    ### LASSO & SLOPE ###
    #####################
    
    print("Starting LASSO & SLOPE")
    
#     d = X.shape[1]
    pen_vec_LASSO = np.ones(d) * np.sqrt(2 * np.log(d))
    print(pen_vec_LASSO)
    pen_vec_SLOPE = np.array([np.sqrt(np.log(2*d/(j+1))) for j in range(d)])
    print(pen_vec_SLOPE)
    
    
    pen_coef_results = {'SLOPE':{},'LASSO':{}}
    for C in pen_coef:
        SLOPE_cv_score = 0
        LASSO_cv_score = 0
        print('testing C: {}'.format(C))
        for oos_fold in k_folds_inds:
            fit_folds = [i for i in indices if i not in oos_fold]
            # val set
            val_set = X_train[oos_fold,:]
            val_y = y_train[oos_fold]
            # fit set
            fit_set = X_train[fit_folds,:]        
            fit_y = y_train[fit_folds]
            if reg_type == 'poisson':
                # SLOPE
                SLOPE_cv_beta = FISTA(fit_set,fit_y,C*pen_vec_SLOPE,reg_type)[0]
                SLOPE_cv_score += -pois_ll(val_set, val_y, SLOPE_cv_beta)
                
                # LASSO
                LASSO_cv_beta = FISTA(fit_set,fit_y,C*pen_vec_LASSO,reg_type)[0]
                LASSO_cv_score += -pois_ll(val_set, val_y, LASSO_cv_beta)
                
            elif reg_type == 'nb':
                # SLOPE
                SLOPE_cv_beta = FISTA(fit_set,fit_y,C*pen_vec_SLOPE,reg_type, alpha = alpha)[0]
                SLOPE_cv_score += -nb_ll(val_set, val_y, SLOPE_cv_beta, alpha = alpha)
                # LASSO
                LASSO_cv_beta = FISTA(fit_set,fit_y,C*pen_vec_LASSO,reg_type, alpha = alpha)[0]
                LASSO_cv_score += -nb_ll(val_set, val_y, LASSO_cv_beta, alpha = alpha)
        
        pen_coef_results['SLOPE'][C] = SLOPE_cv_score/5
        pen_coef_results['LASSO'][C] = LASSO_cv_score/5
    
    # Finding the best penalty value
    
    # Eliminating nans #
    
    pen_coef_results['SLOPE'] = {pen: score for pen, score in pen_coef_results['SLOPE'].items() if np.isnan(score) == False}
    pen_coef_results['LASSO'] = {pen: score for pen, score in pen_coef_results['LASSO'].items() if np.isnan(score) == False}
        
    print('SLOPE result')
    print(pen_coef_results['SLOPE'].items())
    print({key: val for key, val in sorted(pen_coef_results['SLOPE'].items(), key=lambda item: item[1])})
    print('LASSO result')
    print(pen_coef_results['LASSO'].items())
    print({key: val for key, val in sorted(pen_coef_results['LASSO'].items(), key=lambda item: item[1])})
    SLOPE_sel_pen = sorted(pen_coef_results['SLOPE'].items(), key=lambda item: item[1])[0][0]
    LASSO_sel_pen = sorted(pen_coef_results['LASSO'].items(), key=lambda item: item[1])[0][0]
    
    
    # Fitting and testing over the best penalty value:
    if reg_type == 'poisson':
#         Fit
        SLOPE_final_beta = FISTA(fit_set,fit_y,SLOPE_sel_pen*pen_vec_SLOPE,reg_type)[0]
        SLOPE_final_beta_size = len(SLOPE_final_beta[SLOPE_final_beta > 0])
        
        LASSO_final_beta = FISTA(fit_set,fit_y,LASSO_sel_pen*pen_vec_LASSO,reg_type)[0]
        LASSO_final_beta_size = len(LASSO_final_beta[LASSO_final_beta > 0])
        
#        eval SLOPE
        SLOPE_nll = -pois_ll( X_test, y_test, SLOPE_final_beta)
        SLOPE_KL = pois_KL( X_test, y_test, SLOPE_final_beta, theta = None)
        SLOPE_KL_theta_train = pois_KL( X_train, y_train, SLOPE_final_beta, theta = theta_train)
        SLOPE_KL_theta_test = pois_KL( X_test, y_test, SLOPE_final_beta, theta = theta_test)
        
#         eval LASSO
        LASSO_nll = -pois_ll( X_test, y_test, LASSO_final_beta)
        LASSO_KL = pois_KL( X_test, y_test, LASSO_final_beta, theta = None)
        LASSO_KL_theta_train = pois_KL(X_train, y_train, LASSO_final_beta, theta = theta_train)
        LASSO_KL_theta_test = pois_KL(X_test, y_test, LASSO_final_beta, theta = theta_test)
    
    elif reg_type == 'nb':
#         Fit
        SLOPE_final_beta = FISTA(fit_set,fit_y,SLOPE_sel_pen*pen_vec_SLOPE,reg_type, alpha = alpha)[0]
        LASSO_final_beta = FISTA(fit_set,fit_y,LASSO_sel_pen*pen_vec_LASSO,reg_type, alpha = alpha)[0]
#        eval
        #        eval SLOPE
        SLOPE_nll = -nb_ll( X_test, y_test, SLOPE_final_beta)
        SLOPE_KL = nb_KL( X_test, y_test, SLOPE_final_beta, theta = None)
        SLOPE_KL_theta_train = nb_KL( X_train, y_train, SLOPE_final_beta, theta = theta_train)
        SLOPE_KL_theta_test = pois_KL( X_test, y_test, SLOPE_final_beta, theta = theta_test)
        
#         eval LASSO
        LASSO_nll = -nb_ll( X_test, y_test, LASSO_final_beta)
        LASSO_KL = nb_KL( X_test, y_test, LASSO_final_beta, theta = None)
        LASSO_KL_theta_train = nb_KL( X_train, y_train, LASSO_final_beta, theta = theta_train)
        LASSO_KL_theta_test = nb_KL( X_test, y_test, LASSO_final_beta, theta = theta_test)
    
    
    model_scores_dict['SLOPE']['nll'] += SLOPE_nll
    model_scores_dict['SLOPE']['KL'] += SLOPE_KL
    model_scores_dict['SLOPE']['KL_theta_train'] += SLOPE_KL_theta_train
    model_scores_dict['SLOPE']['KL_theta_test'] += SLOPE_KL_theta_test
    model_scores_dict['SLOPE']['size'] += SLOPE_final_beta_size
    
    model_scores_dict['LASSO']['nll'] += LASSO_nll
    model_scores_dict['LASSO']['KL'] += LASSO_KL
    model_scores_dict['LASSO']['KL_theta_train'] += LASSO_KL_theta_train
    model_scores_dict['LASSO']['KL_theta_test'] += LASSO_KL_theta_test
    model_scores_dict['LASSO']['size'] += LASSO_final_beta_size
    
    print('Finished all')
    
    return model_scores_dict

def train_test_allocator(X, y, theta, n_test = 100):
    X_inds = [*range(X.shape[0])]
    test_sample = np.random.choice(X_inds, 100,replace = False)
    train_sample = [i for i in X_inds if i not in test_sample]
    train_set = (X[train_sample], y[train_sample], theta[train_sample])
    test_set = (X[test_sample], y[test_sample], theta[test_sample])
    return train_set, test_set

def matrix_simulator(d, d0
                     , rho
                     , beta_set = [0.5, -0.5, 0.6, -0.6]
                     , n = 300
                     , sim_num = 100):
    
    means = np.zeros(d)
    if rho == 0:
        conv_mat = np.diagflat(np.ones(d))
    else:
        conv_mat = cov_creator(d, rho)
    X = multivariate_normal(means, conv_mat, size = n*sim_num)
    X = X/norm(X, axis = 0)
    
    beta = beta_creator(d, d0, beta_set)
    theta = np.dot(X,beta)
    Lambda = np.exp(theta) # This is Lambda, not theta
#     y = poisson.rvs(theta) 
    y = poisson(Lambda)
    return X, y, theta
    
def cov_creator(d, rho):
    arr = np.zeros(d)
    for i in range(d):
        arr[i] = rho**((i+1)-1)
    cov = np.zeros((d,d))
    for i in range(d):
        cov[i,:] = np.concatenate([arr[:i+1][::-1],arr[1:d-i]])
    return cov

def beta_creator(d, d0, beta_set = [0.5, -0.5, 0.6, -0.6]):
    b_d0 = np.random.choice(beta_set, d)
    zero_inds = np.random.choice([*range(d)],d-d0,replace = False)
    b_d0[zero_inds] = 0
    return b_d0

    
# def mp_apply():
# for d in simulation_settings:
#     # Simulate 100 matrices of 300 X d for each d_0
#     # And generate the dpendent variable
#     np.