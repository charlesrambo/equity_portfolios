# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:56:02 2024

@author: charlesr
"""

# Copied from https://jfin-swufe.springeropen.com/articles/10.1186/s40854-022-00394-x

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import fmin
from scipy.stats import ks_2samp
import warnings


# ---------------------------------------------------------------------------
def prep_x(x, is_price):
    
    if is_price:
        
        y = np.log(x)
    
    else:
        
        y = np.cumsum(x)
    
    return y


# ---------------------------------------------------------------------------
# Define KS-statistic function
def ks_stat(y, q, H, tau, sigma, dist_fun): 
        
    ks_stat = ks_2samp(dist_fun(y, tau, q)/tau**(q * H), 
                           dist_fun(y, sigma, q)/sigma**(q * H))
        
    return ks_stat

    
def ks_test(y, dist_fun, q = 1, sigma_vals = [1, 3]):
    
    # Get N
    N = len(y)
    
    if N < 120:
        
        warnings.warn(f'The KS-method is ineffective for N less than 120. Got N = {N}.')
    
    # Sample more values for small tau
    tau_vals = np.array([int(2**i) for i in range(int(np.log2(N)) - 2)])
       
    # Define objective function; needs to be within local scope
    obj = lambda H: np.nansum([ks_stat(y, q, H, tau, sigma, dist_fun).statistic 
                               for tau in tau_vals for sigma in sigma_vals])
    
    # Minimize obj using simplex method 
    H = fmin(obj, x0 = 0.5, disp = False)[0]
    
    return H


def get_log_reg_coef(tau_vals, mean_vals):
    
    # Perform regression
    reg = LinearRegression().fit(np.log(tau_vals).reshape((-1, 1)), 
                                     np.log(mean_vals))

    return reg.coef_[0] 


def calc_self_similar(y, H, dist_fun, q = 1, sigma_vals = [1, 3]):
    
    N = len(y)
    
    tau_vals = [int(2**i) for i in range(0, int(np.log2(N)) - 2)]
    
    p = np.mean([ks_stat(y, q, H, tau, sigma, dist_fun).pvalue 
                 for tau in tau_vals for sigma in sigma_vals])
    
    return p  


# ---------------------------------------------------------------------------  
def calc_sum(x, tau, q = 1):
        
    return np.abs(x[tau:] - x[:-tau])**q
        

def calc_generalized_hurst_exp(x, q = 1, is_price = False, ks = False, 
                               sigma_vals = [1, 3]):
        
    y = prep_x(x, is_price = is_price)
     
    if ks:
        
        H = ks_test(y, calc_sum, q = q, sigma_vals = sigma_vals)
        
    else:
        
        N = len(y)
    
        tau_vals = np.arange(1, N)
        
        mean_vals = np.array([np.mean(calc_sum(y, tau, q)) for tau in tau_vals])
        
        H = get_log_reg_coef(tau_vals, mean_vals)/q 
    
    return H


# ---------------------------------------------------------------------------  
def calc_area(y, tau, q = 1):
    
    return tau/2 * np.abs(y[2*tau:] - 2 * y[tau:-tau] + y[:-2*tau])**q


def calc_triangle_hurst_exp(x, is_price = False, q = 1, ks = False, 
                            sigma_vals = [1, 3]):
    
    y = prep_x(x, is_price = is_price)
         
    if ks:
    
        c = ks_test(y, calc_area, q = q, sigma_vals = sigma_vals)
        
    else:
        
        N = len(y)
        
        tau_vals = np.arange(1, int(N/2))
        
        area = np.array([np.mean(calc_area(y, tau)) for tau in tau_vals])
        
        c = get_log_reg_coef(tau_vals, area)/q  
    
    return c - 1


# ---------------------------------------------------------------------------  
def calc_m(y, tau, q):
    
    series = pd.Series(y)
    
    f = lambda a: np.max(a - a.iloc[0]) - np.min(a - a.iloc[0])
    
    series = (series.rolling(window = tau + 1).apply(f))**q
    
    return series.dropna().values


def calc_fd_hurst_exp(x, q = 0.01, is_price = False, ks = False, 
                      sigma_vals = [1, 3]):
    
    y = prep_x(x, is_price = is_price)
     
    if ks:
        
        H = ks_test(y, calc_m, q = q, sigma_vals = sigma_vals)
        
    else:
        
        N = len(y)
           
        tau_vals = np.arange(1, N)
        
        m_vals = np.array([np.mean(calc_m(y, tau, q)) for tau in tau_vals])
        
        H = get_log_reg_coef(tau_vals, m_vals)/q 
    
    return H


# Seems to match ellx128 discription in Testing the Algorithms but not formula for FD 
def calc_fd_hl_hurst_exp(x_h, x_l, q = 0.01, is_price = False, ks = False, 
                      sigma_vals = [1, 3]):
    
    y_h = prep_x(x_h, is_price = is_price)
    y_l = prep_x(x_l, is_price = is_price) 
    
    # Calculate difference
    y = y_h - y_l
     
    if ks:
        
        H = ks_test(y, calc_sum, q = q, sigma_vals = sigma_vals)
        
    else:
        
        N = len(y)
    
        tau_vals = np.arange(1, N)
        
        mean_vals = np.array([np.mean(calc_sum(y, tau, q)) for tau in tau_vals])
        
        H = get_log_reg_coef(tau_vals, mean_vals)/q 
    
    return H



if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import time   
    
    start_time = time.perf_counter()
     
    # Create function to report results
    def hurst_results(rho, sigma, n_obs, price = False):
        
        # Shocks are Student's t-distribution so more like market returns
        x = sigma * np.random.standard_t(4.5, size = n_obs + 1)
        
        for j in range(1, n_obs + 1):
                
            x[j] += rho * x[j - 1] 
        
        # Drop the one with no autocorrelation
        x = x[1:]
        
        if price:
            
            x = 50 * np.exp(np.cumsum(x))
        
        gen = calc_generalized_hurst_exp(x, q = 0.01, is_price = price)
        tri = calc_triangle_hurst_exp(x, is_price = price)
        fd = calc_fd_hurst_exp(x, is_price = price)
    
        gen_ks = calc_generalized_hurst_exp(x, q = 0.01, ks = True, is_price = price)
        tri_ks = calc_triangle_hurst_exp(x, ks = True, is_price = price)
        fd_ks = calc_fd_hurst_exp(x, ks = True, is_price = price)
        
        return rho, gen, tri, fd, gen_ks, tri_ks, fd_ks
    
    # Vectorize function
    hurst_results = np.vectorize(hurst_results)
    
    # Define n_obs
    n_obs = 252
      
    # Perform multithreading since slow
    results = pd.DataFrame(np.array(hurst_results(rho = np.linspace(-1, 1, 500),
                                                  sigma = 0.005, n_obs = n_obs)).T)
    
    # Rename columns
    results.columns = ['rho', 'gen', 'tri', 'fd', 'gen_ks', 'tri_ks', 'fd_ks']
    
    # Sort by rho 
    results = results.sort_values(by = 'rho')
        
    # Print graphs so we can so 
    fig, ax = plt.subplots(2, 3, dpi = 300, figsize = (12, 9), sharey = True)
    
    for i, col in enumerate(results.columns[1:]):
        
        nrow, ncol = i //3, i % 3
    
        ax[nrow, ncol].plot(results['rho'], results[col])
        ax[nrow, ncol].axhline(y = 0.5, xmin = 0.0, xmax = 1.0, color = 'red', 
                               linestyle = 'dashed')
        ax[nrow, ncol].axvline(x = 0.0, ymin = 0.0, ymax = 1.0, linewidth=2, 
                               color = 'red', linestyle = 'dashed')
        ax[nrow, ncol].set_title(col)
        ax[nrow, ncol].set_xlabel('rho')
        ax[nrow, ncol].set_ylabel('Hurst Exponent')
    
    fig.suptitle(f'N = {n_obs}')
    
    plt.show()
    
    print(f'{(time.perf_counter() - start_time)/60 : .2f} minutes')