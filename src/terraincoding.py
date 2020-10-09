import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Scikit imports
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

#Importt from the other files
from utils import *
from assessment import *
from regression_methods import *
from secondary_utils_changed_functions import *

"""Showing and choosing the terrain"""
terrain = imread("./data/SRTM_data_Norway_1.tif")
# Plot the terrain
print(type(terrain), len(terrain))
plt.figure()
plt.title("Terrain over Norway")
plt.imshow(terrain, cmap="bone")
plt.xlabel("X")
plt.ylabel("Y")
#plt.savefig('Terrain_SRTM_data_norway_1')
plt.show()

#Chose a random part of the terrain
part_of_terrain = terrain[1510:1530,1010:1030]

# Plot part of the terrain
plt.figure()
plt.title('Small part of terrain over Norway')
plt.imshow(part_of_terrain, cmap='bone', extent=[1010,1030,1530,1510])
#Other cool colors: copper, winter, gray, pink
plt.xticks(rotation=45)
plt.xlabel('X')
plt.ylabel('Y')
#plt.savefig('part_of_terrain')
plt.show()

"""Defining the data the same way as we did in franke's function"""
y_arr, x_arr = part_of_terrain.shape
#Fixing the axes
x = np.arange(x_arr)
y = np.arange(y_arr)
x, y = np.meshgrid(x/(np.amax(x)), y/(np.amax(y)))
#Ravel the arrays to make them easier to work with
x1 = np.ravel(x)
y1 = np.ravel(y)
z1 = np.ravel(part_of_terrain)
#Defining the amount of observations
n = len(z1)


def OLS_terrain(*args):
    #Finding the best degree of polynomial to use, check up to 30
    maxd = 30
    interval_degrees = np.arange(1, maxd+1)

    #Defining empty lists for the errors and r scores
    errors_test = []
    errors_train = []
    r2_list = []

    if 'best_deg' in args :
        for deg in interval_degrees:
            test_error, train_error, r2 = k_Cross_Validation(x1, y1, z1, d=deg)
            errors_test.append(test_error)
            errors_train.append(train_error)
            r2_list.append(r2)

        # Plotting test and train errors to compare
        plt.figure()
        plt.plot(interval_degrees, errors_test, label='Test MSE')
        plt.plot(interval_degrees, errors_train, label='Train MSE')
        plt.legend()
        plt.xlabel('Degree of polynomial')
        plt.ylabel('Error')
        if 'save' in args :
            plt.savefig('terrain_ols_error_plot')
        #plt.title('Training and test error vs. polynomial degree')
        plt.show()

        #Checks which degree is best
        k=1000
        for i in range(0,len(interval_degrees)-1):
            temp=abs(errors_test[i]-errors_train[i])
            if temp<k:
                k=temp
                best_deg=i

        print(f"The best degree to use further is d={best_deg}")

    """Performing OLS with the best degree"""
    if 'OLS_plot' and not 'best_deg' in args:
        best_deg=13
    elif 'OLS_plot' in args:
        X=design_matrix(x1, y1, best_deg)

        #Couldn't call class, got assertion error for some reason
        #OLS_reg=OLS(X,z1)
        #beta=OLS_reg.fit_beta()
        #z_pred=OLS_reg.predict()

        #OLS method by using inverse function from lecture slides
        A=X.T @ X
        beta=SVDinv(A)@X.T@z1
        #beta=np.linalg.pinv(X.T@X)@X.T@z1
        z_pred=X@beta

        #Plotting the OLS terrain result with the best degree
        plt.figure()
        #plt.title("OLS")
        plt.imshow(z_pred.reshape(y_arr, x_arr), cmap='bone', extent=[1010,1030,1530,1510])
        plt.xticks(rotation=45)
        plt.xlabel('X')
        plt.ylabel('Y')
        if 'save' in args :
            plt.savefig('Terrain_OLS_bestdegree')
        plt.show()

        print(f"OLS-Mean Squared Error: {MSE(z1,z_pred)}")
        print(f"OLS-R2-score: {R2(z1,z_pred)}")

    """Calculating and plotting the confidence intervals"""
    if 'CI' in args:
        beta_x = np.arange(len(beta))
        beta_lower, beta_upper =CI(X, beta, 95, z1, z_pred, n)

        #Plotting the confidence intervals for OLS terrain
        plt.plot(beta_x, beta, 'mo', label='betas')
        plt.plot(beta_x, beta_upper, 'k+')
        plt.plot(beta_x, beta_lower, 'k+')
        plt.xlabel('Betas')
        if 'save' in args :
            plt.savefig('Terrain_OLS_CI')
        plt.show()

    return
def ridge(*args):
    """Finding the best degree of polynomial and lambda"""
    #Defining max degree and max lambda to check
    maxd = 30       #30
    n_lmb = 35      #35

    #Defining ranges to loop over
    lambdas = np.logspace(-12,0, n_lmb)
    interval_degrees = np.arange(1, maxd+1)

    mse_values, best_deg, best_lmb, min_MSE=best_d_l(maxd, n_lmb, interval_degrees, lambdas, x1, y1, z1, 'Ridge')

    print(f"Best MSE while finding best parameters: {min_MSE}")
    print(f"with lambda={best_lmb}, and degree={best_deg}")

    # Plot MSE with color map
    im = plt.imshow(mse_values, cmap=plt.cm.RdBu, extent = [-12, 0, 1, maxd],
                interpolation=None, aspect='auto')
    plt.colorbar(im)
    plt.xlabel('log10(lambda)')
    plt.ylabel('degree of polynomial')
    #plt.title('MSE colormap (Ridge)')
    if 'save' in args :
        plt.savefig('terrain-ridge-degree-lambda-colormap')
    plt.show()

    """Performing Ridge with the best degree and best lambda"""
    X = design_matrix(x1, y1, best_deg)
    p = X.shape[1]
    I = np.eye(p)

    A=(X.T @ X) + best_lmb*I
    beta=SVDinv(A) @ X.T @ z1
    z_pred=X@beta

    #Plotting the ridge terrain result with the best degree
    plt.figure()
    plt.imshow(z_pred.reshape(y_arr, x_arr), cmap='bone', extent=[1010,1030,1530,1510])
    plt.xticks(rotation=45)
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.title('Ridge regression')
    if 'save' in args :
        plt.savefig('Terrain_ridge_bestdegree')
    plt.show()

    print(f"Ridge-Mean Squared Error: {MSE(z1,z_pred)}")
    print(f"Ridge-R2-score: {R2(z1,z_pred)}")

    """Calculating and plotting the confidence intervals"""
    beta_x = np.arange(len(beta))
    beta_lower, beta_upper =CI(X, beta, 95, z1, z_pred, n)

    #Plotting the confidence intervals for OLS terrain
    plt.plot(beta_x, beta, 'mo', label='betas')
    plt.plot(beta_x, beta_upper, 'k+')
    plt.plot(beta_x, beta_lower, 'k+')
    plt.xlabel('Betas')
    if 'save' in args :
        plt.savefig('Terrain_ridge_CI')
    plt.show()

    return
def lasso(*args):
    """Finding the best degree of polynomial and lambda"""
    #Defining max degree and max lambda to check
    maxd = 50   #30 for ridge
    n_lmb = 50  #35 for ridge

    #Defining ranges to loop over
    lambdas = np.logspace(-10,0, n_lmb)
    interval_degrees = np.arange(1, maxd+1)

    #Returns the best lambdas and degrees, along with some MSE values
    mse_values, best_deg, best_lmb, min_MSE=best_d_l(maxd, n_lmb, interval_degrees, lambdas, x1, y1, z1, 'Lasso')
    print(f"Max lambda used={n_lmb} and max degree used={maxd}")
    print(f"Best MSE while finding best parameters: {min_MSE}")
    print(f"with lambda={best_lmb}, and degree={best_deg}")

    # Plot MSE with color map
    im = plt.imshow(mse_values, cmap=plt.cm.RdBu, extent = [-10, 0, 1, maxd],
                interpolation=None, aspect='auto')
    plt.colorbar(im)
    plt.xlabel('log10(lambda)')
    plt.ylabel('degree of polynomial')
    #plt.title('MSE colormap (Ridge)')
    if 'save' in args :
        plt.savefig('terrain-lasso-degree-lambda-colormap')
    plt.show()

    """Performing lasso with the best degree and best lambda"""
    X = design_matrix(x1, y1, best_deg)

    sklearn_lasso = linear_model.Lasso(alpha=best_lmb)
    fit=sklearn_lasso.fit(X,z1)
    z_pred=fit.predict(X)

    #Plotting the ridge terrain result with the best degree
    plt.figure()
    plt.imshow(z_pred.reshape(y_arr, x_arr), cmap='bone', extent=[1010,1030,1530,1510])
    plt.xticks(rotation=45)
    plt.ylabel('Y')
    plt.xlabel('X')
    #plt.title('Lasso regression')
    if 'save' in args :
        plt.savefig('Terrain_lasso_bestdegree')
    plt.show()

    print(f"Lasso-Mean Squared Error: {MSE(z1,z_pred)}")
    print(f"Lasso-R2-score: {R2(z1,z_pred)}")

    return

"""Tasks as arguments in OLS
best_deg= Find best polynomial degree
OLS_plot= Perform OLS on the choosen part of terrain, and show the plot
CI=Find the confidence intervals and plot them"""

"""If you want to save the figures, use 'save' as argument"""
OLS_terrain('best_deg', 'OLS_plot', 'CI','save')
#ridge('save')
#lasso('save')
