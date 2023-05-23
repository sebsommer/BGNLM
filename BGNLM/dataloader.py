import pandas as pd
import numpy as np
import sys

from nonlinear import *

def standardize(data: np.array):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return np.divide((data - data_mean), data_std)

def maxabs(data: np.array):
    x = data.copy()
    if len(x.shape) > 1:
        for i in range(x.shape[1]):
            x[:,i] = np.divide(x[:,i], np.max(np.abs(x[:,i])))
    else:
        x = np.divide(x, np.max(np.abs(x)))
    return x

def minmax(data: np.array):
    x = data.copy()
    if len(x.shape) > 1:
        for i in range(x.shape[1]):
            x[:,i] = np.divide(x[:,i] - np.min(x[:,i]), np.max(x[:,i]) - np.min(x[:,i]))
    else:
        x = np.divide(x, np.max(np.abs(x)))
    return x


def dataloader(dataset_nr, normalize = standardize):
    
    if normalize == None:
        norm = False       
    
    #Breast Cancer (binomial)
    if dataset_nr == '1':
        x_df = pd.read_csv(
            'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/test.csv').iloc[
            :, 1:-1]

        y_df = pd.read_csv(
            'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/test.csv').iloc[
            :, -1]

        x_test = np.array(x_df)
        y_test = np.array(y_df)

        x_train = pd.read_csv(
            'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/train.csv').iloc[
                :, 1:-1].to_numpy()
        y_train = pd.read_csv(
            'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/train.csv').iloc[
                :, -1].to_numpy()

        family = 'binomial'
        
        BATCH_SIZE = 500
        epochs = 1000


        print("Data: Breast cancer")
    #simx (binomial)
    elif dataset_nr == '2':
        x_df = pd.read_csv(
            'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-X.txt',
            header=None)

        y_df = pd.read_csv(
            'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-Y.txt',
            header=None)
        x = np.array(x_df)
        y = np.array(y_df)

        # Split:
        x_train = x_df.sample(frac=0.8, random_state=100)
        y_train = y_df.sample(frac=0.8, random_state=100)

        x_test = x_df[~x_df.index.isin(x_train.index)]
        y_test = y_df[~y_df.index.isin(y_train.index)]

        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

        family = 'binomial'
        print("Data: Sim-data 1, binomial")
    #abalone (gaussian)
    elif dataset_nr == '3':
        te_ids = pd.read_csv("https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/abalone%20age/teid.csv", header = 1, sep = ";").iloc[:,-1] -1
        df = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/abalone%20age/abalone.data', header=None)
        x_df = df.iloc[:,:-1]
        y_df = df.iloc[:,-1]
        

        dummies = pd.get_dummies(x_df.iloc[:,0])
        res = pd.concat([x_df, dummies], axis = 1)
        x_df = res.drop([0], axis = 1)
        x_cols=x_df.columns = ['Length', 'Diameter', 'Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight', 'Female', 'Infant', 'Male']

        x_df = x_df.drop(['Infant'], axis =1)
        x_cols = ['Length', 'Diameter', 'Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight', 'Female', 'Male']
        
        x_test = x_df.iloc[te_ids,:]
        y_test = y_df.iloc[te_ids]

        x_train = x_df[~x_df.index.isin(te_ids)]
        y_train = y_df[~y_df.index.isin(te_ids)]

        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

        family = 'gaussian'
    #kepler (gaussian)
    elif dataset_nr == '4':

        df = pd.read_csv(
            'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/kepler%20and%20mass/exa1.csv')#[:,1:10]
        x_df = df.iloc[:, [1,2,3,5,6,7,8,9,10]]
        y_df = df.iloc[:, 4]

        
        x_cols = ['Mp', 'Rp', 'P', 'e', 'Mh', 'Rh', 'Feh', 'Th', 'Dp']

        x_test = x_df
        y_test =  y_df

        x_train = x_df
        y_train = y_df

        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

        print(x_train.shape)

        family = 'gaussian'
        print("Data: Kepler's third law")  
    #ct slice (gaussian)
    elif dataset_nr == '5':
        df = pd.read_csv(
            'slice_localization.csv',
            header=1)
        
        #df = df.dropna()

        x_df = df.iloc[:, :-1]
        y_df = df.iloc[:, -1]

        x_df = x_df.loc[:, x_df.std(axis=0)>0.01]

        # Split:
        x_train = x_df.sample(frac=0.8, random_state=1104)
        y_train = y_df.sample(frac=0.8, random_state=1104)

        x_test = x_df[~x_df.index.isin(x_train.index)]
        y_test = y_df[~y_df.index.isin(y_train.index)]

        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

        #print(np.min(x_train[:, 276]), np.max(x_train[:, 276]), np.mean(x_train[:, 276]), np.std(x_train[:, 276]))
        #print(x_train[:, 276][x_train[:, 276] > -0.25])

        print(x_train.shape, x_test.shape)

        #x_train = x_train[:, x_train.std(axis=0)>0.01]
        #x_test = x_test[:, x_train.std(axis=0)>0.01]

        family = 'gaussian'
        
        print("Data: CT slice localization, normal")

    #spam (binomial)
    elif dataset_nr == '6':
        idx = pd.read_csv("spam_idx.txt", header = None).squeeze()
        df = pd.read_csv('spam.txt', header=None)
        x_df = df.iloc[:,:-1]
        y_df = df.iloc[:,-1]


        x_test = x_df[idx==1]
        y_test = y_df[idx==1]

        x_train = x_df[idx==0]
        y_train = y_df[idx==0]

        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

        family = 'binomial'
        
        BATCH_SIZE = 500
        epochs = 600

        LEARNIG_RATE = 0.01

        print("Data: Spam")
        norm = True
    #superconductor (gaussian)
    elif dataset_nr == '7':
        df = pd.read_csv(
            'superconductor.txt',
            header=0)
        
        df = df.dropna()


        x_df = df.iloc[:, :-1]
        y_df = df.iloc[:, -1]

        x_df = x_df.loc[:, x_df.std(axis=0)>0.01]

        x_cols = x_df.columns.to_numpy()

        # Split:
        x_train = x_df.sample(frac=0.75, random_state=1104)
        y_train = y_df.sample(frac=0.75, random_state=1104)

        x_test = x_df[~x_df.index.isin(x_train.index)]
        y_test = y_df[~y_df.index.isin(y_train.index)]

        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

        #print(np.min(x_train[:, 276]), np.max(x_train[:, 276]), np.mean(x_train[:, 276]), np.std(x_train[:, 276]))
        #print(x_train[:, 276][x_train[:, 276] > -0.25])

        print(x_train.shape, x_test.shape)

        #x_train = x_train[:, x_train.std(axis=0)>0.01]
        #x_test = x_test[:, x_train.std(axis=0)>0.01]

        family = 'gaussian'
        
        print("Data: Superconductor, normal")
    #normal independent (gaussian)
    elif dataset_nr == 'sim1':
        X = np.random.normal(0,1, (10000, 20))
        beta = np.array([0,0,0,0,0, 1.5, -4, 3, -0.2, 1,0,0,0,0,0, -2, 1.3, 0.3, -0.8, 3])

        

        sigma = float(np.sqrt(sys.argv[6]))
        noise = np.random.normal(0,sigma, size = 10000)

        #X[:, 2] = (1-alpha)*X[:, 2] + (alpha)*X[:, 5]

        y = X @ beta.T + noise


        x_test = X
        x_train = X

        y_test = y
        y_train = y

        family = 'gaussian'
        print("Data: sim-data 1, Normal; mu = 0, sigma =", sigma)

    elif dataset_nr == 'sim1_corr':
        X = np.random.normal(0,1, (15000, 20))
        beta = np.array([0,0,0,0,0, 1.5, -4, 3, -0.2, 1,0,0,0,0,0, -2, 1.3, 0.3, -0.8, 3])

        noise = np.random.normal(0,1, size = 15000)

        alpha = float(sys.argv[5])

        X[:, 2] = (1-alpha)*X[:, 2] + (alpha)*X[:, 5]

        y = X @ beta.T + noise


        x_test = X
        x_train = X

        y_test = y
        y_train = y

        family = 'gaussian'
        print("Data: sim-data 1, corr, alpha=", alpha)

    elif dataset_nr == 'sim1_compcorr':
        lst = []
        for _ in range(20):
            a = np.random.RandomState(1104).choice(100, size = 20)
            lst.append(a)

        cov = np.cov(np.array(lst), bias= False)

        X = np.random.RandomState(1104).multivariate_normal(mean = np.zeros(20), cov = cov, size = 15000)

        beta = np.array([0,0,0,0,0, 1.5, -4, 3, -0.2, 1,0,0,0,0,0, -2, 1.3, 0.3, -0.8, 3])

        noise = np.random.RandomState(110495).normal(0,1, size = 15000)

        y = X @ beta.T + noise

        y_df = pd.DataFrame(y)
        x_df = pd.DataFrame(X)

        x_train = x_df.sample(frac=0.75, random_state=1104)
        y_train = y_df.sample(frac=0.75, random_state=1104)

        x_test = x_df[~x_df.index.isin(x_train.index)]
        y_test = y_df[~y_df.index.isin(y_train.index)]

        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

        family = 'gaussian'
        print("Data: sim-data 1, Normal; mu = 0, sigma =", 1)

        N_GENERATIONS = 5
        norm = False
        BATCH_SIZE = 2000
        N_IAFLAYERS = 5
        epochs = 500 
    #art study (gaussian)
    elif dataset_nr == 'sim2':
        try: sim2y
        except NameError:
            print("Need to specify sigma! \n")
            sim2y = input("Provide sigma for sim study 2: ")
        X = pd.read_csv('art.txt', header = 0)
        
        x_cols = X.iloc[:,2:14].columns.to_numpy()
        Y = X.iloc[:,14:]

        #x = normalize(X.iloc[:,2:14].to_numpy())

        #mu = 0.1 + normalize(p05('x1').evaluate(x[:,0])) + x[:,0] + normalize(pm05('x3').evaluate(x[:,2])) + normalize(p0pm05('x3').evaluate(x[:,2])) + x[:,8] + normalize(pm1('x5').evaluate(x[:,3])) + normalize(p1('x6').evaluate(x[:,4])) + x[:,6] + x[:,7]

        x = normalize(X)


        mu = normalize(p05('x1').evaluate(X['x1'])) + x['x1'] + normalize(pm05('x3').evaluate(X['x3'])) + normalize(p0pm05('x3').evaluate(X['x3'])) + x['x4a'] + normalize(pm1('x5').evaluate(X['x5'])) + normalize(p1('x6').evaluate(X['x6'])) + x['x8'] + x['x10']

        sds = float(sim2y)
        y = np.random.normal(mu, scale = sds)

        x_train = X.iloc[:,2:14].to_numpy()
        x_test = X.iloc[:,2:14].to_numpy()
        y_train = y
        y_test = y

        family = 'gaussian'

        print('Data: Sim-data 2, sigma =', sim2y)
    elif dataset_nr == 'sim2_precomputed':

        try: sim2y
        except NameError:
            print("Need to specify y! Options are:\n y_50\n y_25\n y_10 \n y_5 \n y_1 \n y_0.1\n y_0.01\n y_0.001 \n y_1e-4 \n y_1e-5 \n y_1e-6\n y_1e-7\n y_1e-7 \n y_1e-8 \n y_1e-9 \n y_1e-10 \n")
            sim2y = input("Provide y for sim study 2: ")
        X = pd.read_csv('art.txt', header = 0)
        


        #x = X.iloc[:,2:14].to_numpy()

        #mu = normalize(p05('x1').evaluate(normalize(X['x1']))) + normalize(X['x1']) + normalize(pm05('x3').evaluate(normalize(X['x3']))) + normalize(p0pm05('x3').evaluate(normalize(X['x3']))) + normalize(X['x4a']) + normalize(pm1('x5').evaluate(normalize(X['x5']))) + normalize(p1('x6').evaluate(normalize(X['x6']))) + normalize(X['x8']) + normalize(X['x10'])

        x_cols = X.iloc[:,2:14].columns.to_numpy()
        Y = X.iloc[:,14:]
    

        x = normalize(X.iloc[:,2:14].to_numpy())

        mu = 0.1 + normalize(p05('x1').evaluate(x[:,0])) + x[:,0] + normalize(pm05('x3').evaluate(x[:,2])) + normalize(p0pm05('x3').evaluate(x[:,2])) + x[:,8] + normalize(pm1('x5').evaluate(x[:,3])) + normalize(p1('x6').evaluate(x[:,4])) + x[:,6] + x[:,7]

        #sds = float(sim2y)
        #y = mu + np.random.normal(0, scale = sds, size = X.shape[0])

        x_train = x_test = x
        #y_train = y_test = normalize(y)

        y_test = Y[sim2y].to_numpy()
        y_train = Y[sim2y].to_numpy()

        family = 'gaussian'

        print('Data: Sim-data 2, sigma =', sim2y)
    elif dataset_nr == 'sim2_nonorm':
        sim2y = float(sys.argv[5])
        try: sim2y
        except NameError:
            print("Need to specify y! Options are:\n y_50\n y_25\n y_10 \n y_5 \n y_1 \n y_0.1\n y_0.01\n y_0.001 \n y_1e-4 \n y_1e-5 \n y_1e-6\n y_1e-7\n y_1e-7 \n y_1e-8 \n y_1e-9 \n y_1e-10 \n")
            sim2y = input("Provide y for sim study 2: ")
        X = pd.read_csv('art.txt', header = 0)
        
        mu = p05('x1').evaluate(X['x1']) + X['x1'] + pm05('x3').evaluate(X['x3']) + p0pm05('x3').evaluate(X['x3']) + X['x4a'] + pm1('x5').evaluate(X['x5']) + p1('x6').evaluate(X['x6']) + X['x8'] + X['x10']

        eps = 0.1

        mu = mu+eps

        x_cols = X.iloc[:,2:14].columns.to_numpy()
        Y = X.iloc[:,14:]
        
        sds = float(sim2y)
        y = np.random.normal(loc = mu, scale = sds, size = X.shape[0])

        x_train = x_test = X.iloc[:,2:14].to_numpy()
        

        y_train = y_test = y

        #y_test = Y[sim2y].to_numpy()
        #y_train = Y[sim2y].to_numpy()

        family = 'gaussian'

        print('Data: Sim-data 2, sigma =', sim2y)
    elif dataset_nr == 'sim2test':
        try: sim2y
        except NameError:
            print("Need to specify sigma! \n")
            sim2y = input("Provide sigma for sim study 2: ")
        X = pd.read_csv('art.txt', header = 0)
        
        x_cols = X.iloc[:,2:14].columns.to_numpy()
        Y = X.iloc[:,14:]

        x = normalize(X)

        mu = normalize(p05('x1').evaluate(x['x1'])) + x['x1'] + normalize(p05('x3').evaluate(x['x3'])) + normalize(p0p05('x3').evaluate(x['x3'])) + x['x4a'] + normalize(pm1('x5').evaluate(x['x5'])) + normalize(p1('x6').evaluate(x['x6'])) + x['x8'] + x['x10']

        sds = float(sim2y)
        y = np.random.normal(mu, scale = sds)

        x_train = X.iloc[:,2:14].to_numpy()
        x_test = X.iloc[:,2:14].to_numpy()
        y_train = y
        y_test = y

        family = 'gaussian'

        print('Data: Sim-data 2, sigma =', sim2y)
    elif dataset_nr == 'sim2_original':
        sim2y = float(sys.argv[5])
        try: sim2y
        except NameError:
            print("Need to specify y! Options are:\n y_50\n y_25\n y_10 \n y_5 \n y_1 \n y_0.1\n y_0.01\n y_0.001 \n y_1e-4 \n y_1e-5 \n y_1e-6\n y_1e-7\n y_1e-7 \n y_1e-8 \n y_1e-9 \n y_1e-10 \n")
            sim2y = input("Provide y for sim study 2: ")
        X = pd.read_csv('art.txt', header = 0)
        


        #x = X.iloc[:,2:14].to_numpy()

        #mu = normalize(p05('x1').evaluate(normalize(X['x1']))) + normalize(X['x1']) + normalize(pm05('x3').evaluate(normalize(X['x3']))) + normalize(p0pm05('x3').evaluate(normalize(X['x3']))) + normalize(X['x4a']) + normalize(pm1('x5').evaluate(normalize(X['x5']))) + normalize(p1('x6').evaluate(normalize(X['x6']))) + normalize(X['x8']) + normalize(X['x10'])

        x_cols = X.iloc[:,2:14].columns.to_numpy()
        Y = X.iloc[:,14:]

        x = normalize(X.iloc[:,2:14].to_numpy())

        mu = 0.1 + normalize(p05('x1').evaluate(x[:,0])) + x[:,0] + x[:,2] + x[:,8] + normalize(xm15('x5').evaluate(x[:,3])) + normalize(Ln1p('x6').evaluate(x[:,4])) + x[:,6] + x[:,7]

        sds = float(sim2y)
        y = mu + np.random.RandomState(1104).normal(0, scale = sds, size = X.shape[0])

        x_train = x_test = x
        y_train = y_test = normalize(y)

        #y_test = Y[sim2y].to_numpy()
        #y_train = Y[sim2y].to_numpy()

        family = 'gaussian'

        print('Data: Sim-data 2, sigma =', sim2y)

    try: x_cols
    except NameError: x_cols = [f'x{j}' for j in range(x_train.shape[1])]
    
    print(x_train.shape)

    return x_train, x_test, y_train, y_test, family, x_cols