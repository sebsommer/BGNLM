class Projection:
    def __init__(self, features, values, g, strategy = 1):
        self.alphas = np.ones(features.shape[0]+1)
        self.g = g
        values = np.column_stack([np.ones([values.shape[0], 1], dtype=np.float32), values])
        self.optimize(features, values, strategy)
            
    def optimize(self, features, values, strategy):
        if strategy == None:
            self.get_feature(features, g)
            self.get_values(values)
        if strategy == 1:
            tmpx = torch.tensor(np.column_stack((values, y_train)), dtype=torch.float32).to(DEVICE)

            data_mean = tmpx.mean(axis=0)[1:-1]
            data_std = tmpx.std(axis=0)[1:-1]
            tmpx[:,1:-1] = (tmpx[:,1:-1] - data_mean) / data_std

            p = tmpx.shape[1]

            model = LogisticRegression(p - 1, 1).to(DEVICE)
            criterion = nn.BCELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            
            # Gradient Descent
            for _ in range(200):
                _x = tmpx[:,:-1]
                _y = tmpx[:,-1]
                target = Variable(_y).to(DEVICE)
                data = Variable(_x).to(DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(torch.squeeze(outputs), target).to(DEVICE)
                loss.backward()
                optimizer.step()
            
            parm = {}
            for name, param in model.named_parameters():
                parm[name]=param.detach().cpu().numpy()

            alphas_out = parm['linear.weight'][0]
            self.alphas = alphas_out
            self.feature = self.get_feature(features, alphas_out)
            self.values = self.get_values(values)
        elif strategy == 2:
            #Not implemented
            return
        elif strategy == 3:
            #Not implemented
            """
            tmpx = torch.tensor(np.column_stack((values, y_train)), dtype=torch.float32).to(DEVICE)

            data_mean = tmpx.mean(axis=0)[1:-1]
            data_std = tmpx.std(axis=0)[1:-1]
            tmpx[:,1:-1] = (tmpx[:,1:-1] - data_mean) / data_std

            p = tmpx.shape[1]  # correct to an argument later

            model = LogisticRegression(p - 1, 1).to(DEVICE)
            criterion = nn.BCELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            for _ in range(100):
                _x = tmpx[:,:-1]
                _y = tmpx[:,-1]
                target = Variable(_y).to(DEVICE)
                data = Variable(_x).to(DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(torch.squeeze(outputs), target).to(DEVICE)
                loss.backward()
                optimizer.step()

            parm = {}
            for name, param in model.named_parameters():
                parm[name]=param.detach().cpu().numpy()
            alphas_out = parm['linear.weight'][0]
            self.alphas = alphas_out
            self.feature = self.get_feature(features, alphas_out)
            self.values = self.get_values(values)
            """
            return

    def get_feature(self, features, alphas_in):
        formula = f'({np.round(float(alphas_in[0]),4)}'
        for i, a in enumerate(alphas_in[1:]):
            if np.sign(a) > 0:
                formula += f'+{np.round(float(a),4)}*{features[i]}'
            else: 
                formula += f'{np.round(float(a),4)}*{features[i]}'
        formula += f')'
        self.f = self.g(formula)
        feature = str(formula)
        return feature


    def get_values(self, values):
        val = np.sum(self.alphas*values, axis = 1)
        return val

    def evaluate(self, values):
        values = np.column_stack([np.ones([values.shape[0], 1], dtype=np.float32), values])
        return np.sum(self.alphas*values, axis = 1)
    
    def __name__(self):
        return 'Projection'

class Modification:
    def __init__(self, feature, values, g):
        self.f = g(feature[0])
        self.feature = str(self.f)
        self.values = self.f.evaluate(values)
    
    def evaluate(self, values):
        return self.f.evaluate(values)

    def __name__(self):
        return 'Modification'

class Multiplication:
    def __init__(self, features, values):
        self.feature = f'{features[0]}*{features[1]}'
        self.values = values[:, 0] * values[:, 1]
    
    def evaluate(self, values):
        return values[:, 0] * values[:, 1]
    
    def __name__(self):
        return 'Multiplication'
