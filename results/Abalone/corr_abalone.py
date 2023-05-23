import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(10,10)})


df = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/abalone%20age/abalone.data', header=None)
x_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]
x_df.columns = ['Sex', 'Length', 'Diameter', 'Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']

x_df.iloc[:,0] = pd.get_dummies(x_df.iloc[:,0])
print(x_df.iloc[:,0])

ax = sns.heatmap(x_df.corr(), annot=True)
plt.xticks(rotation=50)
plt.title('Correlation matrix for Abalone dataset', fontsize = 16)
plt.tight_layout()
fig = ax.get_figure()
fig.savefig("corr_abalone.png")
print("figure saved as 'corr_abalone.png'")