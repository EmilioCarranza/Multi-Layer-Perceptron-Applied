from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd

   
data= pd.read_excel(
    PercMultAplicado.xlsx')
data['Mora'] = data['Mora'].map({'SI': 1, 'NO': 0})
carga= data['Mensualidad']/data['Ingreso mensual']*100
data['Carga sobre salario']= carga

#%%division entre train y test

train, test = (train_test_split(data, test_size=0.3))
#%% monto, carga sobre el salario el pago de la mensualidad y la antiguedad laboral
pretrainx=( train.drop(['Entidad','Mensualidad','Tasa anual','Plazo (años)','Ingreso mensual','Mora'],axis=1)).to_numpy()
train_d= train['Mora'].to_numpy()

#%% normalizacion de datos train
train_x= preprocessing.normalize(pretrainx,axis=0)

N= 3 #number of inputs
L= 6 # numero de neuronas ocultas
M= 1  # numero de outputs
Q= 700  #different inputs filas
wh= np.random.random([L,N])*2-1
wo= np.random.random([M,L])*2-1
E=1 # error

#%% training data
while E>= 0.000001:
    for i in range(Q):
    # Forward
        
        neth= np.matmul(wh,train_x[i].T)
        yh= 1/(1+np.exp(-neth))
        neto= np.matmul(wo,yh)
        y= 1/(1+np.exp(-neto))
       
    
    # Backward
    
        so= ((train_d[i].T)-y)*(y*(1-y))
        sh=  yh*(1-yh)*[np.matmul(wo.T,so)]
        alpha= 0.5
        aso= alpha*so[np.newaxis].T
        yht= yh[np.newaxis]
        dwO= np.matmul(aso,yht)
        wo+= dwO
        
        ash= (alpha*sh).T
        xx= train_x[i][np.newaxis]
        dwh= np.matmul(ash,xx)
        wh+= dwh

# error
        E= max(abs(so))
        

#%% prepare test data
pretestx=( test.drop(['Entidad','Mensualidad','Tasa anual','Plazo (años)','Ingreso mensual','Mora'],axis=1)).to_numpy()
test_x= preprocessing.normalize(pretestx,axis=0)
test_d= test['Mora'].to_numpy()
y_final=[]
for i in range(300):
    # Forward
    xt_predict= test_x[i,:]
    neth_t= np.matmul(wh,xt_predict.T)
    yh_t= 1/(1+np.exp(-2*neth_t))
    neto_t= np.matmul(wo,yh_t)
    y_t= 1/(1+np.exp(-2*neto_t))
    y_red= np.round(y_t)
    y_final.append(y_red)

    print('prediccion',xt_predict,y_red)


#%%validation
Counter=0
for i in range(300):
    if y_final[i] == test_d[i]:
        Counter+=1
    else:
        pass
Accuracy= (Counter/300) *100
print('El accuracy es de:',Accuracy,'%')
