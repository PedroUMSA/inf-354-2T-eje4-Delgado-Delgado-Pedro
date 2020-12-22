
import numpy as np
np.random.seed(150)

class NeuralNet():
    def __init__(sl,ind,hnd,ond,l):
        sl.w1=np.random.normal(0.0,pow(hnd,-0.5),(ind,hnd))
        sl.w2=np.random.normal(0.0,pow(ond,-0.5),(hnd,ond))
        sl.lr=l
    #Funcion activacion sigmoiede
    def funact(sl,z):
        return 1/(1+np.exp(-z))
    #Derivada de la funcion de activacion sigmoide
    def derfunact(sl,z):
        return sl.funact(z)*(1-sl.funact(z))
    #Predice el resultado
    def predict(sl,inp):
        x=np.array(inp,ndmin=2)
        z1=np.dot(x,sl.w1)
        a=sl.funact(z1)
        z2=np.dot(a,sl.w2)
        y_hat=sl.funact(z2)
        return y_hat
    #Entrenamos al modelo
    def train(sl,inp,out):
        x = np.array(inp,ndmin=2)
        y = out
        z1 = np.dot(x,sl.w1)
        a = sl.funact(z1)
        z2 = np.dot(a,sl.w2)
        y_hat = sl.funact(z2)       
        error = y-y_hat
        delta2 = np.multiply(error,sl.derfunact(z2))
        djdw2 = np.dot(np.transpose(a),delta2)
        delta1 = np.dot(delta2,sl.w2.T)*sl.derfunact(z1)
        djdw1 = np.dot(x.T,delta1)
        sl.w1 += sl.lr*djdw1
        sl.w2 += sl.lr*djdw2

xor = np.array([[0,0,0],
                [1,0,1],
                [0,1,1],
                [1,1,0]])
x = []
y = []

for i in xor:
    x.append(i[:-1])
    y.append(i[len(i)-1])
x = np.array(x)
y = np.array(y)

#atributos de la red neuronal
inputnodes = 2
hiddennodes = 3
outputnodes = 1
learningrate = 0.1
#cremos la red neuronal
n=NeuralNet(inputnodes,hiddennodes,outputnodes,learningrate)
#entrenamos a la red neuronal
for e in range(10000):
    for x_t,y_t in zip(x,y):
        n.train(x_t,y_t)

# Entrada de datos de prueba        
entxor = np.array([[0,0],
                [1,0],
                [0,1],
                [1,1]])


pred = n.predict(entxor)
print("A B XOR")
presicion = 0
#prubeas 
for i in range(len(pred)):
    print(f'{entxor[i][0]} {entxor[i][0]}',end = " ")
    res = round(pred[i][0])
    print("", res)
    presicion += 1 if res == xor[i][2] else 0

presicion= presicion/len(xor)*100

print(f'\nPresicion del modelo {presicion}%')
    
