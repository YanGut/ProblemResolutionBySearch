import numpy as np
import matplotlib.pyplot as plt


def roleta(C, probs):
    i = 0
    s = probs[i]
    r = np.random.uniform()
    while s < r:
        i += 1
        s += probs[i]
    return C[i,:]

def f(x):
    return x**2
def phi(x, inf, sup):
    s = 0
    for i in range(len(x)):
        s+= x[len(x)-i-1]*2**i
    
    return inf + (sup-inf)/(2**len(x)-1)*s



def f(x, y):
    return np.abs(x*y*np.sin(y*np.pi/4))







N = 16
p = 2
nd = 8
P = np.random.uniform(low=0,high=2,size=(N,p*nd)).astype(int) #representação canônica (bits)
individuo = np.split(P[0,:],p)
decodificado = [phi(i,-1,15) for i in individuo]
aptidao = f(*decodificado)




x1 = P[7,:]
x2 = P[3,:]
f1 = np.copy(x1)
f2 = np.copy(x2)
mask = np.zeros(len(x1))
idx = np.random.randint(low = 1,high=len(x1))
mask[idx:] = 1
f1[mask[:]==1] = x2[mask[:]==1]
f2[mask[:]==1] = x1[mask[:]==1]
bp=1












P = np.random.uniform(low=-3,high=15,size=(N,p))#representação não canônica (contínuo)
P = np.random.uniform(low=0, high=8, size=(N,8)).astype(int) #representação não canônica (discreta)
P = [np.random.permutation(10) for i in range(N)]







bp=1












# C = np.random.randint(low=0,high=2,size=(N,nd*p))

C = np.array([
    [1,1,0,0,0],
    [1,0,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,1],
])

avaliacao = []
for i in range(C.shape[0]):
    avaliacao.append(
        f(
            phi(C[i,:], 0, 15), 
            phi(C[i,:], 0, 15)
        )
    )

total = np.sum(avaliacao)
probs = []
for i in range(C.shape[0]):
    probs.append(avaliacao[i]/total)


selecionados = np.empty((0,5))
for i in range(C.shape[0]):
    selecionados = np.concat((
        selecionados,
        roleta(C,probs).reshape(1,5)
    ))

plt.pie(probs,labels=probs)
plt.show()
bp = 1