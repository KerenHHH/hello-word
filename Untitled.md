
# 线性回归：LR

$$w=(X^TX)^{-1}X^Ty$$


```python
import numpy as np
import numpy.linalg as lg
```


```python
X = np.random.randint(0,10,(4,3))
y = np.random.randint(0,10,(4,1))
w = (lg.inv(X.T.dot(X)).dot(X.T)).dot(y)
```


```python
print(X)
```

    [[8 4 3]
     [7 2 9]
     [2 4 9]
     [9 6 5]]
    


```python
print(y)
```

    [[4]
     [6]
     [1]
     [9]]
    


```python
print(w)
```

    [[0.7761194 ]
     [0.01699834]
     [0.00995025]]
    

# 线性回归：LWLR

$$w=(X^TWX)^{-1}X^TWy$$


```python
W = np.random.randint(0,10,(4,4))
w = ((lg.inv((X.T.dot(W)).dot(X)).dot(X.T)).dot(W)).dot(y)
```


```python
print(W)
```

    [[6 8 7 9]
     [3 6 1 4]
     [1 8 8 0]
     [2 5 6 2]]
    


```python
print(w)
```

    [[ 0.91762792]
     [-0.09379221]
     [-0.05922098]]
    

# Fit w to minimize

$$\sum_iw^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2$$


```python
i=0
result=np.array([[0]],dtype='float64')
```


```python
while i<3 :
    result+=w[i].dot((y[i]-w.T.dot(X[i])).dot((y[i]-w.T.dot(X[i]))))
    i+=1
```


```python
print(result)
```

    [[7.12505267]]
    
