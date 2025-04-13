'''linear regression implementation using gradient descent ,
 a simple implementation using numpy library and matplotlib for visualization'''
# importing the two necessary libraries 
import numpy as np
import matplotlib.pyplot as plt
# defining linear regression class for the implementation of ml model
class LinearRegression:
    '''defining the constructor for the class 
        and initializing the parameters for the model'''
    def __init__(self,learning_rate=0.01,max_iter=1000,tol=1e-6,verbose=True):
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        self.tol=tol
        self.verbose=verbose
        self.w=None
        self.b=None
        self.loss_history=[]
    '''defining the fit method for the model ,which will take the input data and the target variable as input , and outputs the model parameters'''
    def fit(self,X,y):
        y=y.reshape(-1,1)
        n_samples,n_features=X.shape
        self.w=np.random.randn(n_features,1)*0.01
        self.b=np.random.randn(1)*0.01
        for i in  range(self.max_iter):
            y_pred=self.predict(X)
            loss=self.compute_loss(y,y_pred)
            self.loss_history.append(loss)

            dw,db=self._compute_gradients(X,y,y_pred)            
            self.w-=self.learning_rate*dw
            self.b-=self.learning_rate*db
            
            if i>0 and abs(self.loss_history[-2]-loss)<self.tol:
                if self.verbose:
                    print(f"Converged at iteration : {i}")
                break
            if self.verbose and i%100==0:
                print(f'iteration {i:4d}:loss {loss:.4f}')
                
    def predict(self,X):
        return np.dot(X,self.w)+self.b

    def compute_loss(self,y_true,y_pred):
        return np.mean(y_true-y_pred)**2
    
    def _compute_gradients(self,X,y_true,y_pred):
         error=y_pred-y
         dw=(2/X.shape[0])*np.dot(X.T,error)
         db=(2/X.shape[0])*np.sum(error)
         return dw,db
    
    def r2_score(self,y_true,y_pred):
        ss_res=np.sum((y_true-y_pred)**2)
        ss_tot=np.sum((y_true-np.mean(y_true))**2)
        return 1-(ss_res/ss_tot)
    
def generate_data(n_samples=100,noise=1.0,random_seed=234):
    np.random.seed(random_seed)
    X=2*np.random.rand(n_samples,1)
    y=4+3*X+noise*np.random.randn(n_samples,1)
    return X,y

def plot_results(X,y,model,save_path=None):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(X,y,label='Training data')
    plt.plot(X,model.predict(X),color='red',label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('linear regression')
    plt.legend()
    plt.grid()
    
    plt.subplot(1,2,2)
    plt.plot(model.loss_history)
    plt.xlabel('iterations')
    plt.ylabel('mean squared error')
    plt.title('training loss history')
    plt.grid()

    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()
if __name__=='__main__':
    X,y=generate_data(noise=1.5,random_seed=234)
    model=LinearRegression(learning_rate=0.1,max_iter=1000)
    model.fit(X,y)
    y_pred=model.predict(X)
    print(f"R^2 score: {model.r2_score(y,y_pred):.4f}")
    plot_results(X,y,model) 


