from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train=X.shape[0]
    classes=W.shape[1]
    score=X.dot(W)
    for i in range(num_train):
      score[i]-=score[i].max()
      tol=np.exp(score[i])/np.sum(np.exp(score[i]))
      loss += -np.log(tol[y[i]])
      for j in range(classes):
        dW[:,j]+=tol[j]*X[i]
      dW[:,y[i]]-=X[i]
    loss/=num_train
    dW/=num_train

     
      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    num_train=X.shape[0]
    classes=W.shape[1]
    loss = 0.0
    dW = np.zeros_like(W)
    score=X.dot(W)


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score-=score.max(axis=1).reshape(num_train,1)
    loss1=np.exp(score[range(num_train),y])/np.exp(score).sum(axis=1)
    loss=-np.log(loss1).sum()/num_train
    matrix=np.exp(score)/np.exp(score).sum(axis=1).reshape(num_train,-1)
    matrix[range(num_train),y]-=1
    dW=X.T.dot(matrix)
    dW/=num_train
    
   

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
