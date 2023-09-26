
# making the model classes directly importable without the reference of the class files
from .simple_linear_regression import simple_linear_regression
from .linear_regression import linear_regression
from .ridge_regression import ridge_regression
from .lasso_regression import lasso_regression

__all__ = ['simple_linear_regression',
           'linear_regression',
           'ridge_regression',
           'lasso_regression']
