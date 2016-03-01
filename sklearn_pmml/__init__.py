from sklearn.linear_model import LinearRegression

def build_model(pmml):
    model_type = "linear_model"
    model_class = model_class_from_model_type(model_type)
    print model_class
    model = model_class.from_pmml(pmml)
    return model
    
def model_class_from_model_type(model_type):
    if model_type == "linear_model":
        return LinearRegression

"""
m = new_pmml_to_sklearn.build_model('asdf')
<class 'sklearn.linear_model.base.LinearRegression'>
 
but not:
n = LinearRegression.from_pmml('asdf')
n = LinearRegression.from_pmml('asdf')
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-6-bf2d82eb5746> in <module>()
----> 1 n = LinearRegression.from_pmml('asdf')

NameError: name 'LinearRegression' is not defined
"""
