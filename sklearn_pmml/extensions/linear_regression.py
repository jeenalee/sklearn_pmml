from sklearn.linear_model import LinearRegression

def from_pmml(self, pmml):
    model = self()
    model.coef_ = [1,2,3]
    return model

LinearRegression.from_pmml = classmethod(from_pmml)


