from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import numpy

def from_pmml(self, pmml):
    """Returns a model with the intercept and coefficients represented in PMML file."""

    model = self()
    
    # Reads the input PMML file with BeautifulSoup.
    with open(pmml, "r") as f:
        lm_soup = BeautifulSoup(f, "xml")

    if not lm_soup.RegressionTable:
        raise ValueError("RegressionTable not found in the input PMML file.")

    else:
    ##### DO I WANT TO PULL THIS OUT AS ITS OWN FUNCTION? #####
        # Pulls out intercept from the PMML file and assigns it to the
        # model. If the intercept does not exist, assign it to zero.
        intercept = 0
        if "intercept" in lm_soup.RegressionTable.attrs:
            intercept = lm_soup.RegressionTable['intercept']
        model.intercept_ = float(intercept)

        # Pulls out coefficients from the PMML file, and assigns them
        # to the model.
        if not lm_soup.find_all('NumericPredictor'):
            raise ValueError("NumericPredictor not found in the input PMML file.")
        else:
            coefs = []
            numeric_predictors = lm_soup.find_all('NumericPredictor')
            for i in numeric_predictors:
                i_coef = float(i['coefficient'])
                coefs.append(i_coef)
            model.coef_ = numpy.array(coefs)
            
    return model

# TODO: check input data's X order and rearrange the array
LinearRegression.from_pmml = classmethod(from_pmml)
