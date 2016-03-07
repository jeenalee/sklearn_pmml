from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup

def build_model(pmml):
    """
    Builds model based on input PMML file.
    
    Finds which sklearn class to call based on model name specified in
    the PMML file. Returns a sklearn classifier with the parameters
    populated, and can be used for prediction.
    """
    with open(pmml, "r") as f:
        soup = BeautifulSoup(f, "xml")

    if "modelName" in soup.RegressionModel.attrs:
        model_type = soup.RegressionModel['modelName']
    else:
        raise IOError("The input PMML file does not have a modelName.")
        # TODO: Confirm R always outputs a modelName"
        
    model_class = model_class_from_model_type(model_type)
    model = model_class.from_pmml(pmml)
    return model


def model_class_from_model_type(model_type):
    if model_type == "Linear_Regression_Model":
        return LinearRegression
    
    # if model_type == "Random_Forest_Model":
    #     return RandomForest
    # if model_type == "SVM_Model":
    #     return SVM
