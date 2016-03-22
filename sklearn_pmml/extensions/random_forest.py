from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import tree
from bs4 import BeautifulSoup
import numpy

def from_pmml(self, pmml):
    """Returns a model with the intercept and coefficients represented in PMML file."""

    model = self()
    
    # Reads the input PMML file with BeautifulSoup.
    with open(pmml, "r") as f:
        rf_soup = BeautifulSoup(f, "xml")

        # Find number of trees in the pmml-forest.
        n_estimators = len(rf_soup.find_all('Segment'))
        # Add number of trees to the model.
        model.n_estimators = n_estimators
        # Add trees to the random forest model.
        model.estimators_ = []

        for i in range(0, n_estimators):
            model.estimators_.append(tree.DecisionTreeClassifier())
            
            # TODO:Build tree
            
    return model


def _extract_tree(pmml):
    pass

def _extract_left_child(pmml):
    pass

def _extract_right_child(pmml):
    pass

def _extract_threshold(pmml):
    pass


RandomForestClassifier.from_pmml = classmethod(from_pmml)

