import unittest
import numpy
from bs4 import BeautifulSoup
import sys
#sys.path.append('/Users/Jeena/src/sklearn_pmml')
import sklearn_pmml
import sklearn_pmml.extensions
from sklearn.linear_model import LinearRegression

class LinearRegressionTests(unittest.TestCase):
    # Setting up for tests. Assign test files.
    def setUp(self):
        self.test_f = './tests/test_pmml_files/linear_regression/original.pmml'
        self.test_f_noModelName = './tests/test_pmml_files/linear_regression/noModelMame.pmml'
        self.test_f_noIntercept = './tests/test_pmml_files/linear_regression/noIntercept.pmml'
        self.test_f_noCoefficient = './tests/test_pmml_files/linear_regression/noCoefficient.pmml'
        self.test_f_noRegressionTable = './tests/test_pmml_files/linear_regression/noRegressionTable.pmml'

    # Tear down after tests.
    def tearDown(self):
        self.test_f = None
        self.test_f_noModelName = None
        self.test_f_noIntercept = None
        self.test_f_noCoefficient = None
        self.test_f_noRegressionTable = None
        
    # Test whether a PMML file is read and a model is returned.
    def test_readPMML(self):
        self.assertTrue(LinearRegression.from_pmml(self.test_f))

    # Test whether model is pulled out from a PMML file.
    def test_pulloutModel(self):
        self.assertTrue(sklearn_pmml.build_model(self.test_f))

    # Test whether model that is returned is an instance of
    # sklearn.linear_model.LinearRegression.
    def test_modelIsLinearModel(self):
        self.assertIsInstance(sklearn_pmml.build_model(self.test_f), LinearRegression)
        self.assertRaises(IOError, LinearRegression.from_pmml, self.test_f_noModelName)

    # Test whether a PMML file without intercept returns a model.
    def test_noIntercept(self):
        self.assertIsInstance(LinearRegression.from_pmml(self.test_f_noIntercept), LinearRegression)

    # Test whether a PMML file without RegressionTable raises ValueError.
    def test_noRegressionTable(self):
        self.assertRaises(ValueError, LinearRegression.from_pmml, self.test_f_noRegressionTable)

    # Test whether a PMML file without Coefficients raises ValueError.
    def test_noCoefficient(self):
        self.assertRaises(ValueError, LinearRegression.from_pmml, self.test_f_noCoefficient)

    # Test whether coefficients of the returned model is an instance
    # of numpy.ndarray.
    def test_coefficientsInNumpyArray(self):
        self.assertIsInstance(LinearRegression.from_pmml(self.test_f).coef_, numpy.ndarray)

    # Assert the number of coefficients are equal between the input
    # PMML file and the model.
    def test_numberOfCoefficients(self):
        with open(self.test_f, "r") as f:
            test_soup = BeautifulSoup(f, "xml")
            test_num_preds = test_soup.find_all("NumericPredictor")
        self.assertEqual(len(LinearRegression.from_pmml(self.test_f).coef_), len(test_num_preds))

if __name__ == '__main__':
    unittest.main()
