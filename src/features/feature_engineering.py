from sklearn.preprocessing import PolynomialFeatures


def get_polynomial_features(degree=2):
    return PolynomialFeatures(degree=degree, include_bias=False)
