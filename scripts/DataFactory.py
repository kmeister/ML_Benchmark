from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_data():
    X, y = make_classification(1000, 20, 15, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return (X_train, X_test, Y_train, Y_test)

def write_to_python(X,Y, filename, x_postfix=None, y_postfix):


def data_to_csv(X,y, filename):
    """
    Writes input data to csv
    :param X:
    :type X:
    :param y:
    :type y:
    :param filename:
    :type filename:
    :return:
    :rtype:
    """
    #TODO implement
    pass

def data_to_header(X, y, filename, Xt = None, yt=None, X_name=None, y_name=None, prefix="", postfix=""):
    """
    generate a header file from a dataset
    :param X:
    :type X:
    :param y:
    :type y:
    :param filename:
    :type filename:
    :param Xt:
    :type Xt:
    :param yt:
    :type yt:
    :param X_name:
    :type X_name:
    :param y_name:
    :type y_name:
    :param prefix:
    :type prefix:
    :param postfix:
    :type postfix:
    :return:
    :rtype:
    """
    #TODO implememt
    pass

