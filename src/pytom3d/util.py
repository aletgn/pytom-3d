import functools
import pickle
import numpy as np

def summation(x,y):
    return x+y

def distance(x,y):
    return (x**2+y**2)**0.5

def distance2(x,y):
    return (abs(2*x)+y**2)**0.5

def prediction_wrapper(regressor, x, y):
    """
    Predict the target variable and its uncertainty for given x and y coordinates using a regressor.

    Parameters
    ----------
    regressor : Regressor
        The trained regressor model.
    x : float
        The x-coordinate for prediction.
    y : float
        The y-coordinate for prediction.

    Returns
    -------
    tuple
        A tuple containing the predicted value and its associated standard deviation (uncertainty).

    """
    p = np.array([x,y]).reshape(1,-1)
    pred, sigma = regressor.predict(p, return_std = True)
    return pred[0], sigma[0]

def save(obj, folder: str = "./", filename: str = "my_file", extension: str = ".bin") -> None:
    """
    Save the given object to a binary file using pickle.

    Parameters
    ----------
    - obj: Any
        The object to be saved.
    - folder: str, optional
        The directory path where the file will be saved. Default is "./".
    - filename: str, optional
        The name of the file to be saved. Default is "my_file".
    - extension: str, optional
        The file extension. Default is ".bin".

    Returns
    -------
    None

    """
    with open(folder + filename + extension, 'wb') as file:
        pickle.dump(obj, file)

def load(filename, folder: str = "./"):
    """
     Load an object from a binary file using pickle.
    
    Parameters
    ----------
    - filename: str
    The name of the file to be loaded.
    - folder: str, optional
    The directory path where the file is located. Default is "./".
    
    Returns
    -------
    Any
        The loaded object.

    """
    with open(folder + filename, 'rb') as file:
        return pickle.load(file)

def update(method: callable):
    """
    Decorator to update edges, centroid, cardinality, and record history after executing a method.

    Parameters
    ----------
    method : callable
        The method to be decorated.

    Returns
    -------
    callable
        Decorated method.

    Notes
    -----
    This decorator assumes that the decorated method returns a list of tuples,
    where each tuple contains key-value pairs to be recorded in the event history.

    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs) -> None:
        """
        Wrapper function to update edges, centroid, cardinality, and record history.
         
        Parameters
        ----------
        self : object
            Instance of the class.
        *args : tuple
            Positional arguments passed to the decorated method.
        **kwargs : dict
            Keyword arguments passed to the decorated method.
         
        Returns
        -------
        None
         
        Raises
        ------
        Any exceptions raised by the decorated method.
         
        Notes
        -----
        This wrapper assumes that the decorated method returns a list of tuples,
        where each tuple contains key-value pairs to be recorded in the event history.
        """
        # retrive values the method returns
        data = method(self, *args, **kwargs)
        
        # update edges, centroid, and cardinality
        self.edges()
        self.centroid()
        self.cardinality()
        
        # structure data for history
        event = {}
        for d in data:
            event[d[0]] = d[1]
        
        self.history_.append(event)
    return wrapper