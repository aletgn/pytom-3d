import functools
import pickle
import numpy as np
from typing import Tuple, List, Any


def summation(x, y):
    return x+y


def distance(x, y):
    return (x**2+y**2)**0.5


def distance2(x, y):
    return (abs(2*x)+y**2)**0.5


def export_regressor(regressor, folder: str = "./", filename: str = "my_regressor", extension: str = ".rg",
                     excluded_keys: List[str] = [], forced_values: List[Any] = []) -> None:
    """
    Export the attributes of a regressor to a dictionary for storage or inspection.

    Parameters
    ----------
    regressor : object
        The regressor object to export.

    path_to_file : str, optional
        The path to the file for exporting the regressor attributes. Default is "./".

    excluded_keys : list of str, optional
        A list of attribute keys to be excluded from the exported dictionary.

    forced_values : list of any, optional
        A list of values to be used for attributes specified in `excluded_keys`.

    Returns
    -------
        The dictionary containing the attributes of the regressor.

    Raises
    ------
    AssertionError
        If the length of `excluded_keys` is not equal to the length of `forced_values`.

    Notes
    -----
        This function is provided for convenience to handle backward compatibility
        of scikit-learn.

    """
    regressor_dict = {}

    assert len(excluded_keys) == len(forced_values)

    for k in vars(regressor).keys():
        if k in excluded_keys:
            regressor_dict[k] = forced_values[excluded_keys.index(k)]
        else:
            regressor_dict[k] = vars(regressor)[k]

    save(regressor_dict, folder, filename, extension)
    return regressor_dict


def import_regressor(folder: str = "./", filename: str = "my_regressor.rg", init_regressor = None) -> dict:
    """
    Import a regressor's attributes from a saved file and update an initialized regressor object.

    Parameters
    ----------
    folder : str, optional
        The path to the folder containing the saved regressor attributes file. Default is "./".

    filename : str, optional
        The name of the file containing the saved regressor attributes. Default is "my_regressor.rg".

    init_regressor : object
        The initialized regressor object to be updated with the imported attributes.

    Returns
    -------
    dict
        A dictionary containing the imported regressor attributes.

    Notes
    -----
        This function is provided for convenience to handle backward compatibility
        of scikit-learn.

    """
    regressor_dict = load(folder, filename)

    for k in regressor_dict.keys():
        setattr(init_regressor, k, regressor_dict[k])

    return regressor_dict


def trials(regressor, mesh, n: int = 1, folder: str = "./") -> None:
    """
    Generate and save trial data using a Gaussian Process Regression model.

    Parameters
    ----------
    regressor :
        The regressor of the topography.
    mesh : Topography
        The topogrphy object containing mesh data points for prediction.
    n : int, optional
        Number of trials to generate (default is 1).
    folder : str, optional
        The folder path to save the trial data files (default is "./").

    Returns
    -------
    None

    """
    for h in range(1, n+1):
        pred, sigma = regressor.predict(mesh.P[:, 0:2], return_std=True)
        noise = np.random.normal(loc=0, scale=sigma)
        output = np.vstack([mesh.P[:, 0], mesh.P[:, 1], mesh.P[:, 2],
                           pred, np.clip(max(0, h-1), 0, 1)*noise]).T
        np.savetxt(folder+mesh.name+"_" + str(h) + ".txt", output)


def predict_at_node(xx, yy, regressor):
    """
    Predict the value at a specific node in a regression model.

    Parameters
    ----------
    xx : float
        The x-coordinate of the node.
    yy : float
        The y-coordinate of the node.
    regressor : numpy.ndarray
        The regression model containing node information.

    Returns
    -------
    float
        The predicted value at the specified node.

    Raises
    ------
    Exception
        If there is not exactly one node matching the specified coordinates.

    """
    node_id = np.where(np.isclose(regressor[:, 0], xx, atol=1e-8) & np.isclose(regressor[:, 1], yy, atol=1e-8))[0]
    xm = regressor[node_id][0]
    ym = regressor[node_id][0]

    print(node_id)
    print("x:", xm, xx)
    print("y:", ym, yy)

    if len(node_id) == 1:
        return regressor[node_id][0][3] + regressor[node_id][0][4]
    else:
        raise Exception("There must be only one node.")


def prediction_wrapper(regressor, x, y) -> Tuple[np.ndarray]:
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
    p = np.array([x, y]).reshape(1, -1)
    pred, sigma = regressor.predict(p, return_std=True)
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


def load(folder: str = "./", filename: str = "my_file.bin"):
    """
     Load an object from a binary file using pickle.

    Parameters
    ----------
    - folder: str, optional
        The directory path where the file is located. Default is "./".
    - filename: str
        The name of the file to be loaded.

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
