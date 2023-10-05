
def append_path(dir):

    '''
    A function that adds given directory to system path
    Arguments:
    dir - directory which is to be appended to the system path
    '''

    # import the python system and os packages
    import sys
    import os

    # retrieve the absulute path of the directory
    path = os.path.abspath(dir)
    # if the given path is not included in the python system path
    # then add the path to python system path
    if path not in sys.path:
        sys.path.append(path)
