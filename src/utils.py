'''
Miscellaneous utility functions
'''

import collections
import operator
import os
import time
import datetime

def order_dictionary(dictionary, mode, reverse=False):
    '''
    Order a dictionary by 'key' or 'value'.
    mode should be either 'key' or 'value'
    http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    '''

    if mode =='key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(0),
                                              reverse=reverse))
    elif mode =='value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(1),
                                              reverse=reverse))

    else:
        raise ValueError("Unknown mode. Should be 'key' or 'value'")

def reverse_dictionary(dictionary):
    '''
    http://stackoverflow.com/questions/483666/python-reverse-inverse-a-mapping
    http://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain
    '''
    #print('type(dictionary): {0}'.format(type(dictionary)))
    if type(dictionary) is collections.OrderedDict:
        #print(type(dictionary))
        return collections.OrderedDict([(v, k) for k, v in dictionary.items()])
    else:
        return {v: k for k, v in dictionary.items()}



def merge_dictionaries(*dict_args):
    '''
    http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def pad_list(old_list, padding_size, padding_value):
    '''
    http://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
    Example: pad_list([6,2,3], 5, 0) returns [6,2,3,0,0]
    '''
    assert padding_size >= len(old_list)
    return old_list + [padding_value] * (padding_size-len(old_list))


def get_basename_without_extension(filepath):
    '''
    Getting the basename of the filepath without the extension
    E.g. 'data/formatted/movie_reviews.pickle' -> 'movie_reviews'
    '''
    return os.path.basename(os.path.splitext(filepath)[0])

def create_folder_if_not_exists(dir):
    '''
    Create the folder if it doesn't exist already.
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_current_milliseconds():
    '''
    http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    '''
    return(int(round(time.time() * 1000)))


def get_current_time_in_seconds():
    '''
    http://stackoverflow.com/questions/415511/how-to-get-current-time-in-python
    '''
    return(time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))

def get_current_time_in_miliseconds():
    '''
    http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    '''
    return(get_current_time_in_seconds() + '-' + str(datetime.datetime.now().microsecond))


def convert_configparser_to_dictionary(config):
    '''
    http://stackoverflow.com/questions/1773793/convert-configparser-items-to-dictionary
    '''
    my_config_parser_dict = {s:dict(config.items(s)) for s in config.sections()}
    return my_config_parser_dict


'''
# http://stackoverflow.com/questions/42257015/how-can-i-list-all-tensorflow-variables-a-node-depends-on
# computation flows from parents to children
def parents(op):
  return set(input.op for input in op.inputs)

def children(op):
  return set(op for out in op.outputs for op in out.consumers())

def get_graph():
  """Creates dictionary {node: {child1, child2, ..},..} for current
  TensorFlow graph. Result is compatible with networkx/toposort"""
  print('get_graph')
  ops = tf.get_default_graph().get_operations()
  return {op: children(op) for op in ops}


def print_tf_graph(graph):
  """Prints tensorflow graph in dictionary form."""
  for node in graph:
    for child in graph[node]:
      print("%s -> %s" % (node.name, child.name))
'''