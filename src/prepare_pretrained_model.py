'''
This script prepares a pretrained model to be shared without exposing the data used for training.
'''
import os
import pickle
from pprint import pprint
import shutil
import utils
import main
from entity_lstm import EntityLSTM
import tensorflow as tf
import utils_tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import glob

def trim_dataset_pickle(input_dataset_filepath, output_dataset_filepath=None, delete_token_mappings=False):
    '''
    Remove the dataset and labels from dataset.pickle. 
    If delete_token_mappings = True, then also remove token_to_index and index_to_token except for UNK.
    '''
    print("Trimming dataset.pickle..")
    if output_dataset_filepath == None:
        output_dataset_filepath = os.path.join(os.path.dirname(input_dataset_filepath), 'dataset_trimmed.pickle')
    dataset = pickle.load(open(input_dataset_filepath, 'rb'))
    count = 0
    print("Keys removed:")
    keys_to_remove = ['character_indices', 'character_indices_padded', 'characters', 'label_indices', 'label_vector_indices', 'labels', 
                      'token_indices', 'token_lengths', 'tokens', 'infrequent_token_indices', 'tokens_mapped_to_unk']
    for key in keys_to_remove:
        if key in dataset.__dict__:
            del dataset.__dict__[key]
            print('\t' + key)
            count += 1            
    if delete_token_mappings:
        dataset.__dict__['token_to_index'] = {dataset.__dict__['UNK']:dataset.__dict__['UNK_TOKEN_INDEX']}
        dataset.__dict__['index_to_token'] = {dataset.__dict__['UNK_TOKEN_INDEX']:dataset.__dict__['UNK']}
    print("Number of keys removed: {0}".format(count))
    pprint(dataset.__dict__)
    pickle.dump(dataset, open(output_dataset_filepath, 'wb'))
    print("Done!")


def trim_model_checkpoint(parameters_filepath, dataset_filepath, input_checkpoint_filepath, output_checkpoint_filepath):
    '''
    Remove all token embeddings except UNK.
    '''
    parameters, _ = main.load_parameters(parameters_filepath=parameters_filepath)
    dataset = pickle.load(open(dataset_filepath, 'rb'))
    model = EntityLSTM(dataset, parameters) 
    with tf.Session() as sess:
        model_saver = tf.train.Saver()  # defaults to saving all variables
        
        # Restore the pretrained model
        model_saver.restore(sess, input_checkpoint_filepath) # Works only when the dimensions of tensor variables are matched.
        
        # Get pretrained embeddings
        token_embedding_weights = sess.run(model.token_embedding_weights) 
    
        # Restore the sizes of token embedding weights
        utils_tf.resize_tensor_variable(sess, model.token_embedding_weights, [1, parameters['token_embedding_dimension']]) 
            
        initial_weights = sess.run(model.token_embedding_weights)
        initial_weights[dataset.UNK_TOKEN_INDEX] = token_embedding_weights[dataset.UNK_TOKEN_INDEX]
        sess.run(tf.assign(model.token_embedding_weights, initial_weights, validate_shape=False))
    
        token_embedding_weights = sess.run(model.token_embedding_weights) 
        print("token_embedding_weights: {0}".format(token_embedding_weights))
        
        model_saver.save(sess, output_checkpoint_filepath)
            
    dataset.__dict__['vocabulary_size'] = 1
    pickle.dump(dataset, open(dataset_filepath, 'wb'))
    pprint(dataset.__dict__)


def prepare_pretrained_model_for_restoring(output_folder_name, epoch_number, model_name, delete_token_mappings=False):
    '''
    Copy the dataset.pickle, parameters.ini, and model checkpoint files after removing the data used for training.
    
    The dataset and labels are deleted from dataset.pickle by default. The only information about the dataset that remain in the pretrained model
    is the list of tokens that appears in the dataset and the corresponding token embeddings learned from the dataset.
    
    If delete_token_mappings is set to True, index_to_token and token_to_index mappings are deleted from dataset.pickle additionally,
    and the corresponding token embeddings are deleted from the model checkpoint files. In this case, the pretrained model would not contain
    any information about the dataset used for training the model. 
    
    If you wish to share a pretrained model with delete_token_mappings = True, it is highly recommended to use some external pre-trained token 
    embeddings and freeze them while training the model to obtain high performance. This can be done by specifying the token_pretrained_embedding_filepath 
    and setting freeze_token_embeddings = True in parameters.ini for training.
    '''
    input_model_folder = os.path.join('..', 'output', output_folder_name, 'model')
    output_model_folder = os.path.join('..', 'trained_models', model_name)
    utils.create_folder_if_not_exists(output_model_folder)

    # trim and copy dataset.pickle
    input_dataset_filepath = os.path.join(input_model_folder, 'dataset.pickle')
    output_dataset_filepath = os.path.join(output_model_folder, 'dataset.pickle')
    trim_dataset_pickle(input_dataset_filepath, output_dataset_filepath, delete_token_mappings=delete_token_mappings)
    
    # copy parameters.ini
    parameters_filepath = os.path.join(input_model_folder, 'parameters.ini')
    shutil.copy(parameters_filepath, output_model_folder)
    
    # (trim and) copy checkpoint files
    epoch_number_string = str(epoch_number).zfill(5)
    if delete_token_mappings:
        input_checkpoint_filepath = os.path.join(input_model_folder, 'model_{0}.ckpt'.format(epoch_number_string))
        output_checkpoint_filepath = os.path.join(output_model_folder, 'model.ckpt')
        trim_model_checkpoint(parameters_filepath, output_dataset_filepath, input_checkpoint_filepath, output_checkpoint_filepath)
    else:
        for filepath in glob.glob(os.path.join(input_model_folder, 'model_{0}.ckpt*'.format(epoch_number_string))):
            shutil.copyfile(filepath, os.path.join(output_model_folder, os.path.basename(filepath).replace('_' + epoch_number_string, '')))

 
def check_contents_of_dataset_and_model_checkpoint(model_folder):
    '''
    Check the contents of dataset.pickle and model_xxx.ckpt.
    model_folder: folder containing dataset.pickle and model_xxx.ckpt to be checked. 
    '''
    dataset_filepath = os.path.join(model_folder, 'dataset.pickle')
    dataset = pickle.load(open(dataset_filepath, 'rb'))
    pprint(dataset.__dict__)
    pprint(list(dataset.__dict__.keys()))

    checkpoint_filepath = os.path.join(model_folder, 'model.ckpt')
    with tf.Session() as sess:
        print_tensors_in_checkpoint_file(checkpoint_filepath, tensor_name='token_embedding/token_embedding_weights', all_tensors=True)
        print_tensors_in_checkpoint_file(checkpoint_filepath, tensor_name='token_embedding/token_embedding_weights', all_tensors=False)


if __name__ == '__main__':
    output_folder_name = 'en_2017-05-05_08-58-32-633799'
    epoch_number = 30
    model_name = 'conll_2003_en'
    delete_token_mappings = False
    prepare_pretrained_model_for_restoring(output_folder_name, epoch_number, model_name, delete_token_mappings)
    
#     model_name = 'mimic_glove_spacy_iobes'
#     model_folder = os.path.join('..', 'trained_models', model_name)
#     check_contents_of_dataset_and_model_checkpoint(model_folder)