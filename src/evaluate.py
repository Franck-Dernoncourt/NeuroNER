import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import os
import utils_plots
import json
import time
import sklearn.preprocessing
import utils_nlp
import copy



def assess_model(y_pred, y_true, labels, target_names, dataset_type, stats_graph_folder, epoch_number, evaluation_mode='BIO', verbose=False):
    '''
    INPUT:
     - y_pred is the list of predicted labels
     - y_true is the list of gold labels
     - dataset_type is either 'train' or 'test'
     - epoch_number is the epoch number

     '''
    results = {}

    assert len(y_true) == len(y_pred)

    # Classification report
    classification_report = sklearn.metrics.classification_report(y_true, y_pred, labels=labels, target_names=target_names, sample_weight=None, digits=4)

    utils_plots.plot_classification_report(classification_report, title='Classification report for epoch {0} in {1} ({2} evaluation)\n'.format(epoch_number, dataset_type, evaluation_mode),
                                           cmap='RdBu')
    plt.savefig(os.path.join(stats_graph_folder, 'classification_report_for_epoch_{0:04d}_in_{1}_{2}_evaluation.png'.format(epoch_number, dataset_type, evaluation_mode)), dpi=300, format='png', bbox_inches='tight') # use 
    plt.close()
    results['classification_report'] = classification_report

    # F1 scores
    results['f1_score'] = {}
    for f1_average_style in ['weighted', 'micro', 'macro']:
        results['f1_score'][f1_average_style] = sklearn.metrics.f1_score(y_true, y_pred, average=f1_average_style, labels=labels)
    results['f1_score']['per_label'] = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)[2].tolist()
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
    results['confusion_matrix'] = confusion_matrix.tolist()
    
    title = 'Confusion matrix for epoch {0} in {1} ({2} evaluation)\n'.format(epoch_number, dataset_type, evaluation_mode)
    xlabel = 'Predicted'
    ylabel = 'True'
    xticklabels = yticklabels = labels 
    utils_plots.heatmap(confusion_matrix, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=True)
    plt.savefig(os.path.join(stats_graph_folder, 'confusion_matrix_for_epoch_{0:04d}_in_{1}_{2}_evaluation.png'.format(epoch_number, dataset_type, evaluation_mode)), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
    plt.close()

    results['accuracy_score'] = sklearn.metrics.accuracy_score(y_true, y_pred)

    return results


def save_results(results, stats_graph_folder):
    '''
    Save results
    '''
    json.dump(results, open(os.path.join(stats_graph_folder, 'results.json'), 'w'), indent = 4, sort_keys=True)
    
def plot_f1_vs_epoch(results, stats_graph_folder, metric, from_json=False):
    '''
    Takes results dictionary and saves the f1 vs epoch plot in stats_graph_folder.
    from_json indicates if the results dictionary was loaded from results.json file.
    In this case, dictionary indexes are mapped from string to int.

    metric can be f1_score or accuracy
    '''

    assert(metric in ['f1_score', 'accuracy_score', 'f1_conll'])

    if not from_json:
        epoch_idxs = sorted(results['epoch'].keys())
    else:
        epoch_idxs = sorted(map(int, results['epoch'].keys()))   
    f1_dict_all = {}
    f1_dict_all['train'] = []
    f1_dict_all['test'] = []
    f1_dict_all['valid'] = []
    for eidx in epoch_idxs:
        if not from_json:
            result_epoch = results['epoch'][eidx][-1]
        else:
            result_epoch = results['epoch'][str(eidx)][-1]    
        f1_dict_all['train'].append(result_epoch['train'][metric])
        f1_dict_all['valid'].append(result_epoch['valid'][metric])
        f1_dict_all['test'].append(result_epoch['test'][metric])


    # Plot micro f1 vs epoch for all classes
    plt.figure()
    plot_handles = []
    f1_all = {}
    for dataset_type in ['train', 'valid', 'test']:
        if dataset_type not in results: results[dataset_type] = {}
        if metric in ['f1_score', 'f1_conll']:
            f1 = [f1_dict['micro'] for f1_dict in f1_dict_all[dataset_type]]
        else:
            f1 = [score_value for score_value in f1_dict_all[dataset_type]]
        results[dataset_type]['best_{0}'.format(metric)] = max(f1)
        results[dataset_type]['epoch_for_best_{0}'.format(metric)] = int(np.asarray(f1).argmax())
        f1_all[dataset_type] = f1
#         print(dataset_type)
#         print(metric)
#         print(results[dataset_type]['best_{0}'.format(metric)])
        plot_handles.extend(plt.plot(epoch_idxs, f1, '-', label=dataset_type + ' (max: {0:.4f})'.format(results[dataset_type]['best_{0}'.format(metric)])))
    # Record the best values according to the best epoch for valid
    best_epoch = results['valid']['epoch_for_best_{0}'.format(metric)]
    plt.axvline(x=best_epoch, color='k', linestyle=':')   # Add a vertical line at best epoch for valid
    for dataset_type in ['train', 'valid', 'test']:
        best_score_based_on_valid = f1_all[dataset_type][best_epoch]
        results[dataset_type]['best_{0}_based_on_valid'.format(metric)] = best_score_based_on_valid
        if dataset_type == 'test':
            plot_handles.append(plt.axhline(y=best_score_based_on_valid, label=dataset_type + ' (best: {0:.4f})'.format(best_score_based_on_valid), color='k', linestyle=':'))
        else:
            plt.axhline(y=best_score_based_on_valid, label='{0:.4f}'.format(best_score_based_on_valid), color='k', linestyle=':')
    title = '{0} vs epoch number for all classes\n'.format(metric)
    xlabel = 'epoch number'
    ylabel = metric
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles=plot_handles, loc=0)
    plt.savefig(os.path.join(stats_graph_folder, '{0}_vs_epoch_for_all_classes.png'.format(metric)))
    plt.close()



    if metric != 'f1_score': return

    # Plot f1 vs epoch per label
    for dataset_type in ['train', 'valid', 'test']:
        #print('dataset_type: {0}'.format(dataset_type))
        plt.figure()
        num_class = len(f1_dict_all[dataset_type][0]['per_label'])
        f1_per_label = {}
        plot_handles = []
        for label_idx in range(num_class):
            f1_per_label[label_idx] = []

        #print('num_class: {0}'.format(num_class))
        for f1_dict in f1_dict_all[dataset_type]:
            f1_labels = f1_dict['per_label']
            for label_idx in range(num_class):
                #print('label_idx: {0}'.format(label_idx))
                f1_per_label[label_idx].append(f1_labels[label_idx])

        for label_idx in range(num_class):
            plot_handles.extend(plt.plot(epoch_idxs, f1_per_label[label_idx], '-', label=str(label_idx) + ' (max: {0:.4f})'.format(max(f1_per_label[label_idx]))))
        title = 'f1 score vs epoch number per label in {0} set\n'.format(dataset_type)
        xlabel = 'epoch number'
        ylabel = 'f1 score'
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(handles=plot_handles, loc=0)
        plt.savefig(os.path.join(stats_graph_folder, 'f1_vs_epoch_per_label_in_{0}_set.png'.format(dataset_type)))#.
        plt.close()

def result_to_plot(folder_name=None):
    '''
    Loads results.json file in the ../stats_graphs/folder_name, and plot f1 vs epoch.
    Use for debugging purposes, or in case the program stopped due to error in plot_f1_vs_epoch.
    '''
    stats_graph_folder=os.path.join('..', 'stats_graphs')
    if folder_name == None:
        # Getting a list of all subdirectories in the current directory. Not recursive.
        subfolders = os.listdir(stats_graph_folder)
    else:
        subfolders = [folder_name]

    for subfolder in subfolders:
        subfolder_filepath = os.path.join(stats_graph_folder, subfolder)
        result_filepath = os.path.join(stats_graph_folder, subfolder, 'results.json')
        if not os.path.isfile(result_filepath): continue
        result = json.load(open(result_filepath, 'r'))
        for metric in ['accuracy_score', 'f1_score']:
            plot_f1_vs_epoch(result, subfolder_filepath, metric, from_json=True)


def remove_bio_from_label_name(label_name):
    if label_name[:2] in ['B-', 'I-']:
        new_label_name = label_name[2:]
    else:
        assert(label_name == 'O')
        new_label_name = label_name
    return new_label_name

def remap_labels(y_pred, y_true, dataset, evaluation_mode='BIO'):
    '''
    y_pred: list of predicted labels
    y_true: list of gold labels
    evaluation_mode: 'BIO', 'token', or 'binary'
    
    Both y_pred and y_true must use label indices and names specified in the dataset (dataset.unique_label_indices_of_interest, dataset.unique_label_indices_of_interest).
    '''
    all_unique_labels = dataset.unique_labels

    if evaluation_mode == 'BIO':
        label_indices = dataset.unique_label_indices_of_interest
        label_names = dataset.unique_labels_of_interest
        return y_pred, y_true, label_indices, label_names 
    elif evaluation_mode == 'token':
        new_label_names = set()
        for label_name in all_unique_labels:
            new_label_name = remove_bio_from_label_name(label_name)  
            new_label_names.add(new_label_name)
        new_label_names = sorted(list(new_label_names))
        new_label_indices = list(range(len(new_label_names)))
        new_label_to_index = dict(zip(new_label_names, new_label_indices))
        
        remap_index = {}
        for label_name in all_unique_labels:
            new_label_name = remove_bio_from_label_name(label_name)  
            label_index = dataset.label_to_index[label_name]
            remap_index[label_index] = new_label_to_index[new_label_name]
        
    elif evaluation_mode == 'binary':
        new_label_names = ['NAMED_ENTITY', 'O']
        new_label_indices = [0, 1]
        new_label_to_index = dict(zip(new_label_names, new_label_indices))

        remap_index = {}
        for label_name in all_unique_labels:
            new_label_name = 'O'
            if label_name != 'O':
                new_label_name = 'NAMED_ENTITY'
            label_index = dataset.label_to_index[label_name]
            remap_index[label_index] = new_label_to_index[new_label_name]
                
    else:
        raise ValueError("evaluation_mode must be either 'BIO', 'token', or 'binary'.")

    new_y_pred = [ remap_index[label_index] for label_index in y_pred ]
    new_y_true = [ remap_index[label_index] for label_index in y_true ]

    new_label_names.remove('O')
    new_label_indices.remove(new_label_to_index['O'])

    return new_y_pred, new_y_true, new_label_indices, new_label_names 
        
  
def evaluate_model(results, dataset, y_pred_all, y_true_all, stats_graph_folder, epoch_number, epoch_start_time, output_filepaths, parameters, verbose=False):
    results['epoch'][epoch_number] = []
    result_update = {}

    for dataset_type in ['train', 'valid', 'test']:
        print('Generating plots for the {0} set'.format(dataset_type))
        result_update[dataset_type] = {}
        y_pred_original = y_pred_all[dataset_type]
        y_true_original = y_true_all[dataset_type]
        
        for evaluation_mode in ['BIO', 'token', 'binary']:
            y_pred, y_true, label_indices, label_names = remap_labels(y_pred_original, y_true_original, dataset, evaluation_mode=evaluation_mode)
            result_update[dataset_type][evaluation_mode] = assess_model(y_pred, y_true, label_indices, label_names, dataset_type, stats_graph_folder, epoch_number, 
                                                       evaluation_mode=evaluation_mode, verbose=verbose)
            if parameters['main_evaluation_mode'] == evaluation_mode:
                result_update[dataset_type].update(result_update[dataset_type][evaluation_mode]) #copy.deepcopy(result_update[dataset_type][evaluation_mode]) 
                
    result_update['time_elapsed_since_epoch_start'] = time.time() - epoch_start_time
    result_update['time_elapsed_since_train_start'] = time.time() - results['execution_details']['train_start']
    results['epoch'][epoch_number].append(result_update)
    
    plot_f1_vs_epoch(results, stats_graph_folder, 'f1_score')
    plot_f1_vs_epoch(results, stats_graph_folder, 'accuracy_score')
    
    # CoNLL evaluation script
    for dataset_type in ['train', 'valid', 'test']:
        conll_evaluation_script = os.path.join('.', 'conlleval')
        conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepaths[dataset_type])
        shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepaths[dataset_type], conll_output_filepath)
        print('shell_command: {0}'.format(shell_command))
        
        os.system(shell_command)
        conll_parsed_output = utils_nlp.get_parsed_conll_output(conll_output_filepath)
        
        results['epoch'][epoch_number][0][dataset_type]['conll'] = conll_parsed_output
        results['epoch'][epoch_number][0][dataset_type]['f1_conll'] = {}
        results['epoch'][epoch_number][0][dataset_type]['f1_conll']['micro'] = results['epoch'][epoch_number][0][dataset_type]['conll']['all']['f1']

    plot_f1_vs_epoch(results, stats_graph_folder, 'f1_conll')

    results['execution_details']['train_duration'] = time.time() - results['execution_details']['train_start']
    save_results(results, stats_graph_folder)