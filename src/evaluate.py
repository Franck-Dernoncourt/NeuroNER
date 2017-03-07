'''

'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import os
import utils_plots
import json
import time
import sklearn.preprocessing
import utils_nlp


def assess_model(dataset, model_options, f_pred_prob, pred_probs, all_y_true, dataset_type, stats_graph_folder, epoch, update, verbose=False,
                 multilabel_prediction=False, save_proba=False):
    '''
    INPUT:
     - dataset is the full data set
     - model_options are all options in the models
     - data is a list of (x, y) pairs
     - dataset_type is either 'train' or 'test'
     - iterator indicates the batches when reading data
     - f_pred_prob is a function that takes x as input and output y_proba (i.e. the probabilities for each label)
     - epoch is the epoch number
     - update is the update number

     http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
     http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
     '''
    results = {}
    print('Generating plots for the {0} set'.format(dataset_type))
    y_true = all_y_true
    y_true_monolabel = all_y_true
    y_pred_monolabel = y_pred = pred_probs
    #print('y_pred[0:10]: {0}'.format(y_pred[0:10]))
    assert len(y_true) == len(y_pred)

    #print('y_true[0:10]: {0}'.format(y_true[0:10]))
    #print('y_pred[0:10]: {0}'.format(y_pred[0:10]))

    # Classification report
    classification_report = sklearn.metrics.classification_report(y_true, y_pred, labels=dataset.unique_label_indices_of_interest,
                                                                  target_names=dataset.unique_labels_of_interest, sample_weight=None, digits=4)

    utils_plots.plot_classification_report(classification_report, title='Classification report for epoch {0} update {2} in {1}\n'.format(epoch, dataset_type, update),
                                           cmap='RdBu')
    plt.savefig(os.path.join(stats_graph_folder, 'classification_report_for_epoch_{0:04d}_update_{2:05d}in_{1}.png'.format(epoch, dataset_type, update)), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
    plt.close()

    #print(classification_report)
    results['classification_report'] = classification_report
    if not multilabel_prediction:
        # for monolabel
        classification_report_monolabel = sklearn.metrics.classification_report(y_true_monolabel, y_pred_monolabel, labels=dataset.unique_label_indices_of_interest,
                                                                                target_names=dataset.unique_labels_of_interest, sample_weight=None, digits=4)
        #print('monolabel')
        #print(classification_report_monolabel)
        results['classification_report_monolabel'] = classification_report_monolabel

    # F1 scores
    results['f1_score'] = {}
    for f1_average_style in ['weighted', 'micro', 'macro']:
        results['f1_score'][f1_average_style] = sklearn.metrics.f1_score(y_true, y_pred, average=f1_average_style, labels=dataset.unique_label_indices_of_interest)
    results['f1_score']['per_label'] = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average=None, labels=dataset.unique_label_indices_of_interest)[2].tolist()
    if not multilabel_prediction:
        # for monolabel
        results['f1_score_monolabel'] = {}
        for f1_average_style in ['weighted', 'micro', 'macro']:
            results['f1_score_monolabel'][f1_average_style] = sklearn.metrics.f1_score(y_true_monolabel, y_pred_monolabel, average=f1_average_style,
                                                                                       labels=dataset.unique_label_indices_of_interest)
        results['f1_score_monolabel']['per_label'] = sklearn.metrics.precision_recall_fscore_support(y_true_monolabel, y_pred_monolabel, average=None,
                                                                                                     labels=dataset.unique_label_indices_of_interest)[2].tolist()

    # Confusion matrix
    if multilabel_prediction:
        results['confusion_matrix'] = 0
    else:
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true_monolabel, y_pred_monolabel, labels=dataset.unique_label_indices_of_interest)
        results['confusion_matrix'] = confusion_matrix.tolist()
        #print(confusion_matrix)
        title = 'Confusion matrix for epoch {0} update {2} in {1}\n'.format(epoch, dataset_type, update)
        xlabel = 'Predicted'
        ylabel = 'True'
        xticklabels = yticklabels = dataset.unique_labels_of_interest #range(model_options['ydim'])
        utils_plots.heatmap(confusion_matrix, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=True)
        plt.savefig(os.path.join(stats_graph_folder, 'confusion_matrix_for_epoch_{0:04d}_update_{2:05d}in_{1}.png'.format(epoch, dataset_type, update)), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
        plt.close()

    # Accuracy
    results['accuracy_score'] = sklearn.metrics.accuracy_score(y_true, y_pred)
    if not multilabel_prediction:
        results['accuracy_score_monolabel'] = sklearn.metrics.accuracy_score(y_true_monolabel, y_pred_monolabel)

    return results


def save_results(results, stats_graph_folder):
    '''
    Save results
    '''
    json.dump(results, open(os.path.join(stats_graph_folder, 'results.json'), 'w'), indent = 4, sort_keys=True)

def assess_and_save(results, dataset, model_options, pred_probs, all_y_true, stats_graph_folder, eidx, uidx, epoch_start_time, multilabel=False):
    '''
    Updates result dictionary by running assess_model and save it in 'results.json' file.
    Returns train_f1, valid_f1, and test_f1
    '''
    f_pred_prob = None
    result_update = {}

    for dataset_type in ['train', 'valid', 'test']:
        result_train = assess_model(dataset, model_options, f_pred_prob, pred_probs[dataset_type], all_y_true[dataset_type],
                                    dataset_type, stats_graph_folder, eidx, uidx, verbose=False, multilabel_prediction=multilabel)
        result_update[dataset_type] = result_train

    result_update['update'] = uidx
    result_update['time_elapsed_since_epoch_start'] = time.time() - epoch_start_time
    result_update['time_elapsed_since_train_start'] = time.time() - results['execution_details']['train_start']
    results['epoch'][eidx].append(result_update)
    #print('results: {0}'.format(results))
    save_results(results, stats_graph_folder)


    '''
    train_f1 = result_train['f1_score']['micro']
    valid_f1 = result_valid['f1_score']['micro']
    test_f1 = result_test['f1_score']['micro']
    '''
    #return train_f1, valid_f1, test_f1


def plot_f1_vs_epoch(results, stats_graph_folder, metric, from_json=False):
    '''
    Takes results dictionary and saves the f1 vs epoch plot in stats_graph_folder.
    from_json indicates if the results dictionary was loaded from results.json file.
    In this case, dictionary indexes are mapped from string to int.

    metric can be f1_score or accuracy
    '''
#     print('metric: {0}'.format(metric))
    save_results(results, stats_graph_folder)


    assert(metric in ['f1_score', 'accuracy_score', 'f1_score_monolabel', 'accuracy_score_monolabel','f1_conll'])

    if not from_json:
        epoch_idxs = sorted(results['epoch'].keys())
    else:
        epoch_idxs = sorted(map(int, results['epoch'].keys()))    # when loading json file
    f1_dict_all = {}
    f1_dict_all['train'] = []
    f1_dict_all['test'] = []
    f1_dict_all['valid'] = []
    for eidx in epoch_idxs:
        if not from_json:
            result_epoch = results['epoch'][eidx][-1]
        else:
            result_epoch = results['epoch'][str(eidx)][-1]    # when loading json file
        f1_dict_all['train'].append(result_epoch['train'][metric])
        f1_dict_all['valid'].append(result_epoch['valid'][metric])
        f1_dict_all['test'].append(result_epoch['test'][metric])


    # Plot micro f1 vs epoch for all classes
    plt.figure()
    plot_handles = []
    f1_all = {}
    for dataset_type in ['train', 'valid', 'test']:
        if dataset_type not in results: results[dataset_type] = {}
        if metric in ['f1_score', 'f1_score_monolabel','f1_conll']:
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
#     else:
#         stats_graph_folder = os.path.join('..','stats_graphs', folder_name)
#         result = json.load(open(os.path.join(stats_graph_folder, 'results.json'), 'r'))
#         for metric in ['accuracy_score', 'f1_score']:
#             plot_f1_vs_epoch(result, stats_graph_folder, metric, from_json=True)


def evaluate_model(results, dataset, all_predictions, all_y_true, stats_graph_folder, epoch_number, epoch_start_time, output_filepaths):
    results['epoch'][epoch_number] = []
    assess_and_save(results, dataset, None, all_predictions, all_y_true, stats_graph_folder, epoch_number, 0, epoch_start_time)
    plot_f1_vs_epoch(results, stats_graph_folder, 'f1_score')
    plot_f1_vs_epoch(results, stats_graph_folder, 'accuracy_score')
    # CoNLL evaluation script
    for dataset_type in ['train', 'valid', 'test']:
        conll_evaluation_script = os.path.join('.', 'conlleval')
        conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepaths[dataset_type])
        shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepaths[dataset_type], conll_output_filepath)
        print('shell_command: {0}'.format(shell_command))
        #subprocess.call([shell_command])
        os.system(shell_command)
        conll_parsed_output = utils_nlp.get_parsed_conll_output(conll_output_filepath)
        #print('conll_parsed_output: {0}'.format(conll_parsed_output))
        results['epoch'][epoch_number][0][dataset_type]['conll'] = conll_parsed_output
        results['epoch'][epoch_number][0][dataset_type]['f1_conll'] = {}
        results['epoch'][epoch_number][0][dataset_type]['f1_conll']['micro'] = results['epoch'][epoch_number][0][dataset_type]['conll']['all']['f1']

    plot_f1_vs_epoch(results, stats_graph_folder, 'f1_conll', from_json=False)