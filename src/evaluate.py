import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import os
import utils_plots
import json
import time
import utils_nlp


def assess_model(y_pred, y_true, labels, target_names, labels_with_o, target_names_with_o, dataset_type, stats_graph_folder, epoch_number, parameters,
                 evaluation_mode='bio', verbose=False):
    results = {}
    assert len(y_true) == len(y_pred)

    # Classification report
    classification_report = sklearn.metrics.classification_report(y_true, y_pred, labels=labels, target_names=target_names, sample_weight=None, digits=4)

    utils_plots.plot_classification_report(classification_report,
                                           title='Classification report for epoch {0} in {1} ({2} evaluation)\n'.format(epoch_number, dataset_type,
                                                                                                                        evaluation_mode),
                                           cmap='RdBu')
    plt.savefig(os.path.join(stats_graph_folder, 'classification_report_for_epoch_{0:04d}_in_{1}_{2}_evaluation.{3}'.format(epoch_number, dataset_type,
                                                                                                                            evaluation_mode, parameters['plot_format'])),
                dpi=300, format=parameters['plot_format'], bbox_inches='tight')
    plt.close()
    results['classification_report'] = classification_report

    # F1 scores
    results['f1_score'] = {}
    for f1_average_style in ['weighted', 'micro', 'macro']:
        results['f1_score'][f1_average_style] = sklearn.metrics.f1_score(y_true, y_pred, average=f1_average_style, labels=labels)*100
    results['f1_score']['per_label'] = [x*100 for x in sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)[2].tolist()]

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels_with_o)
    results['confusion_matrix'] = confusion_matrix.tolist()
    title = 'Confusion matrix for epoch {0} in {1} ({2} evaluation)\n'.format(epoch_number, dataset_type, evaluation_mode)
    xlabel = 'Predicted'
    ylabel = 'True'
    xticklabels = yticklabels = target_names_with_o
    utils_plots.heatmap(confusion_matrix, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=True, fmt="%d", 
                        remove_diagonal=True)
    plt.savefig(os.path.join(stats_graph_folder, 'confusion_matrix_for_epoch_{0:04d}_in_{1}_{2}_evaluation.{3}'.format(epoch_number, dataset_type,
                                                                                                                       evaluation_mode, parameters['plot_format'])),
                dpi=300, format=parameters['plot_format'], bbox_inches='tight')
    plt.close()

    # Accuracy
    results['accuracy_score'] = sklearn.metrics.accuracy_score(y_true, y_pred)*100

    return results


def save_results(results, stats_graph_folder):
    '''
    Save results
    '''
    json.dump(results, open(os.path.join(stats_graph_folder, 'results.json'), 'w'), indent = 4, sort_keys=True)


def plot_f1_vs_epoch(results, stats_graph_folder, metric, parameters, from_json=False):
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
        epoch_idxs = sorted(map(int, results['epoch'].keys()))    # when loading json file

    dataset_types = []
    for dataset_type in ['train', 'valid', 'test']:
        if dataset_type in results['epoch'][epoch_idxs[0]][-1]:
            dataset_types.append(dataset_type)
    if len(dataset_type) < 2:
        return

    f1_dict_all = {}
    for dataset_type in dataset_types:
        f1_dict_all[dataset_type] = []
    for eidx in epoch_idxs:
        if not from_json:
            result_epoch = results['epoch'][eidx][-1]
        else:
            result_epoch = results['epoch'][str(eidx)][-1]    # when loading json file
        for dataset_type in dataset_types:
            f1_dict_all[dataset_type].append(result_epoch[dataset_type][metric])


    # Plot micro f1 vs epoch for all classes
    plt.figure()
    plot_handles = []
    f1_all = {}
    for dataset_type in dataset_types:
        if dataset_type not in results: results[dataset_type] = {}
        if metric in ['f1_score', 'f1_conll']:
            f1 = [f1_dict['micro'] for f1_dict in f1_dict_all[dataset_type]]
        else:
            f1 = [score_value for score_value in f1_dict_all[dataset_type]]
        results[dataset_type]['best_{0}'.format(metric)] = max(f1)
        results[dataset_type]['epoch_for_best_{0}'.format(metric)] = int(np.asarray(f1).argmax())
        f1_all[dataset_type] = f1
        plot_handles.extend(plt.plot(epoch_idxs, f1, '-', label=dataset_type + ' (max: {0:.4f})'.format(results[dataset_type]['best_{0}'.format(metric)])))
    # Record the best values according to the best epoch for valid
    best_epoch = results['valid']['epoch_for_best_{0}'.format(metric)]
    plt.axvline(x=best_epoch, color='k', linestyle=':')   # Add a vertical line at best epoch for valid
    for dataset_type in dataset_types:
        best_score_based_on_valid = f1_all[dataset_type][best_epoch]
        results[dataset_type]['best_{0}_based_on_valid'.format(metric)] = best_score_based_on_valid
        if dataset_type == 'test':
            plot_handles.append(plt.axhline(y=best_score_based_on_valid, label=dataset_type + ' (best: {0:.4f})'.format(best_score_based_on_valid),
                                            color='k', linestyle=':'))
        else:
            plt.axhline(y=best_score_based_on_valid, label='{0:.4f}'.format(best_score_based_on_valid), color='k', linestyle=':')
    title = '{0} vs epoch number for all classes\n'.format(metric)
    xlabel = 'epoch number'
    ylabel = metric
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles=plot_handles, loc=0)
    plt.savefig(os.path.join(stats_graph_folder, '{0}_vs_epoch_for_all_classes.{1}'.format(metric, parameters['plot_format'])))
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


def remap_labels(y_pred, y_true, dataset, evaluation_mode='bio'):
    '''
    y_pred: list of predicted labels
    y_true: list of gold labels
    evaluation_mode: 'bio', 'token', or 'binary'

    Both y_pred and y_true must use label indices and names specified in the dataset
#     (dataset.unique_label_indices_of_interest, dataset.unique_label_indices_of_interest).
    '''
    all_unique_labels = dataset.unique_labels
    if evaluation_mode == 'bio':
        # sort label to index
        new_label_names = all_unique_labels[:]
        new_label_names.remove('O')
        new_label_names.sort(key=lambda x: (utils_nlp.remove_bio_from_label_name(x), x))
        new_label_names.append('O')
        new_label_indices = list(range(len(new_label_names)))
        new_label_to_index = dict(zip(new_label_names, new_label_indices))

        remap_index = {}
        for i, label_name in enumerate(new_label_names):
            label_index = dataset.label_to_index[label_name]
            remap_index[label_index] = i

    elif evaluation_mode == 'token':
        new_label_names = set()
        for label_name in all_unique_labels:
            if label_name == 'O':
                continue
            new_label_name = utils_nlp.remove_bio_from_label_name(label_name)
            new_label_names.add(new_label_name)
        new_label_names = sorted(list(new_label_names)) + ['O']
        new_label_indices = list(range(len(new_label_names)))
        new_label_to_index = dict(zip(new_label_names, new_label_indices))

        remap_index = {}
        for label_name in all_unique_labels:
            new_label_name = utils_nlp.remove_bio_from_label_name(label_name)
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
        raise ValueError("evaluation_mode must be either 'bio', 'token', or 'binary'.")

    new_y_pred = [ remap_index[label_index] for label_index in y_pred ]
    new_y_true = [ remap_index[label_index] for label_index in y_true ]

    new_label_indices_with_o = new_label_indices[:]
    new_label_names_with_o = new_label_names[:]
    new_label_names.remove('O')
    new_label_indices.remove(new_label_to_index['O'])

    return new_y_pred, new_y_true, new_label_indices, new_label_names, new_label_indices_with_o, new_label_names_with_o


def evaluate_model(results, dataset, y_pred_all, y_true_all, stats_graph_folder, epoch_number, epoch_start_time, output_filepaths, parameters, verbose=False):
    results['execution_details']['num_epochs'] = epoch_number
    results['epoch'][epoch_number] = []
    result_update = {}

    for dataset_type in ['train', 'valid', 'test']:
        if dataset_type not in output_filepaths.keys():
            continue
        print('Generating plots for the {0} set'.format(dataset_type))
        result_update[dataset_type] = {}
        y_pred_original = y_pred_all[dataset_type]
        y_true_original = y_true_all[dataset_type]

        for evaluation_mode in ['bio', 'token', 'binary']:
            y_pred, y_true, label_indices, label_names, label_indices_with_o, label_names_with_o = remap_labels(y_pred_original, y_true_original, dataset,
                                                                                                                evaluation_mode=evaluation_mode)
            result_update[dataset_type][evaluation_mode] = assess_model(y_pred, y_true, label_indices, label_names, label_indices_with_o, label_names_with_o,
                                                                        dataset_type, stats_graph_folder, epoch_number,  parameters, evaluation_mode=evaluation_mode,
                                                                        verbose=verbose)
            if parameters['main_evaluation_mode'] == evaluation_mode:
                result_update[dataset_type].update(result_update[dataset_type][evaluation_mode])

    result_update['time_elapsed_since_epoch_start'] = time.time() - epoch_start_time
    result_update['time_elapsed_since_train_start'] = time.time() - results['execution_details']['train_start']
    results['epoch'][epoch_number].append(result_update)

    # CoNLL evaluation script
    for dataset_type in ['train', 'valid', 'test']:
        if dataset_type not in output_filepaths.keys():
            continue
        conll_evaluation_script = os.path.join('.', 'conlleval')
        conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepaths[dataset_type])
        shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepaths[dataset_type], conll_output_filepath)
        print('shell_command: {0}'.format(shell_command))
        os.system(shell_command)
        conll_parsed_output = utils_nlp.get_parsed_conll_output(conll_output_filepath)
        results['epoch'][epoch_number][0][dataset_type]['conll'] = conll_parsed_output
        results['epoch'][epoch_number][0][dataset_type]['f1_conll'] = {}
        results['epoch'][epoch_number][0][dataset_type]['f1_conll']['micro'] = results['epoch'][epoch_number][0][dataset_type]['conll']['all']['f1']
        if parameters['main_evaluation_mode'] == 'conll':
            results['epoch'][epoch_number][0][dataset_type]['f1_score'] = {}
            results['epoch'][epoch_number][0][dataset_type]['f1_score']['micro'] = results['epoch'][epoch_number][0][dataset_type]['conll']['all']['f1']
            results['epoch'][epoch_number][0][dataset_type]['accuracy_score'] = results['epoch'][epoch_number][0][dataset_type]['conll']['all']['accuracy']
            utils_plots.plot_classification_report(results['epoch'][epoch_number][0][dataset_type]['conll'],
                title='Classification report for epoch {0} in {1} ({2} evaluation)\n'.format(epoch_number, dataset_type, 'conll'),
                cmap='RdBu', from_conll_json=True)
            plt.savefig(os.path.join(stats_graph_folder, 'classification_report_for_epoch_{0:04d}_in_{1}_conll_evaluation.{3}'.format(epoch_number, dataset_type,
                                                                                                                                    evaluation_mode, parameters['plot_format'])),
                        dpi=300, format=parameters['plot_format'], bbox_inches='tight')
            plt.close()

    if  parameters['train_model'] and 'train' in output_filepaths.keys() and 'valid' in output_filepaths.keys():
        plot_f1_vs_epoch(results, stats_graph_folder, 'f1_score', parameters)
        plot_f1_vs_epoch(results, stats_graph_folder, 'accuracy_score', parameters)
        plot_f1_vs_epoch(results, stats_graph_folder, 'f1_conll', parameters)

    results['execution_details']['train_duration'] = time.time() - results['execution_details']['train_start']
    save_results(results, stats_graph_folder)