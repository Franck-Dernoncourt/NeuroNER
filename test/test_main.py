'''
Tests for main.py
'''

import os
from shutil import rmtree
import unittest

from neuroner import neuromodel

class TestMain(unittest.TestCase):

    outputFolder = os.path.join('.', "output")
    test_param_file = os.path.join('.', 'test-parameters-training.ini')

    def setUp(self):
        # delete the outputFolder
        if os.path.isdir(self.outputFolder):
            rmtree(self.outputFolder)

    def tearDown(self):
        # delete the outputFolder
        if os.path.isdir(self.outputFolder):
            rmtree(self.outputFolder)

    def test_ProvideOutputDir_CorrectlyOutputsToDir(self):
        """
        Sanity test to check if all proper model output files are created in the output folder
        """
        nn = neuromodel.NeuroNER(output_folder=self.outputFolder, parameters_filepath=self.test_param_file)
        nn.fit()

        # find the newest dir, from: http://stackoverflow.com/questions/2014554/find-the-newest-folder-in-a-directory-in-python
        run_outputdir = max([os.path.join(self.outputFolder,d) for d in os.listdir(self.outputFolder)], key=os.path.getmtime)

        # assert the model has been written to files
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'checkpoint')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'dataset.pickle')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'model_00001.ckpt.data-00000-of-00001')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'model_00001.ckpt.index')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'model_00001.ckpt.meta')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'model_00002.ckpt.data-00000-of-00001')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'model_00002.ckpt.index')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'model_00002.ckpt.meta')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'parameters.ini')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'projector_config.pbtxt')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'tensorboard_metadata_characters.tsv')))
        self.assertTrue(os.path.isfile(os.path.join(run_outputdir, 'model', 'tensorboard_metadata_tokens.tsv')))