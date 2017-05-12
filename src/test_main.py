'''
Tests for main.py
'''

import unittest
import os
import main
from shutil import rmtree

class TestMain(unittest.TestCase):

    testFolder = os.path.join(os.path.dirname(__file__), "test")
    outputFolder = os.path.join(testFolder, "output")

    def setUp(self):
        # delete the outputFolder
        if os.path.isdir(self.outputFolder):
            rmtree(self.outputFolder)

    def tearDown(self):
        # delete the outputFolder
        if os.path.isdir(self.outputFolder):
            rmtree(self.outputFolder)

    def test_ProvideOutputDir_CorrectlyOutputsToDir(self):
        ''' Sanity test to check if all proper model output files are created in the output folder'''
        main.main(argv=['', '--output_folder', self.outputFolder,'--parameters_filepath', os.path.join(self.testFolder, 'test-parameters-training.ini')])

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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()