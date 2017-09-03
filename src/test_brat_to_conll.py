'''
Tests for BRAT parsing.
'''

import unittest
import os
import brat_to_conll

GOLD_EXPANDED = [
        'Lorem ipsum dolor sit amet',
        'purus\nconvallis',
        'Nam consequat nisi quis eleifend sodales',
        'Aliquam lobortis',
        'Sed; fringilla tellus quis quam tempor; semper',
        ]

GOLD_SPLIT = [
        'Lorem ipsum dolor sit amet',
        'purus',
        'convallis',
        'Nam',
        'sodales',
        'Aliquam lobortis',
        'Sed;',
        'tellus',
        'tempor; semper',
        ]

class TestBrat(unittest.TestCase):
    test_folder = os.path.join(os.path.dirname(__file__), "test")
    txt = os.path.join(test_folder, 'test-brat.txt')
    ann = os.path.join(test_folder, 'test-brat.ann')

    def test_fragments(self):
        for expand, gold in zip([True, False], [GOLD_EXPANDED, GOLD_SPLIT]):
            print('expand_fragments={}'.format(expand))
            text, entities = brat_to_conll.get_entities_from_brat(self.txt,
                    self.ann, expand)
            self.assertTrue(len(entities) == len(gold))
            for i, entity in enumerate(entities):
                print('[parse {}] {}'.format(i, entity['text']))
                print('[truth {}] {}'.format(i, gold[i]))
                self.assertTrue(entity['text'] == gold[i])
            print('')

if __name__ == "__main__":
    unittest.main()

