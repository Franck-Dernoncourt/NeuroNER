

# Performance of trained models


| Dataset | State-of-the-art F1-score | NeuroNER F1-score | Model | Entities |
| :---         |     :---:      |          :---: | :---: |  :---: |
| CoNLL-2003-en   | 90.9     | 90.5    | [Link](conll_2003_en/) |  Location, misc, organization, person |
| i2b2 2014     | 97.9*       | 97.7*      | [Link](i2b2_2014_glove_stanford_bioes/) | 18 PHI types |
| MIMIC deid 2016     | 98.5*       | 98.6*      | [Link](mimic_glove_stanford_bioes/) | 18 PHI types |

PHI = [protected health information](https://en.wikipedia.org/wiki/Protected_health_information).

`*` indicates that the F1-score  is binary, i.e. that the  predicted class of the named-entity  is not taken into account when computing the F1-score. This is most important  metric for some use cases such as de-identification, where  the main goal is to tag  as many PHI (protected health information) entities  as possible,  the correct prediction of the PHI class being secondary.
