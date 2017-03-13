

# Performance of trained models


| Dataset | State-of-the-art F1-score | NeuroNER F1-score | Model | Entities |
| :---         |     :---:      |          :---: | :---: |  :---: |
|  CoNLL-2003-en   | 90.1     | 90.1    | [Link](conll-2003-en/) |  Location, misc, organization, person |
| i2b2 2014     | 97.7*       | 97.8*      | [Link](i2b2-2014/) | 18 PHI types |
| MIMIC deid 2016     | 98.5*       | 98.6*      | [Link](mimic-2016/) | 18 PHI types |

`*` indicates that the F1-score  is binary, i.e. that the  predicted class of the named-entity  is not taken into account when computing the F1-score. This is most important  metric for some use cases such as de-identification, where  the main goal is to tag  as many PHI (protected health information) entities  as possible,  the correct prediction of the PHI class being secondary.