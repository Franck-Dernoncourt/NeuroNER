The i2b2 2014 de-identification dataset can be downloaded at https://www.i2b2.org/NLP/DataSets/Download.php, section "2014 De-identification and Heart Disease Risk Factors Challenge". A data use agreement needs to be signed, which prevents us from directly providing the dataset with NeuroNER. You can send the completed data use agreement by email (schurchill@partners.org) or fax (617-525-4491; https://faxzero.com allows to send faxes free of charge).

The three following archive files need to be downloaded:

- Training: PHI Gold Set 1
- Training: PHI Gold Set 2
- Testing: PHI Gold Set - Fixed

One may use the Python script [`xml_to_brat.py`](xml_to_brat.py) to convert the i2b2 2014 de-identification dataset into the BRAT format by running `python xml_to_brat.py` (we tested the script with Python 3.5).  The script expects that the three archive files are uncompressed in the same folder, as shown below:


```
 Directory of C:\Users\Franck\GitHub\NeuroNER\data\i2b2_2014_deid

04/26/2017  05:23 PM    <DIR>          .
04/26/2017  05:23 PM    <DIR>          ..
04/26/2017  05:22 PM               495 readme.md
01/18/2016  07:13 PM    <DIR>          testing-PHI-Gold-fixed
04/26/2017  04:56 PM         1,085,448 testing-PHI-Gold-fixed.tar.gz
01/14/2016  08:18 PM    <DIR>          training-PHI-Gold-Set1
04/26/2017  04:55 PM         1,149,506 training-PHI-Gold-Set1.tar.gz
01/14/2016  08:18 PM    <DIR>          training-PHI-Gold-Set2
04/26/2017  04:55 PM           493,898 training-PHI-Gold-Set2.tar.gz
04/26/2017  05:15 PM             2,616 xml_to_brat.py
               5 File(s)      2,731,963 bytes
               5 Dir(s)   3,479,998,464 bytes free
```

Once the i2b2 2014 de-identification dataset is converted into the BRAT format, NeuroNER can be trained on it by specifying the following in the [`../../src/parameters.ini`](../../src/parameters.ini) configuration file.

```
dataset_text_folder = ../data/i2b2_2014_deid
```


If you use the i2b2 2014 de-identification dataset, please cite as:

 - Stubbs A, Uzuner O. (2015). "[Annotating longitudinal clinical narratives for de-identification: The 2014 i2b2/UTHealth corpus](http://www.ncbi.nlm.nih.gov/pubmed/26319540.)". J Biomed Inform. 2015 Aug 28. PII: S1532-0464(15)00182-3. DOI: 10.1016/j.jbi.2015.07.020.
 - Stubbs A, Kotfila C, Uzuner O. (2015). "[Automated systems for the de-identification of longitudinal clinical narratives: Overview of 2014 i2b2/UTHealth shared task Track 1](http://www.ncbi.nlm.nih.gov/pubmed/26225918)". J Biomed Inform. 2015 Jul 28. PII: S1532-0464(15)00117-3. DOI: 10.1016/j.jbi.2015.06.007.
