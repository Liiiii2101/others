# SureTypeSC
SureTypeSC is implementation of algorithm for regenotyping of single cell data.

## Getting Started

After unpacking the installation package, run the setup.py script by supplying command below


```
pip install .
```

### Prerequisites
* git-lfs https://git-lfs.github.com/
* python 2 (tested on Python 2.7.5)
* scikit >= v0.19.1 (http://scikit-learn.org/stable/)
* numpy >= v1.14.1 (http://www.numpy.org/)
* pandas >= v0.22.0 (https://pandas.pydata.org/)


### Usage

* create genome studio file (include name,chromosome,position, genotype, gencall score, x raw intensities, x normalized intensities, y raw instensities and y normalized intensities) [format, pandas dataframe]

```
import SureTypeSC as sc

df = sc.basic("/Users/apple/BeadArrayFiles-develop/library/Lishan/SingleCellExampleData/GTCs","/Users/apple/BeadArrayFiles-develop/library/Lishan/Manifest_and_Cluster/HumanKaryomap-12v1_A.bpm","/Users/apple/BeadArrayFiles-develop/library/Lishan/Manifest_and_Cluster/HumanKaryomap-12v1_A.egt","/Users/apple/BeadArrayFiles-develop/library/Lishan/SingleCellExampleData/Samplesheetr.csv",'\t')
```


* Index rearrangement (set index levels (including name chromosome and position))
```
dfs = sc.Data.create_from_frame(df)

dfs is Data type
```

*The attribute of Data type

```
dfs.restrict_chromosomes(['1','2']) (The parameters should be a list include the chromosome name)

dfs.apply_NC_threshold_3(threshold,inplace = True) (the threshold is based on the gencall score)
```

* M,A calculation
```
dfs.calculate_transformations_2()
```
* Load classifier
```
from suretypesc import loader

clf = loader('/Users/apple/SureTypeSC/clf/clf_30trees_7228_ratio1_lightweight.clf')

clf_2 = loader('/Users/apple/SureTypeSC/clf/clf_GDA_7228_ratio1_58cells.clf') (input should be the path of classifier)
```
* predict

```
result_rf = clf.predict_decorate(test,clftype='rf',inn=['m','a'])  (test is the dataset,clftype is the short for classifier like 'rf' or 'gda'. inn is the input feature)

result_gda = clf.predict_decorate(result_rf,clftype='gda',inn=['m','a'])
```

* Train and predict

```
train = sc.Trainer(result_rf,clfname='gda',inner=['m','a'],outer='rf_ratio:1.0_pred')

train.train('rf')

result_end = Tr.predict_decorate(result_gda,clftype='rf-gda',inn=['m','a'])
```


* save the result
```
result_end.save_complete_table('fulltable.txt',header=False)
```
* save the different modes

```
recall mode: result_end.save_mode('recall','recall.txt',header=False,ratio=1)

standard mode: result_end.save_mode('standard','st.txt',header=False,ratio=1)

precision mode: result_end.save_mode('precision','precision.txt',header=False,ratio=1)
```

```
The program enriches every sample in the input data by :

| Subcolumn name  | Meaning |
| ------------- | ------------- |
| rf_ratio:1_pred  | Random Forest prediction (binary)  |
| rf_ratio:1_prob  | Random Forest Score for the positive class |
| gda_ratio:1_prob | Gaussian Discriminant Analysis score for the positive class  | 
| gda_ratio:1_pred | Gaussian Disciminant Analysis prediction (binary) | 
| rf-gda_ratio:1_prob | combined 2-layer RF and GDA - probability score for the positive class | 
| rf-gda_ratio:1_pred | binary prediction of RF-GDA | 
```


<!---## Running the program - validation--->
<!--- Validation procedures are implemented in SureTypeSC.py. To run a validation procedure equivalent to basic configuration, run:--->
<!---```--->
<!---python genotyping/SureTypeSC.py config/GM12878_basic_test.conf--->
<!---```--->


### Contact
In case of any questions please contact Ivan Vogel (ivogel@sund.ku.dk)
