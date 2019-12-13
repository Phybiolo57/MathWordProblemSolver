## Experimental Evaluation of Math Word Problem Solver

### Our contributions:
- Creation of a new English corpus of arithmetic word problems dealing with {+, -, *, /} and linear variables only. We named this as Dolphin300 which is a subset of publicly available Dolphin18k.
- Creation of equation templates and normalizing equations in par with Math23K dataset [1].
- Experimental evaluation of T-RNN and retrieval baselines on Math23K, Dolphin300 and Dolphin1500.

## Sample data processing and cleaning:

- What is the value of five times the sum of twice of three-fourths and nine? ==> What is the value of 5 times the sum of 2 of 3/4 and 9?
- please help with math! What percentage of 306,800 equals eight thousands? ==> please help with math! What percentage of 306800 equals 8000?
- help!!!!!!!(please) i cant figure this out!? what is the sum of 4 2/5  and 17 3/7  ?  ==> help!!!!!!!(please) i cant figure this out!? what is the sum of   22/5  and 122/7 ? 
- math homework help?  question: 15 is 25%  of what number? ==> math homework help?  question: 15 is 25/100  of what number?

## list of folders:
- Web scraping: it contains the code (OriginalDataExtractor.py) to scrap the math word problems from yahoo answers. A basic data cleaning has also been carried out (CleanVersionExtractor.py) to get the questions in the desired format.
- Data_Cleaning: it contains the code for the data cleaning of Dolphin DataSet, MWP_DataCleaning.py file has all the rule based and filtering logic for transforming the candidate dolphin datasets to cleaned templates. Inside cleaned_data_examples folder, uncleaned_dolphin_data.csv  contains the raw data from dolphin dataset and filtered_cleaned_dolphin_data.json contains the filtered out cleaned template json from the csv.
- T-RNN and baselines: contain T-RNN code and baseline models, output folder within contains runs of retrieval model (named as pred_*.txt) and runs of TRNN (named as run_*.txt)


## Implementation:

- Implemented in >=py3.6 environment with pytorch
- We used part of T-RNN code [1] and added some more implementations for Math23K
- We used replicate.py and MWP_DataCleaning.ipynb to replicate data and process raw noisy Dolphin18k data.
- Finally we obtain Dolphin300 and Dolphin1500 after running replicate.py on Dolphin300.
- Run T-RNN code as :
$ python T-RNN/src/main.py 
(Please see the details in the code to change input files)

### T-RNN for Math23K
- In the template prediction module, we use a pre-trained word embedding with 128 units, a two-layer Bi-LSTM with 256 hidden units as encoder, a two-layer LSTM with 512 hidden units as decoder. As to the optimizer, we use Adam with learning rate set to 1e−3, β1 = 0.9 and β2 = 0.99. In the answer generation module, we use a embedding layer with 100 units, a two-layer Bi-LSTM with 160 hidden units. SGD with learning rate 0.01 and momentum factor 0.9 is used to optimize this module. In both components, the number of epochs, mini-batch size and dropout rate are set 100, 32 and 0.5 respectively. 

### T-RNN for Dolphin1500 & Dolphin300

- Template prediction module: 128 units, two-layer Bi-LSTM with 256 hidden units as encoder, a two-layer LSTM with 512 hidden units as decoder.
ADAM optimizer with default parameters.
Answer generation module - embedding layer with 100 units, a two-layer Bi-LSTM with 160 hidden units, RNN classes = 4.SGD with learning rate 0.01 and momentum factor 0.9 is used to optimize this module. the number of epochs, mini-batch size and dropout rate are set 50, 32 and 0.5 respectively.


References:

[1] Lei Wang, Dongxiang Zhang, Jipeng Zhang, Xing Xu, Lianli Gao, Bingtian Dai, and Heng Tao Shen. Template-based math word problem solvers with recursive neural networks. 2019.

[2] Yan Wang, Xiaojiang Liu, and Shuming Shi. Deep neural solver for math word problems. In Proceedings of the 2017 Conference on Empirical Methods in Natural  Language Processing, pages 845–854, 2017.
