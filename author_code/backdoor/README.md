## The code is tested at:

Tensorflow version: 1.31.1
Python 2.7.15rc1




## Training:

### Train a baseline model with backdoor ratio r_p, for example, for r_p=0.3:

	python iclr_mnist_backdoor.py  --backdoor_portion 0.3


### Train a differential privacy model with outlier ratio r_p and noise scale sigma, for example, for sigma=0.5 and r_p=0.3:

	python iclr_mnist_backdoor.py --dpsgd True --noise_multiplier 0.5 --backdoor_portion 0.3


#### Training a model will automatically create a folder under "./models/" with a filename format similar to:
	train_lr0.150000_portion0.300000_batch200_epochs60_mb200_sigma0.500000_dpsgd

## Test:

(Training a DP model may take hours. You can use the pre-trained model available at folder "./models/")

### After a model is trained, suppose the model directory is "./models/train_lr0.150000_portion0.300000_batch200_epochs60_mb200_sigma0.500000_dpsgd", to test this model:

###	Test backdoor attack success rate and benign accuracy:

	python iclr_mnist_backdoor.py --test True --model_dir ./models/train_lr0.150000_portion0.300000_batch200_epochs60_mb200_sigma0.500000_dpsgd


### Test detection performance (AUPR score and AUROC score):

	python iclr_mnist_backdoor.py --detection True --model_dir ./models/train_lr0.150000_portion0.300000_batch200_epochs60_mb200_sigma0.500000_dpsgd





#############################################################################################


## Directories:


### ./iclr_mnist_backdoor.py 
	source code for model training and test

### ./models.tar.gz
	stores previously trained models; unpack it using the command "tar -xzvf models.tar.gz", and then you could use all models inside folder "./models" for testing. Note that the unpacked folder could be up to 14GB.

### ./privacy
	source code to support differentially-private SGD, inherited from https://github.com/tensorflow/privacy


### ./results
	results files that records privacy bound epsilon etc.
