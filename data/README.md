# Commands to download datasets

Go to the appropriate folder and run the commands given below. 
The test set is created by holding out a fraction of each user's data.

## EMINST (Called FEMNIST here)

For Simplicial-FL experiments:
> time ./preprocess.sh -s niid --sf 1.0 -k 100 -t user --tf 0.5


Takes about 30-60 minutes to run and occupies 25GB on hard disk


## SENT140
For Simplicial-FL experiments
> time ./preprocess.sh -s niid --sf 1.0 -k 50 -t user --tf 0.5


The above command was used to generate 438 train users and 439 test users for safe FL expt.

Occupies 1.2GB on hard dish


## SHAKESPEARE

For Simplicial-FL, experiments, run: 
> time ./preprocess.sh -s niid --sf 1.0 -k 100 -t user --tf 0.5

Runs under a minute and occupies 287MB
