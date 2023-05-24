## BGNLM in PyTorch
Implementation of BGNLM in PyTorch.

Scripts and results are made to fit the datasets from my Master's thesis. Code to run and plot results, as well as plots used in thesis, can be found in results-folder.

## Usage
To run the scripts, make sure all the required modules are loaded.
To see options, run:
```
python final_bgnlm.py -h
```

### Example
The following command can be used to run 10 generations on Abalone age dataset with 40 extra features, using flow approximations, AIC-like prior, batch size of 500 and 600 epochs:

```
python final_bgnlm.py -d 3 -g 10 -e 40 -f True --prior AIC --batch_size 500 --epochs 600
```
