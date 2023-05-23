## BGNLM in PyTorch
Implementation of BGNLM in PyTorch. Based on my Master's thesis.

Scripts and results based on datasets from the thesis.

## Usage
To run the scripts, make sure all the required modules are loaded.
Run python final_bgnlm.py -h to see options

### Example
The following command can be used to run 10 generations on Abalone age dataset with 40 extra features, using flow approximations and AIC-like prior:

```
python final_bgnlm.py -d 3 -g 10 -e 40 -f True --prior AIC 
```
