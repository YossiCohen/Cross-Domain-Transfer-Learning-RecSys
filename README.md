# Cross Domain Transfer Learning RecSys

Notice that you need Amazon's data to run this code.
It's location is hard coded in the .py files


You can download all the relevant CSV files from:

http://jmcauley.ucsd.edu/data/amazon/
Alternative: https://drive.google.com/file/d/0B2dMmrBCQarIX2RoTV9xZFBCYzA/view?usp=sharing

After downloading the data, you need to run Steps[1..5] in .\src\ folder to generate summation of the raw data

After the summary finished use RMGM_Boost_multiple_runs.py to run experiments.

in any case you will need to update the Data folders - i.e. change:
 ```
 DATA_ROOT = "C:\\RS\\Amazon\\Tresholds\\MINIMAL_THRESHOLD_30\\"
 ```
 To a value relevant for your machine.
 
 Good luck!
