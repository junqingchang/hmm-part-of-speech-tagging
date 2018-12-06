# Part-Of-Speech Tagging Project SUTD
SUTD Term 6 Machine Learning Project

## Running
<Dataset> = {EN,FR,SG,CN} 
Datasets must be placed in their respective folder, for example, opening the terminal in the folder of the source code, the datasets have to be in the folder EN/dev.in for the EN dataset

```
$ python part2.py <Dataset>
$ python part3.py <Dataset>
$ python part4.py <Dataset>
$ python part5.py <Dataset>
```

The above runs the files train, dev.in and produces dev.p\<partnumber\>.out

To run on the test file, open part 5 and change the variables at the top to

```
filetest = "{}/test.in".format(running)
filep5out = "{}/test.p5.out".format(running)
```