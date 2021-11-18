# Experimental Data Analysis Kit for EP05, EP06

It is time-consuming that doing some laboratory data analysis repeatly by existing softwares.To address this concern, I firstly set up the algorithm for precision and linearity analysis that comply with EP05, 06's regulations. Next, I used PyTK to take GUI to shape, so that our members executed analysis easily. This kit also provide data visualization function which can let user plot usable graph for reports via only clicking a bottun. 

## GUI

![EP05_06_analysisGUI](./images/GUI.png)

## run

``` 
python main.py
```

## output of precision

1. laboratory data

![rawdata_scatterplot_A-D](./EP05_result/rawdata_scatterplot_A-D.png)

2. report of nestANOVA analysis

![precision](./images/precision.png)

## graph output of linearity

1. 1 to 4 degree regression to fit laboratory data

![EP06_nonlinearity_rate_0.05](./EP06_result/EP06_nonlinearity_rate_0.05.png)

2. acceptable nonlinearity range of laboratory data

![regression_4th](./EP06_result/regression_4th.png)