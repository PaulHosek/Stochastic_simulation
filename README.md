# Area of the Mandelbrot set

## Contributors:

* Paul Hosek (2753446)
* Marcel van de Lagemaat (10886699)

## Program Overview
Goal of this work is to investigate the area of the mandelbrot set using Monte Carlo Integration.
Here, pure random sampling, latin hypercube sampling and orthogonal sampling are compared.
Later, antithetic variates are used to reduce variance.
Effectiveness of all approaches is compared.

## Requirements
* Python 3.9+
* math
* numpy
* matplotlib
* numba


### Running the code

All results are aggregated into a single Jupyter Notebook named MarcelvandeLagemaat_10886699_PaulHosek_12637033_1.ipynb.
This notebook imports all relevant files and generates the plots and data found in the report.


### Repository structure


| File Name           | Description                                                                                                                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gen_mandelbrot.py   | Functions related to the mandelbrot including monte-carlo integration and generating a complex grid to draw the mandelbrot on.                                                                       |
| sampling_alg.py     | Includes functions to generate points on a 2-d plane based on pure random, latin hypercube and orthogonal sampling. Further, a function converting points into their antithetic inverse is included. |
| area.py             | Compares different sampling algorithms on their convergence towards the area of the mandelbrot set. Results are saved in pickle files for further processing.                                        |
| figures (directory) | Includes all figures generated from plotting and drawing the mandelbrot set.                                                                                                                         |
| legacy (directory)  | Lists files not relevant for the current version of the program.                                                                                                                                     |
|                     |                                                                                                                                                                                                      |
|                     |                                                                                                                                                                                                      |
|                     |                                                                                                                                                                                                      |




### Results


```sh
exp_name
│
├── best_results
│   └── best_individuals
├── evoman_logs.txt
└── plots
```

## Key Source Files
> **task2.py**  
> CMA-ES and MO-CMA-ES used to defeat multiple enemies

> **utils.py**  
> defining the fuctions used for tuning of the hyperparameters

> **task2_tuning.py**  
> tuning the hyperparameters

> **parameter_tests.py**  
> testing the parameters from the framework

> **statistical_tests.py**  
> Run pairwise t-test on all enemies for multiple set of weights.
> Run wilcoxon signed-rank test on gain for all enemies.


> **plot_bar_enemies.py**  
> Comparing multiple algorithms over
> multiple runs for all enemies.
> Draws boxplot comparison plot.

> **find_best.py**  
> Attempts to find best set of weights maximising gain, fitness and the 
> number of defeated enemies respectively. Generates and writes three files with best found weights and their values.

> **line_plot.py**  
> Plots the mean and max generational fitness for two EAs

> **box_plot.py**  
> Plots box plot summary of the Gain values for two EAs
