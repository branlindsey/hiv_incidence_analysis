# HIV Incidence Prediction Model

## Background
Due to the development of anti-retroviral therapies the HIV/AIDS epidemic is 
generally considered to be under control in the US.  However, as of 2015 there 
were 971,524 people living with diagnosed HIV in the US with an estimation of 
37,600 new HIV diagnoses in 2014.  HIV infection rates continue to be particularly
problematic in communities of color, among men who have sex with men (MSM), the
transgender community, and other vulnerable populations in the US. Socioeconomic 
factors are a significant risk factor for HIV infection and likely contribute 
to HIV infection risk in these communities.  The current US opioid crisis has 
further complicated the efforts to combat HIV with HIV infection outbreaks now 
hitting regions that weren’t previously thought to be vulnerable to such outbreaks.  

A model that can accurately forecast regional HIV infection rates would be 
beneficial to local public health officials.  Provided with this information, 
these officials will be able to better marshal the resources necessary to combat
HIV and prevent outbreaks from occurring.  Accurate modeling will also identify 
risk factors for communities with high HIV infection rates and provide clues 
as to how officials may better combat HIV in their respective communities. 

---

## Data Sources

The `./data` folder contains data from three publically available sources.  Groups should feel
free to supplement this data if they wish.
1. The largest collection of HIV and opioid data was obtained from the [opioid database](http://opioid.amfar.org/) maintained by the American Foundation for AIDS Research (amfAR).  
2. Demographic and economic data were obtained from the 5yr - American Community Survey which are available at the [US census bureau website](https://factfinder.census.gov/faces/nav/jsf/pages/searchresults.xhtml?refresh=t).
3. Estimates for the [MSM population](http://emorycamp.org/item.php?i=48) in each county were obtained from the Emory Coalition for Applied Modeling for Prevention (CAMP).

Data dictionaries that indicate what each column in the data means are included in the folder associated with each data set.

---

## Workflow

1. The plan was to build and test a series of regression models on various subsets of the data to best be able to predict HIV incidences.

2. The data was split into training data and holdout data saved for the final testing.

3. The team split into 3 functions. One person performed EDA to determine which subsets of the data to use. One person wrote functions in python for rapid, repeated regression analysis with different models. One person wrote a python notebook with the workflow for the regression analysis.


---

## Regression Models

In order lower the mean squared error, we ran 3 models.

### Model 1: Training Data Mean 

Our first model used the mean of training data which was 17.84 as the predictor.  Our MSE on this model is 135.47.  We used this method because it provided a good baseline value to judge our models on. 

### Model 2: Linear Regression

We used all of the features and performed a linear regression model.  Our MSE was 302.72 on this model. 

### Model 3: Linear Regression with Specific Features 

After additional EDA we chose features that appeared to have a strong relationship. 

We chose these features: 
- 'HIVprevalence'
- 'MSM12MTH'
- 'unemployment_rate'
- 'poverty_rate'
- 'household_income'
- 'percent_uninsured'
- 'ADULTMEN'

Our MSE was 99.85 on this model.  This model was an improvement over both of our models.

---

## Credit
This case study is based on [Eric Logue's project](https://github.com/elogue01/Forecasting-HIV-Infections).