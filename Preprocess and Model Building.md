# Data Preprocessing and Model Building of the Project

* Even though some of the works to be mentioned here already have included in the previous [Initial Data Exploration file](https://github.com/sthenusan/ml-project-assignment/blob/main/Initial%20Exploration.md) but, I have mentioned them here also to get better understanding for the readers.

* Before start the the process we need to import nesssary tools such as libraries and data for the project.

 ![image](https://user-images.githubusercontent.com/46936272/133043808-51f868d3-7f32-4482-9a04-53cc1fefebc4.png)

### Descriptive statistics


![image](https://user-images.githubusercontent.com/46936272/133043910-8e5e4d82-bfc1-4edf-b883-35d6766f2cd1.png)

* I examine the data distribution using these tables. I notice that the min of the feature variables has multiple missing values. This refers to the existence of missing values that must be addressed before moving forward with the models.

* There are 59400 records in the training set, with 41 columns/features. The column status group displays the label for each pump, while the remaining 40 variables correspond to the attributes, with 10 being numeric and the rest being category. I'll start by looking into the numerical ones.

* Next I will observe the distribution of the target variable label in train that will serve us for the calculations of our predictions.

  ![image](https://user-images.githubusercontent.com/46936272/133044677-7bd69896-526f-46f3-883d-01b0e42865ca.png)
  
* I can see from above graph that, I can start to estimate the 54.31% probability that any one pump in this database will work fine (that is, it is functional). This will be used to make future predictions.

Since our target variable is discrete, I will need a **supervised classification algorithm**, which I will apply later.

* Column wise explanation and analysis can be found [here](https://github.com/sthenusan/ml-project-assignment/blob/main/Initial%20Exploration.md).

### Correlation between variables
* This section displays the correlation between variables. It should be noted that the presence of outliers can have an impact on this.
* In order to improve the model using linear regression, it is required to exclude variables that are highly associated.
* The following is a graph of the correlation:

  ![image](https://user-images.githubusercontent.com/46936272/133046714-6cb9c881-6996-4045-aa76-616b7c2937ea.png)
  
* The connection between district_code and region_code is pretty high, as we can see. As a result, one of them may need to be eliminated.

* The correlation between construction_year and gps_height is also high, but these two variables do not have such an obvious relationship, so we will investigate it further before making any decisions.

* The most associated variables in connection to label are:

![image](https://user-images.githubusercontent.com/46936272/133047122-b4c3ff2f-58ba-4781-b81c-c561426cebdb.png)

* The region_code has a higher negative linear correlation with the target variable than district_code. I keep the one that has it higher.

* All other variables have a low linear correlation with the target variable, but this could indicate a non-linear correlation.

### Data preprocessing

I'll deal with data preprocessing in this section, which involves preparing and optimizing data in order to run future tests with Machine Learning models.

### Extraction of similar variables

As I mentioned [here](https://github.com/sthenusan/ml-project-assignment/blob/main/Initial%20Exploration.md) I need to handle similar variables in the dataset to get better model.

### The following attributes

    - (extraction_type, extraction_type_group, extraction_type_class),
    - (water_quality, quality_group),
    - (waterpoint_type, waterpoint_type_group)
    - (scheme_name, scheme_management)
    - (source, source_class),
    - (subvillage, region, region_code, district_code, lga, ward),
    - (payment, payment_type),

* they provide very similar information, which indicates that there is a high correlation between them. By leaving them, would risk an overfitting.

* What's more:

   - num_private consists of 99% zeros and does not have a clear description, so we cannot interpret it
   - wpt_name is not very informative as it has fewer values than the number of observations

* Because there is a strong association between the district_code and region_code, I will eliminate a variable between them. The one having the highest correlation with the target variable will be chosen. The region_code has a stronger negative connection with the goal than district_code.

### Ordinal encoding of categorical data

* This methodology was chosen to prevent having too many columns and to provide the model some logic when analyzing the features. 
* For example, the higher the category in the variable quality_group, the better the water quality and the more probable the pump will perform correctly.

#### quality_group

![image](https://user-images.githubusercontent.com/46936272/133055948-fe9c59a4-5b0a-45ec-be49-b07eddb2504f.png)

#### quantity_group

![image](https://user-images.githubusercontent.com/46936272/133056371-e0996825-99d8-42a5-8323-f148480f6ccd.png)

#### payment_type

![image](https://user-images.githubusercontent.com/46936272/133056464-4a41c3c1-af92-4daf-8df7-b79b778debbc.png)

#### public_meeting

![image](https://user-images.githubusercontent.com/46936272/133056562-2707d48f-db9f-43ef-973f-b36dec016364.png)

#### permit

![image](https://user-images.githubusercontent.com/46936272/133056640-8b7067ca-64ca-4143-b49c-117a04b5d7b7.png)

### More improvements to the model

* Following that, I'll create new variables (depending on the properties of the dataset) to better characterize the target.
* amount_tsh - I developed a criteria to differentiate the pumps that work from those that don't after the exploratory analysis of the data. I'll go ahead and build a new binary variable to reflect this information.
* construction_year - I then transform construction_year into a categorical variable containing the following decades of years: '60s',' 70s', '80s',' 90s, '00s',' 10s' and 'unknown' for unknown years.

### One-Hot encoding for categorical variables

* For categorical variables where there is no order relationship, integer encoding is generally not appropriate. In these cases, special encoding can be applied where a new binary variable (with true or false values) is added for each possible category value.
* The One-Hot encoding is a method of labeling which class the data belongs to and the idea is to assign 0 to the entire dimension except 1 for the class the data belongs to.

### Regularization with Logistic Regression
* In L1 Regularization, the complexity C is measured as the mean of the absolute value of the model coefficients. With it, we favor that some of the coefficients end up being 0. This can be useful to discover which of the input attributes are relevant and, in general, to obtain a model that generalizes better. L1 helps us make the selection of input attributes.

* After these processes, I export training and testing data set.

![image](https://user-images.githubusercontent.com/46936272/133086765-10fc94d3-ca9d-4c57-81a2-d2e38ab3c26b.png)


* **I am moving into the model selection and development part of the project.**

**Training Data set is divided into parts for training and testing.**

![image](https://user-images.githubusercontent.com/46936272/133087384-10f45308-e5f5-4e4a-b295-2f27c51cbc9a.png)

### Standard Scaling

* It's a technique for transforming your data into a distribution with a mean of 0 and a standard deviation of 1.
This is done independently column by column for multivariate data. Sources of the information [this](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

![image](https://user-images.githubusercontent.com/46936272/133087855-d0094a72-a388-4c3e-85cd-d458d403982b.png)

### Principal Component Analysis

* Even if it did not increase the final score, it is crucial to remark because it was not included in the model.

![image](https://user-images.githubusercontent.com/46936272/133088235-0ee4c048-219a-4519-a20e-59137daddfcd.png)

## Model selection

* I tried mulitiple models and check with the validation set and comapare it.

1. Decision trees - 74.61
2. RandomForestClassifier - 80.05
3. GradientBoostingClassifier - 79.87
4. LGBMClassifier - 79.01 
5. BaggingClassifier - 76.27
6. XGBClassifier - 74.12
7. ExtraTreesClassifier - 78.13
8. LogisticRegression - 70.64
9. KNeighborsClassifier - 75.36
10. GaussianNB - 75.36

  ![image](https://user-images.githubusercontent.com/46936272/133127978-829a76f3-e0b6-4e53-8546-0a9df1ee61ed.png)


As we can see, the three best models are:

   * **Gradient Boosting Classifier - 79.87**

   * **Random Forest- 80.05**

   * **Light GBM - 79.01**

The **RandomForestClassifier** model gives best accuracy score among these models.

### Tuning of models and tuning of parameters

* I decided to adjust the parameters based on the tuning of the three models just mentioned.


# Conclusions

* The purpose of this study was to anticipate whether a pump will function or need to be repaired based on information about the pump, the well, its surroundings, who was in charge of it, and the date.

* I began by conducting an exploratory data analysis. Calculating precision and separating data into numerical and categorical categories based on their typology. I then looked for outliers, examined and acted on the different correlations between some of the variables, and detected missing data to deal with them later in the preprocessing phase.

* I then cleaned and preprocessed the data in the following step. To reduce multicollinearity, I started by eliminating properties that held comparable information. The missing data was then processed, with ordinal type coding for those who needed it and One-Hot type for the rest. Finally, I've added new variables to the target that better characterize it.

* After the preprocessing, I have selected with Logistic Regression the 80 most important variables from a total of 90 columns. Finally, various models have been checked and the results shown in a graph, obtaining that the best ones are:
 
    * **Gradient Boosting Classifier**

    * **Random Forest**

    * **Light GBM**

* I have selected Random Forest as the best solution. After that, I created validation results and submit it to the data driven competion.

# Outcome

* Before make submisiion to [Data Driven](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/), We need to convert back out class labels from 0,1,2 to non functional, functional needs repair and functional.

![image](https://user-images.githubusercontent.com/46936272/133129258-d5e847c6-cfc1-4ded-9623-7b0117eba154.png)

* Finally, using the format specified in *Submission format.csv*, I construct the file that need to submit to the *riven Data* Competition.
* Best submision file is [here](https://github.com/sthenusan/ml-project-assignment/blob/main/submission_final_RFC.csv)

