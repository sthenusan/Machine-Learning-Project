# Data Preprocessing and Model Building of the Project

* Even though some of the works to be mentioned here already have included in the previous [Initial Data Exploration file](https://github.com/sthenusan/ml-project-assignment/blob/main/Initial%20Exploration.md) but, I have mentioned them here also to get better understanding for the readers.

#### * Before start the the process we need to import nesssary tools such as libraries and data for the project.

 ![image](https://user-images.githubusercontent.com/46936272/133043808-51f868d3-7f32-4482-9a04-53cc1fefebc4.png)

### Descriptive statistics

![image](https://user-images.githubusercontent.com/46936272/133043910-8e5e4d82-bfc1-4edf-b883-35d6766f2cd1.png)

* I may examine the data distribution using these tables. I notice that the *min* of the feature variables has multiple missing values. This refers to the existence of missing values that must be addressed before moving forward with the models.

* There are 59400 records in the training set, with 41 columns/features. The column **status group** displays the label for each pump, while the remaining 40 variables correspond to the attributes, with 10 being numeric and the rest being category. I'll start by looking into the numerical ones.

* Next I will observe the distribution of the target variable *label* in *train* that will serve us for the calculations of our predictions.

  ![image](https://user-images.githubusercontent.com/46936272/133044677-7bd69896-526f-46f3-883d-01b0e42865ca.png)

* Column wise explanation and analysis can be found [here](https://github.com/sthenusan/ml-project-assignment/blob/main/Initial%20Exploration.md).

### Correlation between variables
* This section displays the correlation between variables. It should be noted that the presence of * outliers * can have an impact on this.
* In order to improve the model using linear regression, it is required to exclude variables that are highly associated.
* The following is a graph of the correlation:

  ![image](https://user-images.githubusercontent.com/46936272/133046714-6cb9c881-6996-4045-aa76-616b7c2937ea.png)
  
* The connection between *district_code* and *region_code* is pretty high, as we can see. As a result, one of them may need to be eliminated.

* The correlation between *construction year* and *gps_height* is also high, but these two variables do not have such an obvious relationship, so we will investigate it further before making any decisions.

* The most associated variables in connection to *label* are:

![image](https://user-images.githubusercontent.com/46936272/133047122-b4c3ff2f-58ba-4781-b81c-c561426cebdb.png)

* The *region_code* has a higher negative linear correlation with the target variable than *district_code*. I keep the one that has it higher.

* All other variables have a low linear correlation with the target variable, but this could indicate a non-linear correlation.




