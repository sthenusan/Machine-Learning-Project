# Pump It Up: Data Mining the Water Table

## **CS 4622 -Machine Learning Individual Project**

## **Introduction of the problem**

   *Currently, the Tanzanian population has very poor access to safe drinking water. 
  Approximately 47% of its citizens do not have access to it. More than $ 1.4 million in foreign aid
  has been donated to the country in an attempt to help solve the freshwater crisis. On the other hand,
  the government of Tanzania is not putting a solution to this problem. A good proportion of the water pumps
  are either not working or barely working and are in need of repair. Many people must drink dirty water or
  walk several miles just to get to the nearest working groundwater pump.* 

   ![pumping](https://user-images.githubusercontent.com/46936272/132973229-8b5c1601-3080-48b7-a71d-b9e614b4df1e.jpg) 

## **Project Data Source** 

  *DrivenData, Train set and train label set will be used in this project. There are 59400 water locations in the train set, with 40 features. The train labels data contains 59400    identical water points with the same train set, but only contains information about the id and status of these points.*

### **Main Objective of the project**

  *Using the data uploaded to the Taarifa website and the Tanzanian Ministry of Water, The goal is to predict which pumps are working,
  which are not, and which are in need of repair. Understanding which ones will fail is important for the following reasons:*

    Predicting the functionality of all groundwater pumps found in the territory with accurate models could help save 
    the Tanzanian government a lot of time and money. These models can help reduce the cost of inspecting each water pump.
    The government can use this study to find out exactly what the pumping situation of its water is.
    This work will be divided into three parts: 
      1. Exploratory Analysis of the Data
      2. Pre-processing of the same Data
      3. Selection of the Model and traing the model for prediction.
      4. Predict labels for test data given

* The details and coding of the initial data exploration of the given test dataset is given as a python notebook  & can be found [here](https://github.com/sthenusan/ml-project-assignment/blob/main/Notebooks/Initial%20Data%20Exploring.ipynb).
* The details of the preprocessing and modeling python notebook can be found [here](https://github.com/sthenusan/ml-project-assignment/blob/main/Notebooks/pump-it-up%20170628x.ipynb).

          
# Initial Data Exploration and Analysis

1. First, we'll look over each column one by one to see what they're about and what they're meaning about.

        Features
                
        amount_tsh - Amount of water to pump
        date_recorded - Date of data insertion
        funder - Who founded the well
        gps_height - Altitude of the well
        installer - Organization that installed it
        longitude - GPS coordinates
        latitude - GPS coordinates
        wpt_name - Pump name (if it has one)
        num_private - Number
        basin - Geographic basin
        subvillage - Geographical location
        region - Geographic location
        region_code - Geographic location (in code)
        district_code - Geographic location (in code)
        lga - Geographical location
        ward - Geographical location
        population - Population by the well
        public_meeting - True / False
        recorded_by - Group entering this data
        scheme_management - Who manages the pump
        scheme_name - Who runs the pump
        permit - Whether the pump is allowed or not
        construction_year - Year of construction of the pump
        extraction_type - Pump extraction type
        extraction_type_group - Pump extraction type
        extraction_type_class - Pump extraction type
        management - How the pump is managed
        management_group - How the pump is managed
        payment - Cost of water
        payment_type - Cost of water
        water_quality - Water quality
        quality_group - Water quality
        quantity - Amount of water
        quantity_group - Amount of water
        source - Water source
        source_type - Water source
        source_class - Water source
        waterpoint_type - Pump type
        waterpoint_type_group - Pump type
        
        Labels
        
        functional - The pump works and does not need to be repaired
        functional needs repair - Works, but needs repair
        non functional - The water pump does not work

* I looked at the data distribution and saw that there are numerous missing values for the variables' min values. This refers to the presence of missing values that must be addressed before the models can be continued. I removed a few of them because repeated and similar values have no effect on the goal, and reducing the data makes it easier to run our models. The mean values can be used to fill in some of the missing values.

   ![image](https://user-images.githubusercontent.com/46936272/132957569-f79d9b16-3acc-4df9-a6b0-92a45f0369ec.png)

  
* The training set contains 59400 observation records  and 41 columns.
The label for each pump is displayed in the column **status group**, while the other 40 variables correspond to the attributes, 10 of which are numerical and the rest are categorical.
'id', 'amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'region_code', 'district_code', 'population', 'construction_year' are numeric values others are categorical values.

* The train data set is labeled as functional - 32259, non functional - 22824 and functional needs repair - 4317. So Given data has highly imbalanced target values.
* It means that we can start to estimate the 54.31% probability that any one pump in this database will work fine (that is, it is *functional*).This will be used to make future predictions.

* Due to the discrete nature of the target variable, a supervised classification technique must be used, which can be performed after data preprocessing and feature engineering.

* Some columns have data that is comparable and similar meaning. As a result, I must choose the best candidate for model training. I'll go over each column individually.

### scheme_management / management / management_group columns

* These scheme management and management columns contain nearly identical information. Because "scheme management" refers to who manages the water station, "management" refers to how it is run. I'd rather keep the 'management' column because the 'scheme management' field has 3877 null entries. In the column 'management group,' similar information on how the water point is managed is maintained..

   ![image](https://user-images.githubusercontent.com/46936272/132957580-5390c41d-3eba-4f6d-a6ee-e7c3c2ce26a5.png)
    
 * I checked the 'user-group' values to locate the subgroups of the 'management_ group' column and discovered that this column is simply the grouped version of 'management'. Despite the fact that the 'management' column includes more detailed information, I choose to remove the 'management group' column. I also categorized it below and saw the numbers of sub-groups to recall and check the sub-groups of the 'management_group' column (management column)
    
   ![image](https://user-images.githubusercontent.com/46936272/132974375-61f64032-3979-4991-90c9-0883fa138711.png)

       
### quantity / quantity_group columns

* Because these two columns contain the identical data, I opted to remove the 'quantity_group' column..
    
   ![image](https://user-images.githubusercontent.com/46936272/132957603-c629a6a8-ba6a-480b-a53f-48c6fe1eef00.png)

* It is clear that, despite the presence of sufficient water, some wells are non-functional. Dry quantity water points have a strong correlation with non-functionality in this graph. There's a good likelihood the water point is non-functional if it's dry or unknown. If the quantity is enough, however, there is a better probability of finding functional water points.
    
    
### source / source_type / source_class columns

* It is self-evident that the information in these three columns is identical. As a result, I've opted to maintain only the 'source' column because it has more detailed information, and I'll omit the others.
    
   ![image](https://user-images.githubusercontent.com/46936272/132957622-3c43a6a2-a097-42ad-8444-7924acf8ca59.png)
    
   ![image](https://user-images.githubusercontent.com/46936272/132957654-f51214ee-b145-4abd-8753-69ef5ecfdcc3.png)


* There is a lot of non-functional ground water when I look at the columns. It's also interesting to note that the functional and non-functional waterpoints for machine dbh and swallow well sources are nearly identical..
    
    
### water_quality / quality_group columns
        
* I'll maintain 'water_quality' and drop 'quality_group' because the 'water_quality' column has more unique values.
   
   ![image](https://user-images.githubusercontent.com/46936272/132957720-6def4cac-0799-4884-adc0-a57b55c15180.png)
    
* It can be seen from the graph that many non-functional water sites have soft, good water quality.
     
### payment / payment_type columns
    
* Because these two columns are identical, I chose to remove one of them.
    
   ![image](https://user-images.githubusercontent.com/46936272/132957676-81aec2b7-95db-4249-b0c8-e3937aa25943.png)

* This feature shows us what the water cost. Mostly, there are lots of non-functioal water points as never paid for them. Repair needed pumps are commonly in never paid category.
    
    
### extraction_type / extraction_type_group / extraction_type_class columns

* It is clear that the information in these three columns is identical. As a result, I choose to keep 'extraction_type_group' and remove the others. Although extraction type has more unique values than extraction type group, according to this large dataset, some of these values are quite small. I prefered to use more compact one. Furthermore, extraction_type_class column contains less information. As a result, extraction_type_group has been chosen to be kept.
    
   ![image](https://user-images.githubusercontent.com/46936272/132957754-552b4bc8-e30a-41a7-a29f-f36d7dcf3c9d.png)
    
* Other and mono extraction kinds, in particular, have a larger chance of being non-functional than functional..
    
### waterpoint_type / waterpoint_type_group columns

* Despite the fact that both have the identical information, I choose to maintain 'waterpoint_type' because it contains more data.
    
   ![image](https://user-images.githubusercontent.com/46936272/132957872-fa755134-9001-405e-80b9-6f3509c1a06b.png)

* It can be seen that waterpoint type has correlation with functionality of water points. As a result, communal standpipes have a higher chance of being functional, while communal standpipe multiple and others have a higher chance of being non-functional.
    
    
### construction_year column
    
* The dataset now includes a new feature. For future encoding, the year values are changed to decades. The missing values are represented as zero. Because it contains the bulk of the data, it will not be converted to the mean or median and will be retained as a new value for decades.
   
   ![image](https://user-images.githubusercontent.com/46936272/132955254-60247de8-bb7a-42ab-acff-a009220dee01.png)
    
   ![image](https://user-images.githubusercontent.com/46936272/132955298-c4e20158-0535-4106-bfc1-1976c1002c2c.png)

* It is clear that missing values and that recent years have shown an increase in the number of functional water points.

### recorded_by column

   ![image](https://user-images.githubusercontent.com/46936272/132998302-5d689438-bdb8-467f-8c7e-c9e3746fd2c1.png)

* There is only one value in the 'recorded_ by' column. Our model will not receive any information as a result of this. As a result, I also dropped it.

### installer column

* There are lots of NaN and 0 values in this column. Firstly, I will convert them to unknown and change the spell mistakes in the data. 
    
* It is interesting that most of water points which central government and district council installed are non-functional. DWE has the majority of functional wells but has also many non-functional wells.
   
   ![image](https://user-images.githubusercontent.com/46936272/132955391-2e6ecef1-8023-4d85-b88b-68a36f55baab.png)

### funder column

* This is a highly categorized column with thousands of possible values. So, for future encoding, I'll use the 20 most common values.

   ![image](https://user-images.githubusercontent.com/46936272/132955434-9a481b6c-7679-43ce-b291-b585f3b55e28.png)

* From the plots, I realize that most of the water points which funded by government are non-functional.

### longitude,latitude column

   ![image](https://user-images.githubusercontent.com/46936272/132956803-77e2f3cb-6114-4b63-addc-e4779ffc8103.png)

* When the longitude is unknown, it is evident that it is written as 0. Because the zero points in the graph above outliers and outside of Tanzania are plainly seen. As a result, I converted them to mean, where the median is nearly the same number.

### wpt_name / scheme_name / id/ region/ region_code columns

* When I checked the wpt_name, scheme_name and id columns, they do not have any information about functionality. So, I decide to drop them. I dropped also region_code column because region column gives more information about the region. Also, before dropping columns i check the dublicated values in dataframe.

   ![image](https://user-images.githubusercontent.com/46936272/132956871-58c2fb27-3e13-4192-bad6-1e9ffe49b6ac.png)

   ![image](https://user-images.githubusercontent.com/46936272/132956887-ef366253-bf18-4636-8d74-48be088f5fe5.png)

* Some areas have a better chance of having a functional water well. Between the basins of Klimanjaro and Arusha is the Pangani basin, which has a higher water point. They also have larger parts for functional wells, as can be seen.

### amount_tsh column

   ![image](https://user-images.githubusercontent.com/46936272/132998248-65e08ed1-1ccc-42f7-8008-68c6283e532e.png)

I decided to drop this column because 70% of the column has no informative values. So, this column will not give idea to the model and i will drop it.

### gps_height column

   ![image](https://user-images.githubusercontent.com/46936272/132956978-4752da50-4a80-429e-98e9-1baec4b2d6f6.png)

* The level of the water point from sea level is shown by the GPS height. I don't modify this column because there are 34% zero values, but maybe 34% of the water spots are at sea level.

### population column

   ![image](https://user-images.githubusercontent.com/46936272/132977514-fab8e831-e843-4554-adc4-abd4f628b0de.png)

* Some functional water points has zero population, it is weird so I will change zero population to mean.

* To see the most populated areas water point functionality , i choose crowded 50 values and did groupby. It shows that higher population areas have more functional water points.

### date_recorded column
* Between 2011 and 2013, approximately 95% of the water points were reported. As a result, I do not believe it contains necessary functionality information at this time. For the time being, I'm going to drop this column.

### num_private column

* This column has no information about it and also mostly have zero values. So, i drop this also.

### basin column

   ![image](https://user-images.githubusercontent.com/46936272/132957038-15fd5ea9-9882-43de-bd6a-673a7039e738.png)

* This column gives an idea about there is correlation between functionality and geographical water basin.

### subvillage column

* This column has location value of water point regions but i already have region column. I will drop this, because it is hard to handle this nunique object values.

### district_code column

   ![image](https://user-images.githubusercontent.com/46936272/132997995-6abc2227-45ba-4bf6-ad62-0ed516513d2f.png)

* It includes numeric values about districts. Each district has one number.

   ![image](https://user-images.githubusercontent.com/46936272/132998021-8c0271ad-7c25-45f1-9e30-32f68ee1f5b1.png)

* It includes numeric values about districts. Each district has one number.


### lga / ward columns

* Now I decided to keep these columns because they contain geographical location. But, I have also other location features so maybe they will be dropped later on.

### public_meeting column

* There are some null values and I convert them to most common data.

### permit column

* This column indicates whether or not the water point is permitted. This column contains 3056 null values. I'll switch them to true, which has a higher value.

* I chose to alter the target value to a numerical number before feeding the data into the model after this Columnwise analysis.

   ![image](https://user-images.githubusercontent.com/46936272/132957171-57a76a9d-ae04-420a-a844-8775661aef2f.png)

## Data Cleaning Process
* The data has lots of null values, missing values and unnecessary dublicated features. Two main challanges are in this project is cleaning data and handling highly imbalanced target labels.
* I tried to solve cleaning challange in this notebook. Some columns which have same information were dropped. Null, wrong and missing values changed to mean, median or unknown. Some values in features collected together and categorized.
* Detailed data cleaning processes can be found in this notebook under the headings of relevant columns. It is stated that how the column was cleaned with reasons.

## Explorations and Analysis Outcomes
* Generally higher population areas has higher number of functional wells.
* Some areas has higher probability to find clean water especially, if they are near to good basins.
* Darul es Salaam is one of the highest populated cities but 35% of good water quality points are non-functional.
* Iringa is one of the important areas but it contains lots of non-functional wateer points which has soft water.
* Mostly the wells which are funded by government are non-functional.
* Most of water points which central government and district council installed are non-functional.
* The most common extraction type is gravity but second is hand pumps. The efficiency of handpumps are less than commercial pumps. It shows that authorities need to focus on pumping type. It is seen that, there are many non-functional water points which belongs to gravity (which is natural force so no need to do anything expensive) as extraction type.
* Some water points which has enough and soft water are non-function.
* The wells which have constructed in recent years are functional then olders. And it is seen that recent years have some functional but needs repair wells. It means that if they will not be repaired recently, they will be non-functional easily.
* There are lots of water wells which has enough water are non-functional.

## Findings and Results Gathered
* 4272 wells were dried but they have good water quality. With finding a solution to give source again these wells, they can be functional. Finding clean water sources is not the only problem, to continue to feed these sources are equaly important.
* 2226 (7%) wells have enough and soft water but needs repair. Authorities must invest on repairing. Otherwise these will be non-functional.
* 8035 (27%) wells has enough, good quality water but they are non-functional. This shows that authorities must work and invest on technology to pump these good sources.
* Authorties should check again the wells which they funded.
* New tecqniques must be found to feed dry wells and repair wells.


## Feature Engineering Proposals
* There are lots of categorical values in funder and installer columns. I create new columns that if the value in the feature is not in first common 20 values, they were collected as 'others'. Also, there are lots of spelling mistakes in this columns which creates new unique values in these columns. I found top 100 common installer and fixed them. Then, builded new column which has categorized values.
* Construction years are in integer format but not continuous data or year values do not make sense for model. So, I divided them decades and assumed every decade as categorical value.

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
This is done independently column by column for multivariate data.

![image](https://user-images.githubusercontent.com/46936272/133087855-d0094a72-a388-4c3e-85cd-d458d403982b.png)

### Principal Component Analysis

* Even if it did not increase the final score, it is crucial to remark because it was not included in the model.

![image](https://user-images.githubusercontent.com/46936272/133088235-0ee4c048-219a-4519-a20e-59137daddfcd.png)

## Model selection

* I tried mulitiple models and check with the validation set and comapare it.

1. Decision trees - 74.61
2. RandomForestClassifier - 82.05
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

**-*Gradient Boosting Classifier* - 79.19**

**-*Random Forest*- 82.93**

**-*Light GBM*- 78.47**

The **RandomForestClassifier** model gives best accuracy score among these models.

### Tuning of models and tuning of parameters

* I decided to adjust the parameters based on the tuning of the three models just mentioned.
* As shown above, Random Forest is the best model. So i use GridSearchCV to get best parameters for the Random Forest Model.

![image](https://user-images.githubusercontent.com/46936272/133236070-253439fe-57cc-4569-831c-9904701a1cce.png)

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

* Before make submisiion to [Data Driven](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/), I need to convert back out class labels from 0,1,2 to non functional, functional needs repair and functional.

![image](https://user-images.githubusercontent.com/46936272/133129258-d5e847c6-cfc1-4ded-9623-7b0117eba154.png)

* Finally, using the format specified in *Submission format.csv*, I construct the file that need to submit to the *Diven Data* Competition.
* Best & Final submission CSV file is [here](https://github.com/sthenusan/ml-project-assignment/blob/main/submissions/submission_best_RFC_0.8215.csv).

* Evidence for the Final Rank of the competition in DataDriven is given below.

![Screenshot 2021-09-14 140058](https://user-images.githubusercontent.com/46936272/133364562-3237f1c5-0694-4f1f-8479-ecd9f84bda1f.png)

