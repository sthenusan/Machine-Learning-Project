## Pump It Up: Data Mining the Water Table

**CS 4622 -Machine Learning Individual Project**

**Project Data Source: DrivenData, Train set and train label set will be used in this project. There are 59400 water locations in the train set, with 40 features. The train labels data contains 59400 identical water points with the same train set, but only contains information about the id and status of these points.**

**Goal: To predict the operating condition of waterpoints in Tanzania i.e. to determine whether the water pump is functional, non-functional or needs repair
using machine learning technics.** 

**Introduction of the problem**

Currently, the Tanzanian population has very poor access to safe drinking water. 
Approximately 47% of its citizens do not have access to it. More than $ 1.4 million in foreign aid
has been donated to the country in an attempt to help solve the freshwater crisis. On the other hand,
the government of Tanzania is not putting a solution to this problem. A good proportion of the water pumps
are either not working or barely working and are in need of repair. Many people must drink dirty water or
walk several miles just to get to the nearest working groundwater pump.

**Main Objective of the project**

Using the data uploaded to the Taarifa website and the Tanzanian Ministry of Water, The goal is to predict which pumps are working,
which are not, and which are in need of repair. Understanding which ones will fail is important for the following reasons:

    Predicting the functionality of all groundwater pumps found in the territory with accurate models could help save 
    the Tanzanian government a lot of time and money. These models can help reduce the cost of inspecting each water pump.
    The government can use this study to find out exactly what the pumping situation of its water is.
    This work will be divided into three parts: 
      1. Exploratory Analysis of the Data
      2. Pre-processing of the same Data
      3. Selection of the Model and traing the model for prediction.
      
      
## Data Exploration and Analysis

1. First we will go through each columns one by one to explore and understand what is the contend of each columns.

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

* I checked the distribution of the data and note that there are several missing values for the min  of the variables. This refers to the existence of  missing values that have to deal with before continuing with the models.  I removed some of them because the same values and dublicated values have no effect on the goal, and simplifying the data makes it easier to run our models. Some of the missing values are filled using mean values.
  
* The training set contains 59400 observation records  and 41 columns.
The column **status_group**  shows the label for each pump, the other 40 variables correspond to the characteristics, 10 of which are numeric and the rest are categorical.
* 'id', 'amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'region_code', 'district_code', 'population', 'construction_year' are numeric values others are categorical values.

* The train data set is labeled as functional - 32259, non functional - 22824 and functional needs repair - 4317. So Given data has highly imbalanced target values.
* It means that we can start to estimate the 54.31% probability that any one pump in this database will work fine (that is, it is *functional*). This serves as the basis for future predictions.

* Since the target variable is discrete, Need to use a supervised classification algorithm, which can be applied later after data preprocessing and feature engineering.

* Some columns have similar data. So I need to select the best one for the model traning. I will analysis each column one by one.

* scheme_management / management / management_group columns

    The information in these scheme_management and management columns is practically identical. Because "scheme_management" denotes who runs the water station,'management' denotes how the station is run. The'scheme_management' field has 3877 null entries, thus I'd rather preserve the'management' column. Similar information on how the water point is maintained is kept in the column 'management_group'.
    
    
    
    
    To find the subgroups of 'management_group' column, I checked the 'user-group' values and saw that this column is just the grouped version of 'management'. Although 'management' has more detailed values, I decided to drop 'management_group' column. Also, to remember and check the sub-groups of the 'management_group' column, I grouped it below and saw the numbers of sub-groups (management column).
    
    
    
       
* quantity / quantity_group columns
    These two columns contain same information so I decided to drop 'quantity_group' column.
    
    
    

    It can be seen obviously that although there are enough water quantity in some wells, they are non-functional. When looking at this graph, dry quantity water points have a highly correlation with non-functionality. If the water point is dry or unknown, there is high chance thw water point is non functional. On the other hand, if the quantity is enough, there is a higher chance to find functional water points.
    
* source / source_type / source_class columns

    It is obvious that these three columns keep same information. so, i decided to keep just 'source' column, because it has more detailed information and I will drop others.
    
    
    
    When i look at the columns, there are lots of non-functional ground water. And, it is interesting that machine dbh and swallow well sources nearly have same functional and non-functional waterpoints.
    
* water_quality / quality_group columns
    
    
    'water_quality' column has more unique values, so I will keep 'water_quality' and drop 'quality_group'.
    
     From the both graphs, it is seen that lots of non-functional water points have soft, good water quality.
     
* payment / payment_type columns
    
    These two columns are same so i decided to drop one of them.
    
    This feature shows us what the water cost. Mostly, there are lots of non-functioal water points as never paid for them. 
    
    
* extraction_type / extraction_type_group / extraction_type_class columns


    It is obviously seen that these three columns keep same information. So, I decided to keep 'extraction_type_group' and drop others. Although, extraction_type has more unique values than extraction_type_group , some of these values are very small amount according to this big dataset. I prefered to use more compact one. Also, extraction_type_class contains less detail. So, extraction_type_group is chosen to keep.
    
    
    
    Especially, other and mono extraction types have higher change to be non-functional than functional.
    
* waterpoint_type / waterpoint_type_group columns
    Eventhough both have same information, I decided to keep 'waterpoint_type' which contains more detail. 
    
    It can be seen that waterpoint type has correlation with funtionality of water points. Such that, communal standpipe has higher possibility to have functional, although communal standpipe multiple and others have higher possibility for non-functionality.
    
* construction_year column
    
    New feature is added to the dataset. The year values are converted to decades for future encoding. Zero shows the missing values. This have majority of the data set so, it will not be changed to the mean or median, kept as new value in decades.
