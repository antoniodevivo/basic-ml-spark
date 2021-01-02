import pyspark 
from pyspark.sql import DataFrameNaFunctions 
from pyspark.sql.functions import lit 
from pyspark.ml.feature import StringIndexer  
from pyspark.ml import Pipeline 
from pyspark.sql import SparkSession
from pyspark.sql import functions
import pandas as pd
import numpy as np

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-xml_2.11:0.4.1 pyspark-shell'

# initializing the session
spark = SparkSession.builder.appName('linear_regression').getOrCreate()

# Download the dataset here -> https://drive.google.com/file/d/1uZ_CMhbf5ujxHRLZ-JEBXjbOOkGQscbr/view
# then put the file in the file section of colab
df = spark.read.csv("iris.csv", inferSchema=True, header=True)

df.head()
### Result: 
# Row(_c0=1, Sepal_Length=5.1, Sepal_Width=3.5, Petal_Length=1.4, Petal_Width=0.2, Species='setosa')

df.show()
### Result: 
""" 
+---+------------+-----------+------------+-----------+-------+
|_c0|Sepal_Length|Sepal_Width|Petal_Length|Petal_Width|Species|
+---+------------+-----------+------------+-----------+-------+
|  1|         5.1|        3.5|         1.4|        0.2| setosa|
|  2|         4.9|        3.0|         1.4|        0.2| setosa|
|  3|         4.7|        3.2|         1.3|        0.2| setosa|
|  4|         4.6|        3.1|         1.5|        0.2| setosa|
|  5|         5.0|        3.6|         1.4|        0.2| setosa|
|  6|         5.4|        3.9|         1.7|        0.4| setosa|
|  7|         4.6|        3.4|         1.4|        0.3| setosa|
|  8|         5.0|        3.4|         1.5|        0.2| setosa|
|  9|         4.4|        2.9|         1.4|        0.2| setosa|
| 10|         4.9|        3.1|         1.5|        0.1| setosa|
| 11|         5.4|        3.7|         1.5|        0.2| setosa|
| 12|         4.8|        3.4|         1.6|        0.2| setosa|
| 13|         4.8|        3.0|         1.4|        0.1| setosa|
| 14|         4.3|        3.0|         1.1|        0.1| setosa|
| 15|         5.8|        4.0|         1.2|        0.2| setosa|
| 16|         5.7|        4.4|         1.5|        0.4| setosa|
| 17|         5.4|        3.9|         1.3|        0.4| setosa|
| 18|         5.1|        3.5|         1.4|        0.3| setosa|
| 19|         5.7|        3.8|         1.7|        0.3| setosa|
| 20|         5.1|        3.8|         1.5|        0.3| setosa|
+---+------------+-----------+------------+-----------+-------+
only showing top 20 rows
 """

df.printSchema()
""" 
root
 |-- _c0: integer (nullable = true)
 |-- Sepal_Length: double (nullable = true)
 |-- Sepal_Width: double (nullable = true)
 |-- Petal_Length: double (nullable = true)
 |-- Petal_Width: double (nullable = true)
 |-- Species: string (nullable = true)
"""

# to use Spark with Python we must always remember that the data columns
# will be only two: the variable to predict, or label, and all the other variables
# aggregates and transformed into a single column

# import the modules to transform the data
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

df.columns
""" 
['_c0',
 'Sepal_Length',
 'Sepal_Width',
 'Petal_Length',
 'Petal_Width',
 'Species']
"""

assembler = VectorAssembler(
    inputCols=["Sepal_Length", "Sepal_Width", "Petal_Length"],
    outputCol="features")

transform = assembler.transform(df)

transform.select("features").show()
"""
+-------------+
|     features|
+-------------+
|[5.1,3.5,1.4]|
|[4.9,3.0,1.4]|
|[4.7,3.2,1.3]|
|[4.6,3.1,1.5]|
|[5.0,3.6,1.4]|
|[5.4,3.9,1.7]|
|[4.6,3.4,1.4]|
|[5.0,3.4,1.5]|
|[4.4,2.9,1.4]|
|[4.9,3.1,1.5]|
|[5.4,3.7,1.5]|
|[4.8,3.4,1.6]|
|[4.8,3.0,1.4]|
|[4.3,3.0,1.1]|
|[5.8,4.0,1.2]|
|[5.7,4.4,1.5]|
|[5.4,3.9,1.3]|
|[5.1,3.5,1.4]|
|[5.7,3.8,1.7]|
|[5.1,3.8,1.5]|
+-------------+
only showing top 20 rows
"""

transform.show()
"""
+---+------------+-----------+------------+-----------+-------+-------------+
|_c0|Sepal_Length|Sepal_Width|Petal_Length|Petal_Width|Species|     features|
+---+------------+-----------+------------+-----------+-------+-------------+
|  1|         5.1|        3.5|         1.4|        0.2| setosa|[5.1,3.5,1.4]|
|  2|         4.9|        3.0|         1.4|        0.2| setosa|[4.9,3.0,1.4]|
|  3|         4.7|        3.2|         1.3|        0.2| setosa|[4.7,3.2,1.3]|
|  4|         4.6|        3.1|         1.5|        0.2| setosa|[4.6,3.1,1.5]|
|  5|         5.0|        3.6|         1.4|        0.2| setosa|[5.0,3.6,1.4]|
|  6|         5.4|        3.9|         1.7|        0.4| setosa|[5.4,3.9,1.7]|
|  7|         4.6|        3.4|         1.4|        0.3| setosa|[4.6,3.4,1.4]|
|  8|         5.0|        3.4|         1.5|        0.2| setosa|[5.0,3.4,1.5]|
|  9|         4.4|        2.9|         1.4|        0.2| setosa|[4.4,2.9,1.4]|
| 10|         4.9|        3.1|         1.5|        0.1| setosa|[4.9,3.1,1.5]|
| 11|         5.4|        3.7|         1.5|        0.2| setosa|[5.4,3.7,1.5]|
| 12|         4.8|        3.4|         1.6|        0.2| setosa|[4.8,3.4,1.6]|
| 13|         4.8|        3.0|         1.4|        0.1| setosa|[4.8,3.0,1.4]|
| 14|         4.3|        3.0|         1.1|        0.1| setosa|[4.3,3.0,1.1]|
| 15|         5.8|        4.0|         1.2|        0.2| setosa|[5.8,4.0,1.2]|
| 16|         5.7|        4.4|         1.5|        0.4| setosa|[5.7,4.4,1.5]|
| 17|         5.4|        3.9|         1.3|        0.4| setosa|[5.4,3.9,1.3]|
| 18|         5.1|        3.5|         1.4|        0.3| setosa|[5.1,3.5,1.4]|
| 19|         5.7|        3.8|         1.7|        0.3| setosa|[5.7,3.8,1.7]|
| 20|         5.1|        3.8|         1.5|        0.3| setosa|[5.1,3.8,1.5]|
+---+------------+-----------+------------+-----------+-------+-------------+
only showing top 20 rows
"""

transformed_df = transform.select('features','Petal_Width')

transformed_df.show()
"""
+-------------+-----------+
|     features|Petal_Width|
+-------------+-----------+
|[5.1,3.5,1.4]|        0.2|
|[4.9,3.0,1.4]|        0.2|
|[4.7,3.2,1.3]|        0.2|
|[4.6,3.1,1.5]|        0.2|
|[5.0,3.6,1.4]|        0.2|
|[5.4,3.9,1.7]|        0.4|
|[4.6,3.4,1.4]|        0.3|
|[5.0,3.4,1.5]|        0.2|
|[4.4,2.9,1.4]|        0.2|
|[4.9,3.1,1.5]|        0.1|
|[5.4,3.7,1.5]|        0.2|
|[4.8,3.4,1.6]|        0.2|
|[4.8,3.0,1.4]|        0.1|
|[4.3,3.0,1.1]|        0.1|
|[5.8,4.0,1.2]|        0.2|
|[5.7,4.4,1.5]|        0.4|
|[5.4,3.9,1.3]|        0.4|
|[5.1,3.5,1.4]|        0.3|
|[5.7,3.8,1.7]|        0.3|
|[5.1,3.8,1.5]|        0.3|
+-------------+-----------+
only showing top 20 rows
"""

train, test = transformed_df.randomSplit([0.7,0.3])

train.show()
"""
+-------------+-----------+
|     features|Petal_Width|
+-------------+-----------+
|[4.3,3.0,1.1]|        0.1|
|[4.4,3.0,1.3]|        0.2|
|[4.4,3.2,1.3]|        0.2|
|[4.6,3.2,1.4]|        0.2|
|[4.6,3.6,1.0]|        0.2|
|[4.7,3.2,1.6]|        0.2|
|[4.8,3.0,1.4]|        0.1|
|[4.8,3.1,1.6]|        0.2|
|[4.8,3.4,1.6]|        0.2|
|[4.8,3.4,1.9]|        0.2|
|[4.9,2.5,4.5]|        1.7|
|[4.9,3.1,1.5]|        0.1|
|[4.9,3.6,1.4]|        0.1|
|[5.0,2.0,3.5]|        1.0|
|[5.0,2.3,3.3]|        1.0|
|[5.0,3.0,1.6]|        0.2|
|[5.0,3.2,1.2]|        0.2|
|[5.0,3.4,1.5]|        0.2|
|[5.0,3.5,1.3]|        0.3|
|[5.0,3.5,1.6]|        0.6|
+-------------+-----------+
only showing top 20 rows
"""

train.describe().show()
"""
+-------+------------------+
|summary|       Petal_Width|
+-------+------------------+
|  count|               108|
|   mean|1.1703703703703703|
| stddev|0.7605039228590301|
|    min|               0.1|
|    max|               2.5|
+-------+------------------+
"""

# let's create the regression model
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol='features', labelCol='Petal_Width', 
                      predictionCol='prediction')

# adapt it to the data
lr_model = lr.fit(train)

# print the coefficients
print("Coefficients: {} Intercept: {}".format(lr_model.coefficients, lr_model.intercept))
# Coefficients: [-0.2384270208878119,0.20670865063951435,0.5267224329790431] Intercept: -0.03679454895160388

# let's create predictions on test data
test_features = test.select('features')

predictions = lr_model.transform(test_features)

predictions.show()
"""
+-------------+-------------------+
|     features|         prediction|
+-------------+-------------------+
|[4.6,3.4,1.4]| 0.3066619733094706|
|[4.7,3.2,1.3]|0.18880529779488214|
|[4.7,3.2,1.6]| 0.3468220276885951|
|[4.8,3.4,1.9]| 0.5223377856214297|
|[4.9,2.5,4.5]| 1.6819356237025975|
|[4.9,3.0,1.4]|0.15245040678732108|
|[5.0,3.2,1.2]|0.06460494823063428|
|[5.0,3.5,1.3]| 0.1792897867203928|
|[5.0,3.6,1.4]|0.25263289508224845|
|[5.1,3.7,1.5]| 0.3021333013553232|
|[5.2,2.7,3.9]| 1.3357157877767312|
|[5.4,3.7,1.5]|0.23060519508897942|
|[5.4,3.9,1.3]|0.16660243862107377|
|[5.6,2.5,3.9]| 1.1990032492937033|
|[5.6,2.7,4.2]| 1.3983617093153191|
|[5.7,2.5,5.0]| 1.7545552234818695|
|[5.7,2.8,4.5]| 1.5532066021842021|
|[5.7,3.0,4.2]| 1.4365316024183923|
|[5.8,4.0,1.2]|0.03923025203199632|
|[6.0,3.4,4.5]| 1.6057036863015675|
+-------------+-------------------+
only showing top 20 rows
"""

test.show()
"""
+-------------+-----------+
|     features|Petal_Width|
+-------------+-----------+
|[4.6,3.4,1.4]|        0.3|
|[4.7,3.2,1.3]|        0.2|
|[4.7,3.2,1.6]|        0.2|
|[4.8,3.4,1.9]|        0.2|
|[4.9,2.5,4.5]|        1.7|
|[4.9,3.0,1.4]|        0.2|
|[5.0,3.2,1.2]|        0.2|
|[5.0,3.5,1.3]|        0.3|
|[5.0,3.6,1.4]|        0.2|
|[5.1,3.7,1.5]|        0.4|
|[5.2,2.7,3.9]|        1.4|
|[5.4,3.7,1.5]|        0.2|
|[5.4,3.9,1.3]|        0.4|
|[5.6,2.5,3.9]|        1.1|
|[5.6,2.7,4.2]|        1.3|
|[5.7,2.5,5.0]|        2.0|
|[5.7,2.8,4.5]|        1.3|
|[5.7,3.0,4.2]|        1.2|
|[5.8,4.0,1.2]|        0.2|
|[6.0,3.4,4.5]|        1.6|
+-------------+-----------+
only showing top 20 rows
"""

test_results = lr_model.evaluate(test)

print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
# RMSE: 0.1562736769938872
# MSE: 0.024421462121189785

training_summary = lr_model.summary

print("RMSE: {}".format(training_summary.rootMeanSquaredError))
print("r2: {}".format(training_summary.r2))
# RMSE: 0.20245315626083596
# r2: 0.9284703394101349

spark.stop()