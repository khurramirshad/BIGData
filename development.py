# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.clustering import KMeans
from mpl_toolkits.mplot3d import Axes3D

# COMMAND ----------

data_df = spark.read.csv("dbfs:/mnt/dbstables/brewery_data_complete_extended.csv", header=True, inferSchema=True)

# IF REQUIRED YOU CAN FILTER THE DATASET INTO SPECIFIC YEARS SINCE YOUR CHART WILL BE CLUSTERED DUE TO HUGE VOLUME OF DATA

# COMMAND ----------

# MAGIC %md
# MAGIC - Calculate summary statistics for numerical variables such as fermentation time, temperature, pH level, gravity, alcohol content, bitterness, color, volume produced, total sales, quality score, brewhouse efficiency, and losses during brewing, fermentation, and bottling/kegging.
# MAGIC - Explore distributions, central tendencies, and variability of these variables across batches and beer styles.

# COMMAND ----------

# Select numerical variables for analysis
numerical_vars = ['Fermentation_Time', 'Temperature', 'pH_Level', 'Gravity', 
                  'Alcohol_Content', 'Bitterness', 'Color', 'Volume_Produced', 
                  'Total_Sales', 'Quality_Score', 'Brewhouse_Efficiency', 
                  'Loss_During_Brewing', 'Loss_During_Fermentation', 'Loss_During_Bottling_Kegging']

# Summary statistics
summary_stats = data_df.select(*numerical_vars).describe()
display(summary_stats) # use this DF for plotting


# COMMAND ----------

# Explore distributions by grouping by beer style
for var in numerical_vars:
    data_df.groupBy('Beer_Style').agg(mean(var), stddev(var), min(var), max(var)).display()

# use this DF for plotting

# COMMAND ----------

# MAGIC %md
# MAGIC - Explore correlations between brewing parameters (e.g., fermentation time, temperature, pH level, gravity) and quality scores or total sales.
# MAGIC - Investigate how different ingredients ratios relate to quality scores and sales figures.

# COMMAND ----------

# Select relevant columns for correlation analysis
columns = ['Fermentation_Time', 'Temperature', 'pH_Level', 'Gravity', 
           'Quality_Score', 'Total_Sales', 'Ingredient_Ratio']

# Drop rows with null values in selected columns
df = data_df.select(*columns).dropna()

# Convert Ingredient_Ratio to numerical values
df = df.withColumn("Malt_Ratio", split(col("Ingredient_Ratio"), ":").getItem(0).cast("double")) \
    .withColumn("Hops_Ratio", split(col("Ingredient_Ratio"), ":").getItem(1).cast("double")) \
    .withColumn("Yeast_Ratio", split(col("Ingredient_Ratio"), ":").getItem(2).cast("double")) \
    .drop("Ingredient_Ratio")

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=['Fermentation_Time', 'Temperature', 'pH_Level', 'Gravity', 'Malt_Ratio', 'Hops_Ratio', 'Yeast_Ratio'], outputCol="features")
df_assembled = assembler.transform(df)

# Calculate correlation matrix
correlation_matrix = Correlation.corr(df_assembled, "features").head()

# Extract correlation matrix
corr_matrix = correlation_matrix[0].toArray()

# Print correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# COMMAND ----------

# MAGIC %md
# MAGIC - Analyze trends and seasonality in sales data over time.
# MAGIC - Investigate if there are any temporal patterns in brewing parameters and quality scores.

# COMMAND ----------

# Extract year and month from Brew_Date
df = data_df.withColumn("Year", year("Brew_Date")).withColumn("Month", month("Brew_Date"))

# Sales analysis over time
sales_trend = df.groupBy("Year", "Month").agg(avg("Total_Sales").alias("Avg_Sales"), stddev("Total_Sales").alias("StdDev_Sales")).orderBy("Year", "Month")


# COMMAND ----------

sales_trend.display() # use this DF for plotting

# COMMAND ----------

# Brewing parameters and quality scores analysis over time
windowSpec = Window.partitionBy("Year", "Month")

# For brewing parameters and quality scores analysis, we use window functions to calculate the average fermentation time and quality score for each month.
brewing_quality_analysis = df.withColumn("Avg_Fermentation_Time", avg("Fermentation_Time").over(windowSpec)) \
                              .withColumn("Avg_Quality_Score", avg("Quality_Score").over(windowSpec)) \
                              .select("Year", "Month", "Avg_Fermentation_Time", "Avg_Quality_Score") \
                              .distinct() \
                              .orderBy("Year", "Month")



# COMMAND ----------

brewing_quality_analysis.display() # use this DF for plotting

# COMMAND ----------

# MAGIC %md
# MAGIC - Visualize sales figures across different beer styles and locations using bar charts or heatmaps.
# MAGIC - Identify top-selling beer styles and locations.
# MAGIC - Explore variations in sales performance over time.

# COMMAND ----------

# Aggregate sales figures across beer styles and locations
sales_by_style_location = data_df.groupBy("Beer_Style", "Location").sum("Total_Sales").orderBy("sum(Total_Sales)", ascending=False)

# Convert Spark DataFrame to Pandas DataFrame for visualization
sales_by_style_location_pd = sales_by_style_location.toPandas()

# Plot sales figures across beer styles and locations using a heatmap
plt.figure(figsize=(12, 8))
heatmap_data = sales_by_style_location_pd.pivot("Beer_Style", "Location", "sum(Total_Sales)")
plt.imshow(heatmap_data, cmap="YlGnBu", interpolation="nearest")
plt.colorbar(label="Total Sales")
plt.xlabel("Location")
plt.ylabel("Beer Style")
plt.title("Sales Figures Across Beer Styles and Locations")
plt.xticks(range(len(sales_by_style_location_pd["Location"].unique())), sales_by_style_location_pd["Location"].unique(), rotation=45)
plt.yticks(range(len(sales_by_style_location_pd["Beer_Style"].unique())), sales_by_style_location_pd["Beer_Style"].unique())
plt.show()


# COMMAND ----------

# Identify top-selling beer styles and locations
top_selling_styles = data_df.groupBy("Beer_Style").sum("Total_Sales").orderBy("sum(Total_Sales)", ascending=False).limit(5)
top_selling_locations = data_df.groupBy("Location").sum("Total_Sales").orderBy("sum(Total_Sales)", ascending=False).limit(5)

# COMMAND ----------

top_selling_styles.display() # use this DF for plotting

# COMMAND ----------

top_selling_locations.display() # use this DF for plotting

# COMMAND ----------

# Explore variations in sales performance over time
sales_over_time = data_df.groupBy("Brew_Date").sum("Total_Sales").orderBy("Brew_Date")

# Convert Spark DataFrame to Pandas DataFrame for visualization
sales_over_time_pd = sales_over_time.toPandas()

# Plot variations in sales performance over time
plt.figure(figsize=(12, 6))
plt.plot(sales_over_time_pd["Brew_Date"], sales_over_time_pd["sum(Total_Sales)"], color='green')
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Variations in Sales Performance Over Time")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

sales_over_time.display() # use this DF for plotting

# COMMAND ----------

# MAGIC %md
# MAGIC - Visualize quality scores distribution across batches using histograms or box plots.
# MAGIC - Identify any outliers or trends in quality scores.
# MAGIC - Investigate if certain brewing parameters consistently lead to higher or lower quality scores.

# COMMAND ----------

# Filter out null values in Quality_Score column
df_filtered = data_df.filter(data_df["Quality_Score"].isNotNull())

# Convert Spark DataFrame to Pandas DataFrame for visualization
quality_scores_pd = df_filtered.select("Quality_Score").toPandas()

# Visualize quality scores distribution across batches using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(quality_scores_pd["Quality_Score"], kde=True, color='skyblue')
plt.xlabel("Quality Score")
plt.ylabel("Frequency")
plt.title("Quality Scores Distribution Across Batches")
plt.show()


# COMMAND ----------

# Visualize quality scores distribution using a box plot
plt.figure(figsize=(8, 6))
sns.boxplot(y="Quality_Score", data=quality_scores_pd, color='lightcoral')
plt.ylabel("Quality Score")
plt.title("Quality Scores Distribution Across Batches")
plt.show()


# COMMAND ----------

# Identify outliers in quality scores using the interquartile range (IQR) method.
quantiles = quality_scores_pd["Quality_Score"].quantile([0.25, 0.75])
iqr = quantiles[0.75] - quantiles[0.25]
lower_bound = quantiles[0.25] - 1.5 * iqr
upper_bound = quantiles[0.75] + 1.5 * iqr
outliers = quality_scores_pd[(quality_scores_pd["Quality_Score"] < lower_bound) | (quality_scores_pd["Quality_Score"] > upper_bound)]

print("Outliers in Quality Scores:")
print(outliers)

# Investigate if certain brewing parameters consistently lead to higher or lower quality scores
brewing_parameters_correlation = df_filtered.select("Fermentation_Time", "Temperature", "pH_Level", "Gravity", "Quality_Score").toPandas().corr()

print("Correlation between Brewing Parameters and Quality Score:")
print(brewing_parameters_correlation["Quality_Score"])

# COMMAND ----------

# MAGIC %md
# MAGIC - Analyze brewhouse efficiency and losses at different stages of production.
# MAGIC - Identify areas for improvement in the supply chain based on production volume and losses.
# MAGIC - Visualize production volume over time to identify any production bottlenecks.

# COMMAND ----------

# Analyze brewhouse efficiency and losses at different stages of production
brewhouse_analysis = data_df.select("Brewhouse_Efficiency", "Loss_During_Brewing", "Loss_During_Fermentation", "Loss_During_Bottling_Kegging").describe()

print("Brewhouse Efficiency and Losses Analysis:")
brewhouse_analysis.display() # use this DF for plotting

# Identify areas for improvement in the supply chain based on production volume and losses
supply_chain_analysis = data_df.select("Volume_Produced", "Loss_During_Brewing", "Loss_During_Fermentation", "Loss_During_Bottling_Kegging").groupBy().sum().collect()[0]

production_volume = supply_chain_analysis["sum(Volume_Produced)"]
losses_during_brewing = supply_chain_analysis["sum(Loss_During_Brewing)"]
losses_during_fermentation = supply_chain_analysis["sum(Loss_During_Fermentation)"]
losses_during_bottling_kegging = supply_chain_analysis["sum(Loss_During_Bottling_Kegging)"]

print("Production Volume:", production_volume)
print("Total Losses During Brewing:", losses_during_brewing)
print("Total Losses During Fermentation:", losses_during_fermentation)
print("Total Losses During Bottling/Kegging:", losses_during_bottling_kegging)

# Visualize production volume over time to identify any production bottlenecks
production_volume_over_time = data_df.groupBy("Brew_Date").sum("Volume_Produced").orderBy("Brew_Date").toPandas()


# COMMAND ----------

plt.figure(figsize=(12, 6))
plt.plot(production_volume_over_time["Brew_Date"], production_volume_over_time["sum(Volume_Produced)"], color='purple')
plt.xlabel("Date")
plt.ylabel("Production Volume")
plt.title("Production Volume Over Time")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Apply techniques like PCA (Principal Component Analysis) to reduce the dimensionality of the dataset.
# MAGIC - Use clustering algorithms (e.g., K-means) to group batches based on brewing parameters and quality scores.
# MAGIC - Visualize clusters using scatter plots or 3D plots to identify any patterns or clusters of similar batches.

# COMMAND ----------

short_df = data_df.withColumn("Year", year("Brew_Date")).withColumn("Month", month("Brew_Date"))
short_df = short_df.filter(col("Year") == 2023)

# COMMAND ----------

short_df.display()

# COMMAND ----------

# We select relevant features for PCA and clustering.
# We assemble the features into a single vector and standardize them.
# We apply PCA to reduce the dimensionality of the dataset to 3 principal components.
# We apply K-means clustering to group batches based on the PCA features.

# COMMAND ----------

# Select relevant features for PCA and clustering
features = ['Fermentation_Time', 'Temperature', 'pH_Level', 'Gravity', 'Alcohol_Content', 'Bitterness', 'Color', 'Volume_Produced', 'Quality_Score']

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_assembled = assembler.transform(short_df)

# Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

# Apply PCA to reduce dimensionality
pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)

# Apply K-means clustering
kmeans = KMeans(k=3, seed=42, featuresCol="pca_features")
kmeans_model = kmeans.fit(df_pca)
df_clustered = kmeans_model.transform(df_pca)

# Visualize clusters using scatter plot or 3D plot
cluster_centers = kmeans_model.clusterCenters()

# Convert Spark DataFrame to Pandas DataFrame for visualization
df_clustered_pd = df_clustered.select("pca_features", "prediction").toPandas()

# Extract PCA features for plotting
pca_features_pd = df_clustered_pd['pca_features'].apply(lambda x: [float(i) for i in x]).tolist()
x_values = [x[0] for x in pca_features_pd]
y_values = [x[1] for x in pca_features_pd]
z_values = [x[2] for x in pca_features_pd]


# COMMAND ----------

# Plot clusters using scatter plot or 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_values, y_values, z_values, c=df_clustered_pd['prediction'], cmap='viridis', s=50)
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='*', s=300)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Clusters of Batches based on PCA Features')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Build predictive models to forecast sales or quality scores based on brewing parameters.
# MAGIC - Evaluate model performance using techniques like cross-validation.
# MAGIC - Visualize predicted vs. actual sales or quality scores to assess model accuracy.

# COMMAND ----------

# We build a linear regression model using the training data.
# We make predictions on the test data and evaluate the model performance using Root Mean Squared Error (RMSE).
# We visualize predicted vs. actual sales or quality scores using a scatter plot.

# COMMAND ----------

# Select features and target variable
features = ['Fermentation_Time', 'Temperature', 'pH_Level', 'Gravity']
target = 'Total_Sales'  # or 'Quality_Score' for quality score prediction

# Prepare data for modeling
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_assembled = assembler.transform(short_df).select("features", target)

# Split data into training and testing sets
(training_data, test_data) = df_assembled.randomSplit([0.8, 0.2], seed=42)

# Build a linear regression model
lr = LinearRegression(featuresCol="features", labelCol=target)
lr_model = lr.fit(training_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate model performance
evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE):", rmse)

# Visualize predicted vs. actual sales or quality scores
predictions_pd = predictions.select(target, "prediction").toPandas()


# COMMAND ----------

predictions.select(target, "prediction").display() # use this DF for plotting

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.scatter(predictions_pd[target], predictions_pd["prediction"], color='blue')
plt.plot(predictions_pd[target], predictions_pd[target], color='red', linestyle='--')
plt.xlabel("Actual " + target)
plt.ylabel("Predicted " + target)
plt.title("Predicted vs. Actual " + target)
plt.grid(True)
plt.show()

# COMMAND ----------


