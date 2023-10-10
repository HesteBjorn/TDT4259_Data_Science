import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot


# Load the dataset
file_path = 'C:\\Users\\johan\\Downloads\\Nedlastninger\\consumption_temp.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())


# Check for missing values in each column
missing_values = df.isnull().sum()
print(missing_values)

# Get the mean of each column
print(df.describe())

#Get descriptive statistics for each location 
print(df.groupby('location').describe())

#display electricy consumption over time  for each location with a graph
df['time'] = pd.to_datetime(df['time'])
df.groupby('location')['consumption'].plot(legend=True)
plt.show()

# Create a new figure for displaying the hourly energy consumption patterns for each location
plt.figure(figsize=(20, 12))

# Loop through each unique location and plot the hourly energy consumption pattern
for i, location in enumerate(df['location'].unique()):
    plt.subplot(3, 2, i+1)
    location_data = df[df['location'] == location].sort_values('time').reset_index(drop=True)
    location_data['hour'] = location_data['time'].dt.hour
    sns.lineplot(x='hour', y='consumption', data=location_data, errorbar=None)
    plt.title(f'Hourly Energy Consumption Pattern in {location}')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Consumption (MW)')

plt.tight_layout()
plt.show()


# Calculate the correlation matrix for numerical variables
correlation_matrix = df[['consumption', 'temperature']].corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Consumption and Temperature')
plt.show()

# Plot the scatter plot between consumption and temperature
plt.figure(figsize=(8, 6))
sns.scatterplot(x='temperature', y='consumption', data=df)
plt.title('Scatter Plot of Consumption and Temperature')
plt.show()


# Initialize a dictionary to store the autocorrelation at 24-hour lag for each location
acf_24hr = {}

# Create a new figure for ACF plots
plt.figure(figsize=(20, 12))

# Loop through each unique location and plot the autocorrelation
for i, location in enumerate(df['location'].unique()):
    plt.subplot(3, 2, i+1)
    location_data = df[df['location'] == location]['consumption'].reset_index(drop=True)
    autocorrelation_plot(location_data.iloc[:500])  # Limiting to first 500 data points for better visibility
    plt.title(f'Autocorrelation for {location}')
    plt.xlim(0, 48)  # Focus on the first 48 lags to capture daily seasonality

plt.tight_layout()
plt.show()

df['week_number'] = df['time'].dt.isocalendar().week
# Calculate the average consumption for each week number for each location
avg_week_consumption = df.groupby(['location', 'week_number'])['consumption'].mean().reset_index()


# Create a new figure for displaying the average consumption for each week number across all locations
plt.figure(figsize=(20, 12))

# Loop through each unique location and plot the average consumption for each week number
for i, location in enumerate(df['location'].unique()):
    

    plt.subplot(3, 2, i+1)
    location_data = avg_week_consumption[avg_week_consumption['location'] == location]
    print(location_data.dtypes)
    print(location_data.isnull().sum())
    # Convert to appropriate data types
    location_data['week_number'] = location_data['week_number'].astype(int)
    location_data['consumption'] = location_data['consumption'].astype(float)

    sns.lineplot(x='week_number', y='consumption', data=location_data, marker='o')
    plt.title(f'Average Weekly Consumption Pattern in {location}')
    plt.xlabel('Week Number')
    plt.ylabel('Average Consumption (MW)')

plt.tight_layout()
plt.show()

# Extract just the date from the time column and create a new column for it
df['date'] = df['time'].dt.date

# Filter the data for 'Helsingfors' and group by the date to get daily consumption totals
daily_consumption_helsingfors = df[df['location'] == 'helsingfors'].groupby('date')['consumption'].sum().reset_index()

# Plot the daily consumption data for 'Helsingfors'
plt.figure(figsize=(20, 6))
sns.lineplot(x='date', y='consumption', data=daily_consumption_helsingfors)
plt.title('Daily Consumption Data for Helsingfors')
plt.xlabel('Date')
plt.ylabel('Total Consumption (MW)')
plt.xticks(rotation=45)
plt.show()


