import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

# Initialize the Dash app
app = dash.Dash(__name__)

# Set Pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load your data from a CSV file
df = pd.read_csv('data.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables (e.g., 'Tm' and 'Pos') using LabelEncoder
label_encoder = LabelEncoder()
df['Tm'] = label_encoder.fit_transform(df['Tm'])
df['Pos'] = label_encoder.fit_transform(df['Pos'])

# Convert 'Ctch%' to a numerical format by removing the '%' and handling any non-numeric values
df['Ctch%'] = pd.to_numeric(df['Ctch%'], errors='coerce')
df['Ctch%'].fillna(0, inplace=True)

# Split your data into features (X) and the target variable (y)
X = df[['Tm', 'Age', 'Pos', 'G', 'GS', 'Tgt', 'Rec', 'Ctch%', 'Y/R', 'TD', '1D', 'Succ%', 'Lng', 'Y/Tgt', 'R/G', 'Y/G', 'Fmb']]
y = df['Yds']

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DataFrame with player names for the test set
player_names = df.loc[X_test.index]['Player']

# Create and train a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the regression model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a scatter plot with player names
scatter_data = pd.DataFrame({'Player': player_names, 'Actual': y_test, 'Predicted': y_pred})

scatter_fig = px.scatter(scatter_data, x='Actual', y='Predicted', text='Player', title='Actual vs. Predicted Values')

# Customize text and layout for better readability
scatter_fig.update_traces(textposition='top center', textfont_size=10)
scatter_fig.update_layout(xaxis_title='Actual', yaxis_title='Predicted')

# Define the layout of the Dash app
app.layout = html.Div(children=[
    html.H1("My Dash App"),
    dcc.Graph(
        id='scatter-plot',
        figure=scatter_fig  # Use the scatter_fig you created in your existing code
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
