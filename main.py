import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv(r'files\power.csv')
X = df[['ts', 'KWH']]  # Assuming 'ts' and 'KWH' are the features
y = df['KWH']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=42)

# Fit Isolation Forest model
clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_train)

# Predict outliers
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Plot outliers
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (14,6), sharex = True, sharey = True)

# Plot training data
ax1.scatter(X_train['ts'], X_train['KWH'], c=y_pred_train, edgecolor='k', label='Inliers')
ax1.scatter(X_train[y_pred_train == -1]['ts'], X_train[y_pred_train == -1]['KWH'], c='r', marker='x', label='Outliers')
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('KWH')
ax1.set_title('Power Consumption Anomalies by Isolation Forest(Training Data)')
plt.legend()

# Plot test data
ax2.scatter(X_test['ts'], X_test['KWH'], c=y_pred_test, edgecolor='k', label='Inliers (Test)')
ax2.scatter(X_test[y_pred_test == -1]['ts'], X_test[y_pred_test == -1]['KWH'], c='r', marker = 'x', label='Outliers (Test)')
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('KWH')
ax2.set_title('Power Consumption Anomalies by Isolation Forest(Testing Data)')
plt.legend()
fig.savefig(r'files\power consumption anomalies.png')
plt.show()
