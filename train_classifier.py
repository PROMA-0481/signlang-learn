import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data from the pickle file
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: 'data.pickle' not found. Make sure you run 'create_dataset.py' first.")
    exit()
except Exception as e:
    print(f"Error loading 'data.pickle': {e}")
    exit()

# Extract data and labels
data = np.asarray(data_dict.get('data'))
labels = np.asarray(data_dict.get('labels'))

# Check if data and labels are loaded correctly and are not empty
if data is None or labels is None or len(data) == 0 or len(labels) == 0:
    print("Error: No data found in 'data.pickle'. Check 'create_dataset.py'.")
    exit()

# Encode the labels (if they are strings) to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, shuffle=True, stratify=encoded_labels)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(random_state=42)  # Added random_state for reproducibility
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Evaluate the model
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model and the label encoder
model_data = {'model': model, 'label_encoder': label_encoder}
try:
    with open('model.p', 'wb') as f:
        pickle.dump(model_data, f)
    print("Trained model and label encoder saved as 'model.p'")
except Exception as e:
    print(f"Error saving the model: {e}")