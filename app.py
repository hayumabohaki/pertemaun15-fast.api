# import library yang diperlukan
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify, render_template
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset contoh
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Membuat model sederhana untuk contoh
X = dataset.drop('class', axis=1)
y = dataset['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Inisialisasi Flask
app = Flask(__name__)

# Route untuk predict API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    prediction = model.predict([input_data])[0]
    return jsonify(prediction)

# Route untuk feature exploration data analysis
@app.route('/explore')
def explore():
    # Melakukan analisis eksplorasi fitur
    plt.figure(figsize=(12, 8))
    sns.pairplot(dataset, hue='class', markers=["o", "s", "D"])
    plt.title('Pairplot of Iris Dataset Features')
    plt.savefig('static/pairplot.png')  # Menyimpan gambar pairplot

    # Mengirimkan nama file gambar ke explore.html
    image_path = 'pairplot.png'
    return render_template('explore.html', image_path=image_path)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
