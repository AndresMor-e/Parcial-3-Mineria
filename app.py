
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

st.set_page_config(page_title="Iris Species Classifier", layout="wide")
st.title("Iris Species Classification Dashboard")
st.write("Integrants: Roberto Escobar, Andrés Moreno, Laura Sanchez, Isabella Vega")

df = pd.read_csv("Iris.csv")

X = df.drop(["Species", "Id"], axis=1)
y = df["Species"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

st.subheader("Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")

st.subheader("Predict Species for New Flower")

colA, colB = st.columns(2)

with colA:
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.0)

with colB:
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.5)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

new_sample = [[sepal_length, sepal_width, petal_length, petal_width]]
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)[0]

st.success(f"Predicted Species: {prediction}")

st.subheader("3D Visualization of Dataset and New Sample")

fig_3d = px.scatter_3d(
    df,
    x="SepalLengthCm",
    y="SepalWidthCm",
    z="PetalLengthCm",
    color="Species",
    title="3D Scatter Plot of Iris Dataset"
)

fig_3d.add_scatter3d(
    x=[sepal_length],
    y=[sepal_width],
    z=[petal_length],
    mode="markers",
    marker=dict(size=8,color="red"),
    name="New Sample"
)

st.plotly_chart(fig_3d, use_container_width=True)

st.subheader("Additional Visualizations")

tab1, tab2 = st.tabs(["Histograms", "Scatter Matrix"])

with tab1:
    fig_hist, ax = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df, x="SepalLengthCm", hue="Species", kde=True, ax=ax[0][0])
    sns.histplot(df, x="SepalWidthCm", hue="Species", kde=True, ax=ax[0][1])
    sns.histplot(df, x="PetalLengthCm", hue="Species", kde=True, ax=ax[1][0])
    sns.histplot(df, x="PetalWidthCm", hue="Species", kde=True, ax=ax[1][1])
    st.pyplot(fig_hist)

with tab2:
    fig_pair = sns.pairplot(df.drop(columns=["Id"]), hue="Species")
    st.pyplot(fig_pair)

st.subheader("Full Dataset")
st.dataframe(df)

