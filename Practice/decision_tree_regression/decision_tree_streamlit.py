import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split

# Load the Boston housing dataset
iris = load_iris()
X = iris.data
y = iris.target

# Streamlit app
st.title("Decision Tree Visualization")


def load_initial_graph(dataset, ax):
    X = dataset.data
    y = dataset.target
    ax.scatter(X[:, 0], y, color="black")
    ax.set_title("Initial Graph")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Target")
    return X, y


# Sidebar for parameter inputs
st.sidebar.header("Model Parameters")
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)

# Load initial graph
fig, ax = plt.subplots()

# plot initial graph
X, y = load_initial_graph(iris, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
orig = st.pyplot(fig)

# Train a Decision Tree Regressor with user-defined parameters
model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf)
model.fit(X_train, y_train)

# Plot the decision tree
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=iris.feature_names, ax=ax)
st.pyplot(fig)

# Print the tree
st.text("Decision Tree Structure:")
st.text(model)
