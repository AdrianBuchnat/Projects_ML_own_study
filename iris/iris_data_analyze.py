import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv(
    r"iris.data", 
    header = None,
    names=[
        'petal lenght', 'petal width', 'sepal lenght', 'sepal width', 'species'
    ])


# print(iris.shape)
# print(iris.head())
# print(iris.shape[0])
# print(iris.shape[1])

x_min, x_max = iris['petal lenght'].min() - .5, iris['petal lenght'].max() + .5
y_min, y_max = iris['petal width'].min() - .5, iris['petal width'].max() + .5

colors = {
    'Iris-setosa': 'red',
    'Iris-versicolor': 'blue',
    'Iris-virginica': 'green'
}

fig, ax = plt.subplots(figsize=(8, 6))

for key, group in iris.groupby(by='species'):
    plt.scatter(
        group['petal lenght'],
        group['petal width'],
        c = colors[key],
        label = key
    )


ax.legend()
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.set_title('Iris dataset categorized')

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

plt_position = 1

feature_x = 'petal width'

for feature_y in iris.columns[:4]:

    plt.subplot(2, 2, plt_position)

    for species, color in colors.items():

        plt.scatter(
            iris.loc[iris['species'] == species, feature_x],
            iris.loc[iris['species'] == species, feature_y],
            label = species,
            alpha = 0.45,
            color = color
        )

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt_position += 1

plt.show()

pd.plotting.scatter_matrix(
    iris,
    figsize=(8, 8), 
    color = iris['species'].apply(lambda x: colors[x])
)

plt.show()
 


