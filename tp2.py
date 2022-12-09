import pandas

import matplotlib.pyplot as plt
import seaborn as sns

#importation de la dataset
D = pandas.read_csv("Iris.csv",header=0)

#Affichage des 10 premiers lignes
print(D.head(10))

#Affichage de la taille du dataset
#print (D.shape)


df = sns.load_dataset('iris')
df.head()
sns.scatterplot(x='sepal_width', y ='petal_length' , data = df , hue = 'species')
plt.show()

#Labelisation des espèces du dataset
D.loc[D["Species"] == "Iris-setosa" , "Species"] = 0
D.loc[D["Species"] == "Iris-versicolor" , "Species"] = 1
D.loc[D["Species"] == "Iris-virginica" , "Species"] = 2

#Affichage des 10 premiers lignes pour vérifier
print(D.head(10))

from sklearn.model_selection import train_test_split

#Séparation des attributs et des classes
X_data = D.iloc[:,1:5].values
Y_data=D.iloc[:,5].values

#détermination de la base de test (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3, shuffle = True, random_state = 123)

#Affichage des dimensions des deux bases
print("Dimensions de la base d'apprentissage: \n")
print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape))
print("Dimensions de la base d'apprentissage: \n")
print("X_test shape: {}".format(X_test.shape))
print("Y_test shape: {}".format(Y_test.shape))

#Affichage des 10 première données d'apprentissage
print("\n 10 premières données d’apprentissage:\n")
print("X_train")
print(X_train[0:10])
print("Y_train")
print(Y_train[0:10])

# #Affichage des 10 premières de test
print("\n 10 premières données de test:\n")
print("X_test")
print(X_test[0:10])
print("Y_test")
print(Y_test[0:10])


from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(3,3), epsilon=0.07, max_iter=1500)
classifier.fit(X_train, Y_train.astype('int'))

from sklearn import metrics
prediction = classifier.predict(X_test)
print('Y_test \n',Y_test)

print ('Prédiction \n',prediction)
print("accuracy of the perception: \n",metrics.accuracy_score(prediction,Y_test.astype('int')))

from pretty_confusion_matrix import pp_matrix_from_data
cmap= "PuRd"
pp_matrix_from_data(Y_test.astype('int'),prediction)    

params = [
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "learning_rate_init": 0.2,
        "max_iter" : 1500
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "learning_rate_init": 0.7,
        "max_iter" : 1500
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "learning_rate_init": 0.2,
        "max_iter" : 1500
    },
    {
        "solver": "sgd",
        "learning_rate": "invscaling",
        "learning_rate_init": 0.7,
        "max_iter" : 3000
    },
    {
        "solver": "adam", "learning_rate_init": 0.01,
        "max_iter" : 3000,
    },
    
]
labels = [
    "constant learning-rate 0.2",
    "constant learning-rate 0.7",
    "inv-scaling learning-rate 0.2",
    "inv-scaling learning-rate 0.7",
    "adam",
]
dataClassifiers = []
for i in range(len(params)):
  classifier = MLPClassifier(random_state=0, **params[i])
  classifier.fit(X_train, Y_train.astype('int'))
  print(labels[i]," : ",classifier.score(X_train, Y_train.astype('int')))
  dataClassifiers.append(classifier)

  plot_args = [
    {"c": "red", "linestyle": "-"},
    {"c": "green", "linestyle": "-"},
    {"c": "blue", "linestyle": "-"},
    {"c": "red", "linestyle": "--"},
    {"c": "green", "linestyle": "--"},
]

for i in range(len(dataClassifiers)):
  plt.plot(dataClassifiers[i].loss_curve_, **plot_args[i])
  plt.title(labels[i], fontsize=14)
  plt.show()