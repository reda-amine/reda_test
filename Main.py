from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

print(type(iris))

print(dir(iris))

print(iris.feature_names)

print(iris.data[:5])

print(iris.target_names )


target = iris.target
print (target )

for i in [0,1,2]:

    print("classe : %s, nb exemplaires: %s" % (i, len(target[ target == i]) ) )


data = iris.data

data_test = train_test_split(data, target
                                 , random_state=0
                                 , train_size=0.75)
data_train, data_test, target_train, target_test = data_test


clf = GaussianNB()
clf.fit(data_train, target_train)
result = clf.predict(data_test)


print("\n\n la qualité de la prédiction: %s" %  accuracy_score(result, target_test) )

conf = confusion_matrix(target_test, result)

print("\n\n la matrice de confusion :")

print(  conf )
