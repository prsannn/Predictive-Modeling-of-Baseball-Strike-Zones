import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

print(aaron_judge.description.unique())
print(aaron_judge.type.unique())

aaron_judge['type'] = aaron_judge['type'].map({'S': 1, 'B': 0})
print(aaron_judge['type'])
print(aaron_judge['plate_x'])
aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z', 'type'])


fig, ax = plt.subplots()
plt.scatter(x = aaron_judge['plate_x'], y = aaron_judge['plate_z'], c = aaron_judge['type'], cmap = plt.cm.coolwarm, alpha = 0.5)
plt.show()

training_set, validation_set = train_test_split(aaron_judge, random_state = 1)

gamma_vals= [0.1, 1, 3, 10, 100]
c_vals = [0.1, 1, 3, 10, 100]
for gamma in gamma_vals:
  for c in c_vals:
    classifier = SVC(kernel='rbf',gamma=gamma, C=c)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']), gamma, c)


classifier = SVC(kernel='rbf',gamma=1, C=3)
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']), gamma, c)
draw_boundary(ax, classifier)
plt.show()
