
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
def zxc(a):
    X = df.drop(a, axis = 1) # данные о всех показателях кроме столбца о выживании
    y = df[a] # столбец о сттусе выжил или нет
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train) #Тренеровочный набор
    X_test = sc.transform(X_test)#Набор для теста
    classifier = KNeighborsClassifier(n_neighbors = 5)# Применение класса с соседями
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    ac = accuracy_score(y_test, y_pred) * 100
    print('Процент', ac)
    return ac
# Шаг 1. Загрузка и очистка данных
import pandas as pd 
dtr = pd.read_csv('train.csv')
dte  = pd.read_csv('test.csv')
print(df.info())
dtr.drop(['id','langs','occupation_type','bdate','has_photo',"has_mobile",'followers_count','graduation','relation','life_main',"people_main",'city','last_seen','occupation_name','career_start',"career_end"], axis = 1, inplace = True)
dtr.dropna(inplace = True)

#
#print(df.groupby(by = '')[''].agg(['min', 'median','mean', 'max']))
#df[df['relation'] == 0.0].groupby(by = 'education_status')['result'].value_counts().plot(kind = 'pie')
#df['sex'].value_counts()
#plt.show()

def ef(status):
    if status == 'Full-time':
        return 2
    elif status == 'Distance Learning':
        return 1
    return 0

def es(status):
    if status == 'Student (Specialist)':
        return 0
    elif status == 'Undergraduate applicant':
        return 0.75
    elif status == 'Alumnus (Specialist)':
        return 1
    elif status == "Student (Bachelor's)":
        return 2
    elif status == "Alumnus (Bachelor's)":
        return 3
    elif status == "Student (Master's)":
        return 4
    elif status == "Alumnus (Master's)":
        return 5
    elif status == "Candidate of Sciences":
        return 6
    elif status == "PhD":
        return 7
    return 0

dtr['education_form'] = dtr['education_form'].apply(ef)
dtr['education_status'] = dtr['education_status'].apply(es)

print(dtr['education_status'].value_counts())
print(dtr.info())
print(zxc('result'))