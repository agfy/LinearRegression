import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import scipy

data_train = pd.read_csv('salary-train.csv')

data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
desc_lower_train = data_train['FullDescription'].str.lower()
vectorizer = TfidfVectorizer(min_df=5)
X_desc_train = vectorizer.fit_transform(desc_lower_train)

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = scipy.sparse.hstack([X_desc_train, X_train_categ])

data_test = pd.read_csv('salary-test-mini.csv')

data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
desc_lower_test = data_test['FullDescription'].str.lower()
X_desc_test = vectorizer.transform(desc_lower_test)

X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_test = scipy.sparse.hstack([X_desc_test, X_test_categ])

ridge = Ridge(alpha=1, random_state=241)
ridge.fit(X_train, data_train['SalaryNormalized'])
result = ridge.predict(X_test)