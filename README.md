# Health-Spend-Recommender

People are often unsure about how much of money they
should spend on healthcare related issues every year.
It is therefore necessary to determine healthcare expenditure
based on users' personal transaction history as well as their attributes or
characteristics that define them.

The model designed in this repo uses a simple statistical model (Random Forest Classifier) to
make informed predictions on how much of money a user may need to save for
expense related to healthcare. Given below is a list of the modules and their
roles in making these predictions.

- [s3.py](https://github.com/skhan1020/health-spend-recommender/blob/master/s3.py): module to wrap boto3 s3 services
- [s3data.py](https://github.com/skhan1020/health-spend-recommender/blob/master/s3data.py): both employee attribute data and transaction
  histories from 10/22/2017 to 04/22/2020, are downloaded from an s3 bucket and saved as csv
  files in the directories './data/emp/' and './data/txns/', respectively.
- [makedb.py](https://github.com/skhan1020/health-spend-recommender/blob/master/makedb.py): creates both Employees and Transactions tables in a sqlite
  database from the csv files downloaded using *s3data.py*.
- [employees.py](https://github.com/skhan1020/health-spend-recommender/blob/master/employees.py): preliminary feature preprocessing is performed on data extracted from the
  Employees table. For plots and additional details on how the raw data is
  cleaned, please refer to [employees.ipynb](https://github.com/skhan1020/health-spend-recommender/tree/master/notebooks) in notebooks repo.
- [inputfeatures.py](https://github.com/skhan1020/health-spend-recommender/blob/master/inputfeatures.py): more advanced feature feature engineering is performed
  on the employee attributes, especially the annual income as a result of which employees are now classified
  into different income groups. Categorical features are
  one-hot-encoded. Plots and a step-by-step transformation of the data are
  provided in [inputfeatures.ipynb](https://github.com/skhan1020/health-spend-recommender/blob/master/notebooks/inputfeatures.ipynb).
- [target.py](https://github.com/skhan1020/health-spend-recommender/blob/master/target.py): this module is perhaps the most important one as it is
  reponsible for building the target variable from the transaction history of
  employees. The function *nospend()* looks at all users who have 12 months of transaction data but did not spend anything on healthcare. 
  By default,  this group of people have been added to the database (boolean variable *add_nospend* is set to True). 
  To reduce the effect of noise, the yearly expenditure is grouped
  into different categories. As before, [target.ipynb](https://github.com/skhan1020/health-spend-recommender/blob/master/notebooks/target.ipynb) offers a step-by-step
  guide with visual aid on how the target variable is built from the raw data.
- [emptransac.py](https://github.com/skhan1020/health-spend-recommender/blob/master/emptransac.py): input features and target variable are merged and stored in a sqlite database (EmployeesTransactions Table)
- [model.py](https://github.com/skhan1020/health-spend-recommender/blob/master/model.py): Random Forest Classifier is used to make predictions on the
  average yearly amount a user should ideally put into their FSA. ROC Curves
  and Confusion Matrix generated are indicators of model accuracy and performance.
  The jupyter notebook [model.ipynb](https://github.com/skhan1020/health-spend-recommender/blob/master/notebooks/model.ipynb) contains plots of ROC Curves as well as Confusion Matrices.

