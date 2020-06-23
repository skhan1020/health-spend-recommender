from flask import Flask, request, render_template
from flask_restful import Resource, Api
from sqlalchemy import  create_engine
from flask_sqlalchemy import SQLAlchemy
from makedb import employees_table, transactions_table
from emptransac import employee_transac
from model_app import  RandomForest

db_name = 'fsa.db'
engine = create_engine('sqlite:///' + db_name, echo=False)

app = Flask(__name__)
api = Api(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_name
db = SQLAlchemy(app)

db.Model.metadata.reflect(db.engine)


class Employees(db.Model):

    __tablename = 'Employees'
    __table_args__  = {'extend_existing':True}
    id = db.Column(db.Integer)
    hashed_employee_id = db.Column(db.Text, unique=True, primary_key=True)
    hashed_employer_id = db.Column(db.Text)
    gross_income =  db.Column(db.Float)
    pay_type = db.Column(db.Text)
    filing_status = db.Column(db.Text)
    language_preference = db.Column(db.Text)
    job_title = db.Column(db.Text)
    created_at = db.Column(db.Text)
    updated_at = db.Column(db.Text)
    account_status = db.Column(db.Text)
    deleted = db.Column(db.Boolean)
    city_id = db.Column(db.Float)
    birth_year  = db.Column(db.Float)
    gender = db.Column(db.Text)
    gender_confidence  =  db.Column(db.Float)
    hourly_rate = db.Column(db.Float)

    filepath = './data/emp/employees_data.csv'

    employees_table(filepath, engine)

class Transactions(db.Model):

    __tablename = 'Transactions'
    amount = db.Column(db.Integer)
    category_identifier = db.Column(db.Float)
    created_at = db.Column(db.Text)
    date =  db.Column(db.Text)
    employee_latitude = db.Column(db.Float)
    employee_longitude = db.Column(db.Float)
    hashed_owner_id = db.Column(db.Text)
    id = db.Column(db.Integer, primary_key=True)
    platform = db.Column(db.Text)
    substantiated_by_employee = db.Column(db.Boolean)
    t_id = db.Column(db.Text)
    txn_latitude = db.Column(db.Float)
    txn_longitude = db.Column(db.Float)
    unified_category_id  = db.Column(db.Float)
    updated_at = db.Column(db.Text)

    filepath = './data/txns/'

    transactions_table(filepath, engine)


class EmployeeTransactions(db.Model):

    __tablename = 'EmployeesTransactions'

    emp_id = db.Column(db.Text, primary_key=True)
    amount = db.Column(db.Integer)
    city_id = db.Column(db.Float)
    gender_female = db.Column(db.Text)
    gender_male = db.Column(db.Text)
    gender_unknown = db.Column(db.Text)
    filing_status_head =  db.Column(db.Integer)
    filing_status_separate =  db.Column(db.Integer)
    filing_status_joint =  db.Column(db.Integer)
    filing_status_single =  db.Column(db.Integer)
    age_category_adult = db.Column(db.Integer)
    age_category_middle_aged = db.Column(db.Integer)
    age_category_no_age = db.Column(db.Integer)
    age_category_retired = db.Column(db.Integer)
    age_category_young_adult = db.Column(db.Integer)
    income_high = db.Column(db.Integer)
    income_medium = db.Column(db.Integer)
    income_low = db.Column(db.Integer)



@app.route('/', methods=['GET', 'POST'])
def parameters():
    return render_template('parameters.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    

    if request.method ==  'POST':
        nclasses = int(request.form['nclasses'])
        amount_threshold = float(request.form['amt_thresh'])
        income_threshold  = float(request.form['income_thresh'])
        amt_vals = request.form['breakpoints']
        breakpoints  = [float(val) for  val  in amt_vals.split(',')]
         
    employee_transac(db_name, nclasses, breakpoints, amount_threshold, income_threshold, engine)

    graph1_url, graph2_url =  RandomForest(db_name, cf_matrix=False)
    graph3_url = RandomForest(db_name, cf_matrix=True)

    return  render_template('result.html', graph1=graph1_url, graph2=graph2_url, graph3=graph3_url)


if __name__ ==  '__main__':
    app.run(debug=True)

