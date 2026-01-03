from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = { 'owner': 'mlops_user', 'retries': 1 }

with DAG(
    'sentiment_retraining_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    retrain_model = BashOperator(
        task_id='retrain_and_serialize',
        bash_command='python /app/train_and_serialize.py ',
    )
