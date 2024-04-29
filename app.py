import pandas as pd
import os
import hashlib
import psycopg2

from typing import List
from fastapi import Depends, FastAPI
from datetime import datetime
from catboost import CatBoostClassifier

from schema import PostGet, Response


def get_model_path(path: str, type_model: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        if type_model == 'model_control':
            CONTROL_MODEL_PATH = '/workdir/user_input/model_control'
            return CONTROL_MODEL_PATH
        elif type_model == 'model_test':
            TEST_MODEL_PATH = '/workdir/user_input/model_test'
            return TEST_MODEL_PATH
    else:
        MODEL_PATH = path
        return MODEL_PATH


def load_models(type_model: str):
    if type_model == 'model_control':
        model_path = get_model_path("model/model_control.cbm", type_model='model_control')
        cbc_mod = CatBoostClassifier().load_model(fname=model_path)
    elif type_model == 'model_test':
        model_path = get_model_path("model/model_test.cbm", type_model='model_test')
        cbc_mod = CatBoostClassifier().load_model(fname=model_path)

    return cbc_mod


control_model = load_models(type_model='model_control')
test_model = load_models(type_model='model_test')


def select(query):
    return pd.read_sql(sql=query,
                       con='postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml')


user = select('select * from "user_data";')
post = select('select * from "kirkor96_post_features_lesson_22"')

app = FastAPI()


def get_time(func, args):
    times = []

    start = time.time()
    for i in range(100):
        func(*args)
        end = time.time()
        times.append(end - start)
        start = time.time()

    return {'iters': 100, 'mean_time': np.array(times).mean(), 'std': np.array(times).std()}


def get_exp_group(user_id: int):
    salt = 'my_experiment'
    user_hash = str(user_id)
    val_str = user_hash + salt
    val_num = int(hashlib.md5(val_str.encode()).hexdigest(), 16) % 2
    return val_num


def prepare_features(user_id, model, current_hh, user_table, post_table):
    one_user = user_table.loc[user_table['user_id'] == user_id,
                              ['user_id'] + [f for f in user_table if f in model.feature_names_]]
    one_user['time_hh'] = current_hh
    one_user['join_id'] = 1

    post = post_table.copy()
    post['join_id'] = 1

    data = post.merge(one_user, how='left', on=['join_id'])

    return user_id, data['post_id'].values, data[model.feature_names_], data


def get_ranking_posts(user_id, data, posts, model, s_data, n_rec=5):
    result_ff = []
    result = pd.DataFrame(
        {
            'post_id': posts,
            'rank': model.predict(data[model.feature_names_], prediction_type="Probability")[:, -1],
        }
    ).sort_values('rank', ascending=False).head(n_rec).set_index('post_id').to_dict()

    for i in result['rank']:
        result_f = {
            'id': i,
            'text': s_data.loc[s_data['post_id'] == i]['text'].values[0],
            'topic': s_data.loc[s_data['post_id'] == i]['topic'].values[0]
        }

        result_ff.append(result_f)

    result_fff = {
        'exp_group': 'control',
        'recommendations': result_ff
    }

    return result_fff


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 5):
    time = pd.to_datetime(str(time)).hour
    if get_exp_group(id) == 0:
        user_id, posts, score_data, data = prepare_features(id, control_model, time, user, post)
        post_recommendation = get_ranking_posts(user_id, score_data, posts, control_model, data, n_rec=limit)
    elif get_exp_group(id) == 1:
        user_id, posts, score_data, data = prepare_features(id, test_model, time, user, post)
        post_recommendation = get_ranking_posts(user_id, score_data, posts, test_model, data, n_rec=limit)
    return post_recommendation


@app.get("/get_user_features/", response_model=TestResponse)
def get_user_features(user_id: int, time: datetime, limit: int = 5) -> TestResponse:
    hour = int(pd.to_datetime(str(time)).hour)
    data = prepare_features(user_id, estimator, hour, user, post)
    result = {'user_id': user_id, 'data': data[estimator.feature_names_].head(3).to_dict()}
    return result


@app.get("/test_prepare_features", response_model=BenchmarResponseF)
def test_prepare_features(user_id: int, time: datetime) -> BenchmarResponseF:
    hour = int(pd.to_datetime(str(time)).hour)
    return get_time(prepare_features, (user_id, estimator, hour, user, post))


@app.get("/test_get_ranking_posts", response_model=BenchmarResponseF)
def test_get_ranking_posts(user_id: int, time: datetime) -> BenchmarResponseF:
    hour = int(pd.to_datetime(str(time)).hour)
    data = prepare_features(user_id, estimator, hour, user, post)
    return get_time(get_ranking_posts, (data, estimator, 5))


@app.get("/test_get_preds", response_model=BenchmarResponseF)
def test_get_preds(user_id: int, time: datetime) -> BenchmarResponseF:
    hour = int(pd.to_datetime(str(time)).hour)
    data = prepare_features(user_id, estimator, hour, user, post)
    return get_time(get_preds, (data, estimator))