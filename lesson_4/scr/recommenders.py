
import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid,             self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
          self.preserved_user_item_matrix = self.user_item_matrix
          self.user_item_matrix = bm25_weight(self.user_item_matrix)

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):

        user_item_matrix = pd.pivot_table(data,
                                  index='user_id', columns='item_id',
                                  values='quantity',
                                  aggfunc='count',
                                  fill_value=0
                                 )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).tocsr(), show_progress=True)

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=128, regularization=0.05, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                             regularization=regularization,
                                             iterations=iterations,
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).tocsr(), show_progress=True)

        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # Получаем идентификатор пользователя из его имени
        user_id = self.userid_to_id[user]

        # Получаем топ-N купленных пользователем товаров
        top_items = self.preserved_user_item_matrix.drop(columns=999999).loc[user_id].sort_values(ascending=False).head(N)

        # Получаем похожие товары для каждого из топ-N товаров
        similar_items = []
        for item_id, score in top_items.items():
            recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
            similar_items.append(self.id_to_itemid[recs[0][1]])

        # Удаляем дубликаты и получаем список рекомендаций товаров, похожих на топ-N купленных пользователем товаров
        recommended_items = list(set(similar_items))

        assert len(recommended_items) == N, 'Количество рекомендаций != {}'.format(N)
        return recommended_items

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        similar_users = self.model.similar_users(self.userid_to_id[user], N=6)

        similar_user_items = []

        for sim_user_id in similar_users[0][1:]:

          recs = self.own_recommender.recommend(userid= sim_user_id, 
                          user_items=csr_matrix(self.user_item_matrix).tocsr()[sim_user_id],   # на вход user-item matrix
                          N=5, 
                          filter_already_liked_items=False, 
                          filter_items=None, 
                          recalculate_user=False)
          similar_user_items.append(self.id_to_itemid[recs[0][0]])
        
        # Удаляем дубликаты и получаем список рекомендаций товаров, похожих на топ-N купленных пользователем товаров
        res = list(set(similar_user_items))

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
