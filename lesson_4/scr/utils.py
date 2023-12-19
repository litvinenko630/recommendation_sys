def prefilter_items(df, item_features=None, take_n_popular=5000):
    # Уберем самые популярные товары (их и так купят)
    popularity = df.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['share_unique_users'] = popularity['user_id'] / df['user_id'].nunique()
    top_popular = popularity[popularity['share_unique_users'] > 0.5]['item_id'].tolist()
    df = df[~df['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.0025].item_id.tolist()
    df = df[~df['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    recently_sold = df[df['day'] < 386]['item_id'].tolist()
    df = df[df['item_id'].isin(recently_sold)]

    # Уберем не интересные для рекоммендаций категории (department)
    unique_departments = item_features.groupby('department')['item_id'].nunique()
    top_departments = unique_departments[unique_departments > 100].index.tolist()
    top_items_deps = item_features[item_features['department'].isin(top_departments)]
    df = df[df['item_id'].isin(top_items_deps['item_id'].unique())]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    price_weights = df.groupby('item_id')['sales_value'].min() / df['sales_value'].max()
    least_expensive = price_weights[price_weights < 0.0005].index.tolist()
    df = df[~df['item_id'].isin(least_expensive)]

    # Уберем слишком дорогие товары
    top_expensive = price_weights[price_weights > 0.12].index.tolist()
    df = df[~df['item_id'].isin(top_expensive)]

    # Возьмем 5000 раиболее полулярных товаров
    popular_5000 = df.groupby('item_id')['quantity'].count().reset_index()
    top_5000 = popular_5000.sort_values(by='quantity', ascending=False)['item_id'].head(take_n_popular).tolist()
    df.loc[~df['item_id'].isin(top_5000), 'item_id'] = 999999

    # ...
    return df

