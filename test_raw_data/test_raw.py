import pandas as pd


def test_column_names(data: pd.DataFrame):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert data.shape[1] == 16
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data: pd.DataFrame):

    known_names = ["Bronx", "Brooklyn", "Manhattan",
                   "Queens", "Staten Island"]

    neigh = data['neighbourhood_group'].unique()

    # Unordered check
    for n in neigh:
        assert n in known_names


def test_room_type(data: pd.DataFrame):

    room_types = ['Private room', 'Entire home/apt', 'Shared room']
    room = data['room_type'].unique()
    for r in room:
        assert r in room_types
