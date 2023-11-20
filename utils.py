import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import genetic_algo as GA

from simulated_annealing import SimulatedAnnealing
from grid_search import GridSearch
from pso import ParticleSwarmOptimization

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

import matplotlib.pyplot as plt

values = [
    "Genetic Algorithm",
    "Simulated Annealing",
    "Particle Swarm Optimization",
    "Grid Search",
]


def get_data(stock):
    filename = stock + ".csv"
    if os.path.exists(filename):
        return pd.read_csv(filename, parse_dates=True, index_col="Date")
    else:
        df = yf.download(stock)[["Close"]]
        df.to_csv(filename)
        return df


def convertStringtoDate(s):
    l = s.split("-")
    y, m, d = int(l[0]), int(l[1]), int(l[2])
    return datetime(year=y, month=m, day=d)


def getNextDateFromData(df, curr_date):
    ind = df.index.get_loc(curr_date)
    return df.index[ind + 1]


def createWindowedDataframe(df, start, end, w):
    start_date = convertStringtoDate(start)
    end_date = convertStringtoDate(end)
    curr_date = start_date
    reached_end = False

    dates = []
    features, target = [], []

    while not reached_end:
        df_window = df.loc[:curr_date].tail(w + 1)

        if len(df_window) != w + 1:
            raise Exception(f"Window size too large for date {curr_date}")

        closing_values = df_window["Close"].to_numpy()
        X, y = closing_values[:-1], closing_values[-1]

        dates.append(curr_date)
        features.append(X)
        target.append(y)

        if curr_date == end_date:
            reached_end = True
        else:
            curr_date = getNextDateFromData(df, curr_date)

    windowed_dataframe = pd.DataFrame({})
    windowed_dataframe["Target Date"] = dates
    features = np.array(features)

    for i in range(w):
        windowed_dataframe[f"Target-{w-i}"] = features[:, i]

    windowed_dataframe["Target"] = target
    return windowed_dataframe


def train_val_test_split(windowed_dataframe, train_ratio, test_ratio):
    df = windowed_dataframe[:]
    dates = df.pop("Target Date")
    df = df.to_numpy()
    features = df[:, :-1]
    features = features.reshape((len(df), features.shape[1], 1)).astype(np.float32)
    target = df[:, -1].astype(np.float32)

    train_split_point = int(len(df) * train_ratio)
    test_split_point = int(len(df) * (1 - test_ratio))

    train_dates = dates[:train_split_point]
    train_X = features[:train_split_point]
    train_y = target[:train_split_point]

    val_dates = dates[train_split_point:test_split_point]
    val_X = features[train_split_point:test_split_point]
    val_y = target[train_split_point:test_split_point]

    test_dates = dates[test_split_point:]
    test_X = features[test_split_point:]
    test_y = target[test_split_point:]

    return (
        train_dates,
        train_X,
        train_y,
        val_dates,
        val_X,
        val_y,
        test_dates,
        test_X,
        test_y,
    )


def train(parameters, train_X, train_y, epochs=150, verbose=0):
    model = Sequential(
        [
            layers.Input((parameters["window_len"], 1)),
            layers.LSTM(parameters["hidden_units_1"], return_sequences=True),
            layers.LSTM(parameters["hidden_units_2"], activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
    model.fit(train_X, train_y, epochs=epochs, verbose=verbose)
    return model


def plot(dates, observed, predicted, label):
    plt.figure(figsize=(12, 7))
    plt.plot(dates, predicted)
    plt.plot(dates, observed)
    plt.grid(True)

    plt.legend([label + " Predictions", label + " Observations"])
    plt.show()


def decode_parameters(s):
    parameters = {}
    parameters["window_len"] = int(s[0:3], 2)
    parameters["hidden_units_1"] = int(s[3:11], 2)
    parameters["hidden_units_2"] = int(s[11:18], 2)
    return parameters


def run_ga(stock: str):
    df = get_data(stock)
    windowed_dataframe = createWindowedDataframe(df, "2021-08-20", "2023-10-05", w=3)

    ga_fitness_values = {}

    def ga_fitness(x: GA.Individual):
        if x.gene in ga_fitness_values:
            return ga_fitness_values[x.gene]

        parameters = decode_parameters(x.gene)

        if (
            parameters["window_len"] == 0
            or parameters["hidden_units_1"] == 0
            or parameters["hidden_units_2"] == 0
        ):
            return float("inf")

        print(parameters)
        windowed_dataframe = createWindowedDataframe(
            df, "2021-08-20", "2023-10-05", w=parameters["window_len"]
        )
        (
            train_dates,
            train_X,
            train_y,
            val_dates,
            val_X,
            val_y,
            test_dates,
            test_X,
            test_y,
        ) = train_val_test_split(windowed_dataframe, 0.8, 0.1)
        model = train(parameters, train_X=train_X, train_y=train_y)
        ga_fitness_values[x.gene] = model.evaluate(val_X, val_y)
        return ga_fitness_values[x.gene]

    ga = GA.GeneticAlgorithm(3, 20, 18, 0.1, ga_fitness)
    ind = ga.run()
    best_param = decode_parameters(ind.gene)
    print("Best params :")
    print(best_param)
    print("Val loss 1 :", ga_fitness_values[ind.gene])

    windowed_dataframe = createWindowedDataframe(
        df, "2021-08-20", "2023-10-05", w=best_param["window_len"]
    )
    (
        train_dates,
        train_X,
        train_y,
        val_dates,
        val_X,
        val_y,
        test_dates,
        test_X,
        test_y,
    ) = train_val_test_split(windowed_dataframe, 0.8, 0.1)
    best_model = train(
        best_param, train_X=train_X, train_y=train_y, verbose=0, epochs=200
    )
    best_model.evaluate(val_X, val_y)
    best_model.evaluate(test_X, test_y)
    test_predictions = best_model.predict(test_X)
    plot(dates=test_dates, observed=test_y, predicted=test_predictions, label="Test")
    return best_param


def run_sa(stock: str):
    df = get_data(stock)
    sa_fitness_values = {}

    def sa_fitness(state):
        if state in sa_fitness_values:
            return sa_fitness_values[state]
        parameters = {
            "window_len": state[0],
            "hidden_units_1": state[1],
            "hidden_units_2": state[2],
        }

        print(parameters)
        windowed_dataframe = createWindowedDataframe(
            df, "2021-08-20", "2023-10-05", w=parameters["window_len"]
        )
        (
            train_dates,
            train_X,
            train_y,
            val_dates,
            val_X,
            val_y,
            test_dates,
            test_X,
            test_y,
        ) = train_val_test_split(windowed_dataframe, 0.8, 0.1)
        model = train(parameters, train_X=train_X, train_y=train_y)
        sa_fitness_values[state] = -model.evaluate(val_X, val_y)
        return sa_fitness_values[state]

    def sa_constraint(_state):
        return True

    sa = SimulatedAnnealing(
        start_state=(
            np.random.randint(1, 10),
            np.random.randint(2, 255),
            np.random.randint(2, 128),
        ),
        T=10,
        Tmin=0.1,
        k=0.5,
        n=6,
        f=sa_fitness,
        constraint=sa_constraint,
        max_state=(1, 1, 1),
    )
    sa_params = sa.run()

    params = {
        "window_len": sa_params[0],
        "hidden_units_1": sa_params[1],
        "hidden_units_2": sa_params[2],
    }

    windowed_dataframe = createWindowedDataframe(
        df, "2021-08-20", "2023-10-05", w=params["window_len"]
    )
    (
        train_dates,
        train_X,
        train_y,
        val_dates,
        val_X,
        val_y,
        test_dates,
        test_X,
        test_y,
    ) = train_val_test_split(windowed_dataframe, 0.8, 0.1)
    best_model = train(params, train_X=train_X, train_y=train_y, verbose=0, epochs=200)
    best_model.evaluate(val_X, val_y)
    best_model.evaluate(test_X, test_y)
    test_predictions = best_model.predict(test_X)
    plot(dates=test_dates, observed=test_y, predicted=test_predictions, label="Test")
    return params


def run_grid_search(stock: str):
    df = get_data(stock)

    search_space = [
        [i for i in range(4, 7)],  # window size
        [2**i for i in range(5, 9)],  # hidden layer 1
        [2**i for i in range(3, 8)],  # hidden layer 2
    ]
    gs_fitness_values = {}

    def grid_search_fitness(state: (int, int, int)):
        if state in gs_fitness_values:
            return gs_fitness_values[state]
        parameters = {
            "window_len": state[0],
            "hidden_units_1": state[1],
            "hidden_units_2": state[2],
        }
        print(parameters)
        windowed_dataframe = createWindowedDataframe(
            df, "2021-08-20", "2023-10-05", w=parameters["window_len"]
        )
        (
            train_dates,
            train_X,
            train_y,
            val_dates,
            val_X,
            val_y,
            test_dates,
            test_X,
            test_y,
        ) = train_val_test_split(windowed_dataframe, 0.8, 0.1)
        model = train(parameters, train_X=train_X, train_y=train_y)

        gs_fitness_values[state] = -model.evaluate(val_X, val_y)
        return gs_fitness_values[state]

    gs = GridSearch(
        init_state=(1, 1, 1), f=grid_search_fitness, search_space=search_space
    )

    gs_params = gs.run()

    params = {
        "window_len": gs_params[0],
        "hidden_units_1": gs_params[1],
        "hidden_units_2": gs_params[2],
    }

    windowed_dataframe = createWindowedDataframe(
        df, "2021-08-20", "2023-10-05", w=params["window_len"]
    )
    (
        train_dates,
        train_X,
        train_y,
        val_dates,
        val_X,
        val_y,
        test_dates,
        test_X,
        test_y,
    ) = train_val_test_split(windowed_dataframe, 0.8, 0.1)
    best_model = train(params, train_X=train_X, train_y=train_y, verbose=0, epochs=200)
    best_model.evaluate(val_X, val_y)
    best_model.evaluate(test_X, test_y)
    test_predictions = best_model.predict(test_X)
    plot(dates=test_dates, observed=test_y, predicted=test_predictions, label="Test")
    return params


def run_pso(stock: str):
    pso_fitness_values = {}
    df = get_data(stock)

    def pso_fitness(x):
        key = tuple(x.tolist())

        if key in pso_fitness_values:
            return pso_fitness_values[key]

        parameters = {
            "window_len": x[0],
            "hidden_units_1": x[1],
            "hidden_units_2": x[2],
        }
        print(parameters)
        windowed_dataframe = createWindowedDataframe(
            df, "2021-08-20", "2023-10-05", w=parameters["window_len"]
        )
        (
            train_dates,
            train_X,
            train_y,
            val_dates,
            val_X,
            val_y,
            test_dates,
            test_X,
            test_y,
        ) = train_val_test_split(windowed_dataframe, 0.8, 0.1)
        model = train(parameters, train_X=train_X, train_y=train_y)
        pso_fitness_values[key] = model.evaluate(val_X, val_y)
        return pso_fitness_values[key]

    pso = ParticleSwarmOptimization(n=5, c1=0.5, c2=0.5, fitness=pso_fitness)

    pso_params = pso.run(epochs=20)

    pso_params = {
        "window_len": pso_params[0],
        "hidden_units_1": pso_params[1],
        "hidden_units_2": pso_params[2],
    }

    windowed_dataframe = createWindowedDataframe(
        df, "2021-08-20", "2023-10-05", w=pso_params["window_len"]
    )
    (
        train_dates,
        train_X,
        train_y,
        val_dates,
        val_X,
        val_y,
        test_dates,
        test_X,
        test_y,
    ) = train_val_test_split(windowed_dataframe, 0.8, 0.1)
    best_model = train(
        pso_params, train_X=train_X, train_y=train_y, verbose=0, epochs=200
    )
    best_model.evaluate(val_X, val_y)
    best_model.evaluate(test_X, test_y)
    test_predictions = best_model.predict(test_X)
    plot(dates=test_dates, observed=test_y, predicted=test_predictions, label="Test")
    return pso_params


def run_algo(stock: str, algo: str):
    if algo == values[0]:
        print("Running GA")
        return run_ga(stock)
    elif algo == values[1]:
        print("Running SA")
        return run_sa(stock)
    elif algo == values[2]:
        print("Running Particle Swarm Optimization")
        return run_pso(stock)
    elif algo == values[3]:
        print("Running Grid Search")
        return run_grid_search(stock)
