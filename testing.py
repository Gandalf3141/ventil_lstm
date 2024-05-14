from ray import train, tune


def objective(config):  # ①
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


search_space = {  # ②
    "a": tune.choice([0.001, 0.01, 0.1, 1.0]),
    
    "b": tune.choice([1, 2, 3]),
}

tuner = tune.Tuner(objective, param_space=search_space, tune_config=tune.TuneConfig(num_samples=10))  # ③

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)
#added a comment