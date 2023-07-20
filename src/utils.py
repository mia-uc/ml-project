class Pipeline:
    def __init__(self, *steps) -> None:
        self.steps = list(steps)

    def fit(self, X):
        for s in self.steps[:-1]:
            X = s.fit_transform(X)

        return
