import numpy as np
from collections import defaultdict
from sksurv.metrics import concordance_index_ipcw, brier_score


class Evaluator:
    def __init__(self, train_data):
        self.train_data = train_data

    def evaluate(self, model, test_data, val_batch_size=None, confidence=None):
        metric_dict = defaultdict(list)
        times = model.config["duration_index"][1:-1]
        horizons = model.config["horizons"]
        get_target = lambda data: (
            data["duration"].values,
            data["event"].values,
        )

        train_duration, train_event = get_target(self.train_data)
        et_train = np.array(
            [(train_event[i], train_duration[i]) for i in range(len(train_event))],
            dtype=[("e", bool), ("t", float)],
        )

        surv = model.predict_surv(test_data[0], batch_size=val_batch_size)
        risk = 1 - surv
        test_duration, test_event = get_target(test_data[1])
        et_test = np.array(
            [(test_event[i], test_duration[i]) for i in range(len(test_event))],
            dtype=[("e", bool), ("t", float)],
        )

        brs = brier_score(
            et_train,
            et_test,
            surv.to("cpu").numpy()[:, 1:-1],
            times,
        )[1]
        cis = []

        for i, _ in enumerate(times):
            ci = concordance_index_ipcw(
                et_train,
                et_test,
                estimate=risk[:, i + 1].to("cpu").numpy(),
                tau=times[i],
            )[0]
            cis.append(ci)
            metric_dict[f"{horizons[i]}_ipcw"].append(ci)
            metric_dict[f"{horizons[i]}_brier"].append(brs[i])

        if confidence is not None:
            stats_dict = defaultdict(list)
            for i in range(10):
                sample_test_data = test_data[0].sample(
                    test_data[0].shape[0], replace=True
                )
                sample_y_test_data = test_data[1].loc[sample_test_data.index]
                sample_data = (sample_test_data, sample_y_test_data)
                res_dict = self.evaluate(model, sample_data, val_batch_size)
                for k in res_dict.keys():
                    stats_dict[k].append(res_dict[k])
            alpha = confidence
            p1 = ((1 - alpha) / 2) * 100
            p2 = (alpha + ((1.0 - alpha) / 2.0)) * 100
            
            for k in stats_dict.keys():
                stats = stats_dict[k]
                lower = max(0, np.percentile(stats, p1))
                upper = min(1.0, np.percentile(stats, p2))
                metric_dict[k] = [(upper + lower) / 2, (upper - lower) / 2]

        return metric_dict
