def calc_mape(y_pred, y_true):
    return ((y_true - y_pred) / y_true).abs().mean() * 100
