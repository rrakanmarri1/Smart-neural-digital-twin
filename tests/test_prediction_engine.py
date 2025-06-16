import prediction_engine as pe


def test_prediction_flow():
    models = pe.create_prediction_model()
    preds = pe.predict_future_values(models, hours_ahead=2)
    assert preds
    first_sensor = list(preds.keys())[0]
    assert len(preds[first_sensor]) == 2
