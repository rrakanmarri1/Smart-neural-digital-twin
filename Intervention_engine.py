def suggest_intervention(sensor_input):
    if sensor_input['pressure'] > 75:
        return "Shut down the pump"
    elif sensor_input['temperature'] > 90:
        return "Activate cooling system"
    elif sensor_input['flow_rate'] < 20:
        return "Check for pipeline blockage"
    return "Continue monitoring"
