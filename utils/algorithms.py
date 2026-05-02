ALGORITHMS = {
    "Existing": "Existing Random Forest	",
    "Proposed": "Proposed CNN + BI-LSTM		",
    "Extension": "Extension CNN + BI-LSTM + BI-GRU	",
    
}
def get_algo(name):
    return ALGORITHMS.get(name)
