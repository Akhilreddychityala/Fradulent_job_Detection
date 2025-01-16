import pickle

# Load the model
with open(r'pickle\app_model.pkl', 'rb') as file:
    model = pickle.load(file)
    print(type(model))  # Check the type of the loaded model
    print(model)  # Print the model contents
