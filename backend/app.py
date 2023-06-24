# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pickle
from fastapi.middleware.cors import CORSMiddleware


# 2. Create the app object
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3001",
    "http://localhost:32827",
    # Add more allowed origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# pickels

pickle_in = open("logreg.pkl", "rb")
logreg = pickle.load(pickle_in)
pickle_in = open("vectorizer.pkl", "rb")
vectorizer = pickle.load(pickle_in)
# print(logreg)

# # 3. Index route, opens automatically on http://127.0.0.1:8000
# @app.get('/')
# def index():
#     return {'message': 'Hello, World'}

# # 4. Route with a single parameter, returns the parameter within a message
# #    Located at: http://127.0.0.1:8000/AnyNameHere
# @app.get('/{name}')
# def get_name(name: str):
#     return {'Welcome To Krish Youtube Channel': f'{name}'}

# # 3. Expose the prediction functionality, make a prediction from the passed
# #    JSON data and return the predicted Bank Note with the confidence


@app.post('/predict')
def predict(data: str):
    print(data)
    sentances = [data]
    v_sentances = vectorizer.transform(sentances)
    predicted_value = logreg.predict(v_sentances)
    # print(v_sentances, sentances, v_sentances.shape, predicted_value)
    print(type(predicted_value.item(0)))
    return predicted_value.item(0)
    # variance = data['variance']
    # skewness = data['skewness']
    # curtosis = data['curtosis']
    # entropy = data['entropy']
    # print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    prediction = logreg.predict([[variance, skewness, curtosis, entropy]])
    # if(prediction[0] > 0.5):
    #     prediction = "Fake note"
    # else:
    #     prediction = "Its a Bank note"
    # return {
    #     'prediction': prediction
    # }

    # # 5. Run the API with uvicorn
    # #    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# #uvicorn app:app --reload
