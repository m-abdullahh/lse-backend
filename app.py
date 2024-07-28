from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from Models.Trademark_model import load_trademark_model, query_trademark_model
from Models.generic_search_model import load_generic_search_model,query_generic_search_model
from Models.Judgement_Classification_Trademark import load_judgement_classification_model,predict_judgement_classification
app = Flask(__name__)
CORS(app)


result = [{
"Facts":"Sdasdas"
,"Issues_framed":"Sdasdas"
,"Decisions_Holdings":"Sdasdas"
,"Reasoning_and_Analysis":"Sdasdas"
,"Title":"Sdasdas"
}]

# Calling Generic Search Model
# generic_model,generic_df,generic_required_columns = load_generic_search_model()

# Calling Trademark Search Model
tm_df, tm_embedder, tm_title_embeddings, tm_desc_embeddings = load_trademark_model()

#Calling Judgement Classification Model
classifierSVM, classifierRF, classifierXG,vectorizer,label_encoder = load_judgement_classification_model()

# Middleware to log requests
@app.before_request
def log_request_info():
    app.logger.info('Path: %s, Method: %s, Query: %s', request.path, request.method, request.args)

# Controller functions
@app.route('/cases', methods=['GET'])
def generic_search():
    req_data = request.args.to_dict()
    if 'text' in req_data:
        query = req_data['text']
        print("Query is:",query)
        # results = query_generic_search_model(generic_model, generic_df, generic_required_columns,query)
        # print("Results are:",jsonify(results))
        # return jsonify(results)

        return jsonify(result)
    return jsonify({"error": "No text query provided"}), 400

@app.route('/trademark', methods=['GET'])
def trademark_search():
    req_data = request.args.to_dict()
    if 'query' in req_data:
        query = req_data['query']
        query_type = req_data.get('type', 'text')

        query = int(query) if query_type == 'number' else query
        results = query_trademark_model(query, tm_df, tm_embedder, tm_title_embeddings, tm_desc_embeddings, query_type)
        print(type(results),results)
        return jsonify(results)
    return jsonify({"error": "No query provided"}), 400

@app.route('/judgementclassification', methods=['GET'])
def judgement_classification():
    req_data = request.args.to_dict()
    if 'query' in req_data:
        query = req_data['query']
        print("Query is:",query)
        result = predict_judgement_classification(query,"svm",classifierSVM, classifierRF, classifierXG,vectorizer,label_encoder)
        print("Results are:",jsonify(result))
        # return jsonify(results)

        return jsonify(result)
    return jsonify({"error": "No text query provided"}), 400

if __name__ == '__main__':
    app.run(port=3000, debug=True)
