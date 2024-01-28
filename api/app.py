from flask import Flask, render_template, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained("//home//ayush//Desktop//why-darkpattern//pickles//models//nn/checkpoint-1885")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


@app.route('/', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            output = []
            data = request.get_json().get('tokens')

            if not isinstance(data, list):
                raise ValueError('Invalid data format. Expected a list of tokens.')

            for token in data:
                inputs = tokenizer(token, return_tensors="pt")
                outputs = model(**inputs)
                predictions = outputs.logits

                probabilities = torch.nn.functional.softmax(predictions, dim=1)

                predicted_class = torch.argmax(probabilities, dim=1).item()

                if predicted_class == 1:
                    output.append("Dark")
                else:
                    output.append("Not dark")

            dark = [data[i] for i in range(len(output)) if output[i] == 'Dark']
            for d in dark:
                print(d)
            print()
            print(len(dark))

            response_data = {'result': output}
            print(response_data)

            return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Bad Request

if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         output = []
#         data = request.get_json().get('tokens')
        
#         for token in data:
#             inputs = tokenizer(token, return_tensors="pt")
#             outputs = model(**inputs)
#             predictions = outputs.logits

#             probabilities = torch.nn.functional.softmax(predictions, dim=1)

#             predicted_class = torch.argmax(probabilities, dim=1).item()
            
#             if predicted_class == 1:
#                 output.append("Dark")
#             else:
#                 output.append("Not dark")

#         dark = [data[i] for i in range(len(output)) if output[i] == 'Dark']
#         for d in dark:
#             print(d)
#         print()
#         print(len(dark))

#         response_data = {'result': output}
#         print(response_data)

#         return jsonify(response_data)

# if __name__ == '__main__':
#     app.run(debug=True)



