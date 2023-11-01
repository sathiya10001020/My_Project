from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Load your text dataset (assuming it's a txt file)
text_dataset_path = "C:\\Users\mathu\Desktop\ibm dataset.txt"
# Assuming your preprocessed data is in 'preprocessed_dataset.txt'
with open('pre.py', 'r', encoding='utf-8') as file:
    preprocessed_data = file.readlines()

# Now 'preprocessed_data' is a list where each element is a preprocessed document.


# Read the text dataset and create a dictionary
chatbot_responses = {}
with open(text_dataset_path, "r", encoding="utf-8") as file:
    for line in file:
        user_input, bot_response = map(str.strip, line.split('\t'))
        chatbot_responses[user_input.lower()] = bot_response

@app.route("/")
def home():
    return render_template("index.html")

@socketio.on("message")
def handle_message(msg):
    user_input = msg.lower()
    
    # Check if the user input matches any predefined responses
    if user_input in chatbot_responses:
        bot_response = chatbot_responses[user_input]
    else:
        # If no predefined response, provide a default response
        bot_response = "I'm not sure how to respond to that. Can you please rephrase?"

    emit("response", {"bot_response": bot_response})

if __name__ == "__main__":
    socketio.run(app, debug=True)
    
    
    from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('seq2seq_chatbot_model.h5')

def generate_response(user_input):
    # Tokenize the user input
    input_seq = input_tokenizer.texts_to_sequences([user_input])
    input_pad = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')

    # Use the model to predict the response
    predicted_output = model.predict([input_pad, input_pad])

    # Convert the predicted sequence to text
    predicted_sequence = np.argmax(predicted_output, axis=-1)
    predicted_text = target_tokenizer.sequences_to_texts(predicted_sequence)

    return predicted_text[0]
@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    bot_response = generate_response(user_input)
    return jsonify({"bot_response": bot_response})