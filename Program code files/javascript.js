const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

sendButton.addEventListener('click', () => {
    const userMessage = userInput.value;
    chatBox.innerHTML += `<p>User: ${userMessage}</p>`;
    
    // Send user message to the server for processing
    fetch('/chat', {
        method: 'POST',
        body: JSON.stringify({ userMessage }),
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        const chatbotMessage = data.chatbotMessage;
        chatBox.innerHTML += `<p>Chatbot: ${chatbotMessage}</p>`;
    });

    userInput.value = '';
});
