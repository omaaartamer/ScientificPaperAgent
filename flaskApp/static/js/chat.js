let isProcessing = false;

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

function setQuery(query) {
    const input = document.getElementById('user-input');
    input.value = query;
    autoResize(input);
    input.focus();
}

function addMessage(type, content) {
    const chatBox = document.getElementById('chat-box');
    const welcomeMessage = document.getElementById('welcome-message');
    
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }
    
    const div = document.createElement('div');
    div.className = `message ${type}`;
    
    if (type === 'bot') {
        div.innerHTML = `<pre class="code-block"><code>${content}</code></pre>`;
    } else {
        div.textContent = content;
    }
    
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addLoadingMessage() {
    const chatBox = document.getElementById('chat-box');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot loading';
    loadingDiv.innerHTML = `
        <div class="loading-content">
            <div class="loading-text">Processing request</div>
            <div class="loading-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return loadingDiv;
}

function removeLoadingMessage() {
    const loadingMessage = document.querySelector('.loading');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

async function sendMessage() {
    if (isProcessing) return;
    
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    if (!message) return;

    try {
        isProcessing = true;
        addMessage('user', message);
        input.value = '';
        input.style.height = 'auto';

        const loadingMessage = addLoadingMessage();

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: message})
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();
        removeLoadingMessage();
        
        if (data.status === 'error') {
            throw new Error(data.message);
        }

        addMessage('bot', data.response);

    } catch (error) {
        removeLoadingMessage();
        addMessage('bot', `Error: ${error.message}`);
    } finally {
        isProcessing = false;
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    const textarea = document.getElementById('user-input');
    textarea.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});
