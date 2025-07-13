require "../src/llama"
require "../src/llama/chat"
require "kemal"
require "option_parser"
require "json"

# Command line arguments
model_path = ""
n_ctx = 2048
ngl = 99
port = 3000

OptionParser.parse do |parser|
  parser.banner = "Usage: #{PROGRAM_NAME} -m MODEL [-c context_size] [-ngl n_gpu_layers] [-p port]"

  parser.on("-m", "--model MODEL", "Path to the model file (required)") { |path| model_path = path }
  parser.on("-c", "--context N", "Context size (default: 2048)") { |n| n_ctx = n.to_i }
  parser.on("-g", "--gpu-layers N", "Number of layers to offload to GPU (default: 99)") { |n| ngl = n.to_i }
  parser.on("-p", "--port PORT", "Port to run the server on (default: 3000)") { |n| port = n.to_i }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end

abort "Error: Model path is required. Use -m or --model option.\nRun with --help for usage information." if model_path.empty?

Llama.log_level = Llama::LOG_LEVEL_ERROR

# Initialize model, vocab, context, and sampler
model = Llama::Model.new(model_path, n_gpu_layers: ngl) || abort "Error: Unable to load model"
vocab = model.vocab
context = model.context(n_ctx: n_ctx.to_u32, n_batch: n_ctx.to_u32) || abort "Error: Failed to create the context"

sampler = Llama::SamplerChain.new
sampler.add(Llama::Sampler::MinP.new(0.05, 1))
sampler.add(Llama::Sampler::Temp.new(0.8))
sampler.add(Llama::Sampler::Dist.new(Llama::DEFAULT_SEED))

tmpl = model.chat_template
if tmpl.nil?
  STDERR.puts "Warning: Model does not provide a chat template, using default"
  tmpl = ""
end

# Generate response as an array of words/tokens
def generate_words(context, vocab, sampler, prompt) : Array(String)
  response = ""
  words = [] of String
  is_first = true
  prompt_tokens = vocab.tokenize(prompt, add_special: is_first, parse_special: true)

  if prompt_tokens.empty?
    STDERR.puts "Failed to tokenize the prompt"
    return words
  end

  batch = Llama::Batch.get_one(prompt_tokens)
  context_limit_reached = false
  loop do
    n_ctx = context.n_ctx
    if batch.n_tokens > n_ctx
      context_limit_reached = true
      break
    end

    if context.decode(batch) != 0
      STDERR.puts "Failed to decode"
      break
    end

    new_token_id = sampler.sample(context)
    break if vocab.is_eog(new_token_id) || new_token_id == vocab.eos || new_token_id == vocab.eot

    piece = vocab.token_to_piece(new_token_id, 0, true)
    response += piece
    # Split by whitespace, punctuation, and newlines (supports English and Japanese)
    piece.split(/([。、！？\n\s]+)/).each do |w|
      words << w unless w.empty?
    end

    batch = Llama::Batch.get_one([new_token_id])
  end
  if context_limit_reached
    words << " [Context length limit reached!]"
  end
  words
end

# Serve the chat UI (SPA)
get "/" do |env|
  <<-HTML
  <!DOCTYPE html>
  <html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Llama.cr Chat Server</title>
    <style>
      body {
        font-family: 'Segoe UI', Arial, sans-serif;
        margin: 0;
        background: #ececec;
      }
      .chat-container {
        max-width: 480px;
        margin: 40px auto;
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 16px #0002;
        padding: 0;
        display: flex;
        flex-direction: column;
        height: 80vh;
      }
      .chat-header {
        padding: 1em;
        border-bottom: 1px solid #e0e0e0;
        font-size: 1.3em;
        font-weight: bold;
        background: #f7f7f7;
        border-radius: 12px 12px 0 0;
      }
      .chat-timeline {
        flex: 1;
        overflow-y: auto;
        padding: 1em;
        background: #f5f6fa;
        display: flex;
        flex-direction: column;
        gap: 0.7em;
      }
      .msg-row {
        display: flex;
        align-items: flex-end;
      }
      .msg-row.user {
        justify-content: flex-end;
      }
      .msg-row.assistant {
        justify-content: flex-start;
      }
      .msg-bubble {
        max-width: 70%;
        padding: 0.7em 1em;
        border-radius: 18px;
        font-size: 1em;
        line-height: 1.5;
        box-shadow: 0 1px 4px #0001;
        word-break: break-word;
        display: flex;
        flex-direction: column;
      }
      .msg-bubble.user {
        background: #d2eaff;
        color: #1a3557;
        border-bottom-right-radius: 4px;
        margin-left: 1em;
      }
      .msg-bubble.assistant {
        background: #f0f0f0;
        color: #222;
        border-bottom-left-radius: 4px;
        margin-right: 1em;
      }
      .msg-label {
        font-size: 0.8em;
        color: #888;
        margin-bottom: 0.2em;
      }
      .input-area {
        display: flex;
        border-top: 1px solid #e0e0e0;
        padding: 0.7em;
        background: #fafbfc;
        border-radius: 0 0 12px 12px;
      }
      .input-area input {
        flex: 1;
        font-size: 1em;
        padding: 0.5em;
        border-radius: 6px;
        border: 1px solid #ccc;
        outline: none;
        margin-right: 0.5em;
      }
      .input-area button {
        font-size: 1em;
        padding: 0.5em 1.2em;
        border-radius: 6px;
        border: none;
        background: #2196f3;
        color: #fff;
        cursor: pointer;
        transition: background 0.2s;
      }
      .input-area button:hover {
        background: #1769aa;
      }
      .words span {
        display: inline-block;
        margin-right: 2px;
      }
    </style>
  </head>
  <body>
    <div class="chat-container">
      <div class="chat-header">Llama.cr Chat</div>
      <div id="timeline" class="chat-timeline"></div>
      <form id="chat-form" class="input-area" autocomplete="off">
        <input type="text" id="user-input" placeholder="Type your message..." autofocus autocomplete="off"/>
        <button type="submit">Send</button>
      </form>
    </div>
    <script>
      const timeline = document.getElementById('timeline');
      const form = document.getElementById('chat-form');
      const input = document.getElementById('user-input');
      let history = [];

      // Render chat history
      function render() {
        timeline.innerHTML = '';
        history.forEach(msg => {
          const row = document.createElement('div');
          row.className = 'msg-row ' + msg.role;
          const bubble = document.createElement('div');
          bubble.className = 'msg-bubble ' + msg.role;
          const label = document.createElement('div');
          label.className = 'msg-label';
          label.textContent = msg.role === 'user' ? 'You' : 'Assistant';
          bubble.appendChild(label);
          if(msg.role === 'assistant') {
            const spanWrap = document.createElement('span');
            spanWrap.className = 'words';
            msg.words.forEach(word => {
              const s = document.createElement('span');
              s.textContent = word;
              spanWrap.appendChild(s);
            });
            bubble.appendChild(spanWrap);
          } else {
            bubble.appendChild(document.createTextNode(msg.content));
          }
          row.appendChild(bubble);
          timeline.appendChild(row);
        });
        timeline.scrollTop = timeline.scrollHeight;
      }

      // Send user message and animate assistant response
      async function sendMessage(text) {
        history.push({role: 'user', content: text});
        render();
        input.value = '';
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({history: history})
        });
        const data = await res.json();
        // Animate assistant response word by word
        let assistantMsg = {role: 'assistant', words: []};
        history.push(assistantMsg);
        render();
        for(let i=0; i<data.words.length; i++) {
          assistantMsg.words.push(data.words[i]);
          render();
          await new Promise(r => setTimeout(r, 60));
        }
      }

      form.onsubmit = e => {
        e.preventDefault();
        const text = input.value.trim();
        if(text) sendMessage(text);
      };
    </script>
  </body>
  </html>
  HTML
end

# Chat API endpoint
post "/api/chat" do |env|
  req = JSON.parse(env.request.body.not_nil!)
  messages = [] of Llama::ChatMessage
  if req["history"]?
    req["history"].as_a.each do |msg|
      if msg["role"].as_s == "user"
        messages << Llama::ChatMessage.new("user", msg["content"].as_s)
      elsif msg["role"].as_s == "assistant"
        # For assistant, join words array or fallback to content
        content = msg["words"]? ? msg["words"].as_a.map(&.as_s).join : (msg["content"]?.try(&.as_s) || "")
        messages << Llama::ChatMessage.new("assistant", content)
      end
    end
  end
  local_context = model.context(n_ctx: n_ctx.to_u32, n_batch: n_ctx.to_u32) || abort "Error: Failed to create the context"
  prompt = local_context.apply_chat_template(messages, true, tmpl)
  words = generate_words(local_context, vocab, sampler, prompt)
  {words: words}.to_json
end

Kemal.config.port = port
at_exit { Llama.uninit }
Kemal.run
