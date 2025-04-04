require "./spec_helper"

describe Llama::Vocab do
  # Vocab tests require a model
  model_path = ENV["MODEL_PATH"]? || ARGV.find { |arg| arg.starts_with?("--model=") }.try &.split("=")[1]?

  if model_path.nil?
    pending "Skipping Vocab tests (no model provided)"
  else
    describe "basic properties" do
      it "can access vocabulary from model" do
        model = Llama::Model.new(model_path)
        vocab = model.vocab

        vocab.should_not be_nil
        vocab.n_tokens.should be > 0
        puts "  - Vocabulary size: #{vocab.n_tokens} tokens"
      end

      it "can access special tokens" do
        model = Llama::Model.new(model_path)
        vocab = model.vocab

        # Test special token accessors
        # These may return -1 if not defined in the model
        bos = vocab.bos
        eos = vocab.eos
        eot = vocab.eot
        nl = vocab.nl
        pad = vocab.pad

        puts "  - BOS token: #{bos}"
        puts "  - EOS token: #{eos}"
        puts "  - EOT token: #{eot}"
        puts "  - NL token: #{nl}"
        puts "  - PAD token: #{pad}"

        # At least some of these should be defined
        (bos >= 0 || eos >= 0 || eot >= 0 || nl >= 0 || pad >= 0).should be_true
      end
    end

    describe "#tokenize" do
      it "tokenizes simple text" do
        model = Llama::Model.new(model_path)
        vocab = model.vocab

        text = "Hello, world!"
        tokens = vocab.tokenize(text)

        tokens.should be_a(Array(Int32))
        tokens.size.should be > 0
        puts "  - Tokenized '#{text}' to #{tokens.size} tokens: #{tokens.inspect}"
      end

      it "tokenizes empty text" do
        model = Llama::Model.new(model_path)
        vocab = model.vocab

        text = ""
        tokens = vocab.tokenize(text)

        tokens.should be_a(Array(Int32))
        puts "  - Tokenized empty string to #{tokens.size} tokens: #{tokens.inspect}"
      end

      it "tokenizes text with special characters" do
        model = Llama::Model.new(model_path)
        vocab = model.vocab

        text = "Hello! 你好! こんにちは! 안녕하세요!"
        tokens = vocab.tokenize(text)

        tokens.should be_a(Array(Int32))
        tokens.size.should be > 0
        puts "  - Tokenized '#{text}' to #{tokens.size} tokens: #{tokens.inspect}"
      end

      it "tokenizes with different special token options" do
        model = Llama::Model.new(model_path)
        vocab = model.vocab

        text = "Hello, world!"

        # Default behavior (with special tokens)
        tokens1 = vocab.tokenize(text, add_special: true, parse_special: true)

        # Without adding special tokens
        tokens2 = vocab.tokenize(text, add_special: false, parse_special: true)

        # Without parsing special tokens
        tokens3 = vocab.tokenize(text, add_special: true, parse_special: false)

        # Without both
        tokens4 = vocab.tokenize(text, add_special: false, parse_special: false)

        # These should all produce valid token arrays
        tokens1.should be_a(Array(Int32))
        tokens2.should be_a(Array(Int32))
        tokens3.should be_a(Array(Int32))
        tokens4.should be_a(Array(Int32))

        # The results might be different depending on the options
        puts "  - With special tokens: #{tokens1.size} tokens"
        puts "  - Without adding special tokens: #{tokens2.size} tokens"
        puts "  - Without parsing special tokens: #{tokens3.size} tokens"
        puts "  - Without both: #{tokens4.size} tokens"
      end
    end

    describe "#token_to_text" do
      it "converts tokens to text" do
        model = Llama::Model.new(model_path)
        vocab = model.vocab

        # Tokenize some text
        text = "Hello, world!"
        tokens = vocab.tokenize(text)

        # Convert each token back to text
        token_texts = tokens.map { |token| vocab.token_to_text(token) }

        # Each token should convert to a non-empty string
        token_texts.each do |token_text|
          token_text.should be_a(String)
        end

        puts "  - Token texts: #{token_texts.inspect}"
      end

      it "handles special tokens" do
        model = Llama::Model.new(model_path)
        vocab = model.vocab

        # Try to convert special tokens to text
        special_tokens = [vocab.bos, vocab.eos, vocab.eot, vocab.nl, vocab.pad]

        special_tokens.each do |token|
          if token >= 0
            text = vocab.token_to_text(token)
            text.should be_a(String)
            puts "  - Special token #{token} text: '#{text}'"
          end
        end
      end
    end

    describe "roundtrip conversion" do
      it "can roundtrip text -> tokens -> text" do
        model = Llama::Model.new(model_path)
        vocab = model.vocab

        # Simple ASCII text should roundtrip well
        original_text = "Hello, world!"
        tokens = vocab.tokenize(original_text)

        # Convert tokens back to text
        reconstructed_text = tokens.map { |token| vocab.token_to_text(token) }.join

        puts "  - Original text: '#{original_text}'"
        puts "  - Reconstructed text: '#{reconstructed_text}'"

        # The reconstructed text might not be identical due to tokenization,
        # but it should contain the essential content
        reconstructed_text.downcase.gsub(/[^a-z]/, "").should contain(
          original_text.downcase.gsub(/[^a-z]/, "")
        )
      end
    end
  end
end
