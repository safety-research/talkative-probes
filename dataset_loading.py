import random
import chess
# Specific setting libraries (add as needed)
try:
    import chess.pgn
    import chess
    print("Chess library loaded.")
except ImportError:
    print("Chess library not found. Install with 'pip install python-chess'")

try:
    from datasets import load_dataset
    print("Hugging Face datasets library loaded.")
except ImportError:
    print("datasets library not found. Install with 'pip install datasets'")


def load_dataset(SETTING, tokenizer, NUM_DATA, IS_INSTRUCTION_MODEL, MAX_NEW_TOKENS):
    # Use a dictionary to store dataset items
    dataset = {
        'original_contexts': [],
        'questions': [],
        'ground_truths': [], # Store the expected correct answer (e.g., token ID, string label)
        'ground_truth_token_ids': [], # Store the token ID of the correct answer word/phrase
        'extra_info': [] # Store the square name of the correct answer
    }

    # Helper function to format questions based on model type
    def format_question(question_text, is_instruction_model=IS_INSTRUCTION_MODEL):
        """Format questions differently for base vs instruction models."""
        if is_instruction_model:
            # For instruction models, use direct questioning
            return question_text
        else:
            # For base models, prime with a completion pattern
            # Add text that encourages the model to begin answering
            if SETTING == 'chess':
                return f"{question_text} It is a"
            else:
                return f"{question_text} The answer is"
    if SETTING == 'chess':
        print("Preparing Chess PGN data...")
        try:
            import io

            # Define multiple PGN strings
            # pgn_strings = [
            #     """
            #     1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
            #     """,
            #     """
            #     1. d4 d5 2. c4 e6 3. Nc3 Nf6 *
            #     """,
            #     """
            #     1. e4 c5 2. Nf3 d6 3. d4 cxd4 *
            #     """,
            #     """
            #     1. e4 e6 2. d4 d5 3. Nc3 Nf6 *
            #     """,
            #     """
            #     1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 *
            #     """,
            #     """
            #     1. e4 c6 2. d4 d5 3. Nc3 dxe4 *
            #     """,
            #     """
            #     1. d4 d5 2. Nf3 Nf6 3. e3 e6 *
            #     """,
            #     """
            #     1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *
            #     """,
            #     """
            #     1. d4 Nf6 2. c4 e6 3. Nf3 d5 *
            #     """,
            #     """
            #     1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 *
            #     """
            # ]
            import pandas as pd

            df = pd.read_csv('/workspace/kitf/data/kaggle_chessgames.csv')
            if 'Moves' not in df.columns:
                raise ValueError("CSV does not contain a 'moves' column")

            pgn_strings = df['Moves'].dropna().tolist()


            num_pgns = min(NUM_DATA["NUM_GAMES"], len(pgn_strings))
            samples_per_pgn = NUM_DATA["NUM_SAMPLES_PER_GAME"] 
            #

            total_loaded = 0
            for idx, pgn_string in enumerate(pgn_strings[:num_pgns]):
                n_samples = samples_per_pgn
                print(f"Processing PGN {idx+1} of {num_pgns}, {n_samples} samples")
                pgn_file = io.StringIO(pgn_string)
                game = chess.pgn.read_game(pgn_file)
                board = game.board()

                moved_squares = set()
                node = game
                move_count = 0
                # Play up to NUM_MOVES moves, track moved piece squares simultaneously
                while node.variations and move_count < NUM_DATA["NUM_MOVES"]:
                    next_node = node.variations[0]
                    move = next_node.move
                    board.push(move)
                    moved_squares.add(move.to_square)
                    node = next_node
                    move_count += 1

                # Reconstruct PGN string up to capped move count
                capped_pgn_moves = []
                node = game
                move_counter = 0
                while node.variations and move_counter < move_count:
                    next_node = node.variations[0]
                    move_san = node.board().san(next_node.move)
                    capped_pgn_moves.append(move_san)
                    node = next_node
                    move_counter += 1

                # Compose PGN string fragment
                capped_pgn_str = ""
                for idx_move, move_san in enumerate(capped_pgn_moves):
                    if idx_move % 2 == 0:
                        capped_pgn_str += f"{(idx_move // 2) +1}. {move_san} "
                    else:
                        capped_pgn_str += f"{move_san} "
                #capped_pgn_str = capped_pgn_str.strip() + " *"
                capped_pgn_str = capped_pgn_str.strip() + f" {idx_move//2+2}."

                count = 0
                while count < n_samples:
                    # Only consider occupied squares where the piece has moved during the game
                    candidate_squares = [sq for sq in moved_squares if board.piece_at(sq)]
                    if not candidate_squares:
                        break

                    target_square = random.choice(candidate_squares)
                    square_name = chess.square_name(target_square)
                    question_text = f"What piece is on {square_name}?"

                    piece = board.piece_at(target_square)
                    piece_name = chess.piece_name(piece.piece_type).capitalize()
                    correct_answer_str = f" {piece_name}".lower()

                    # Use the PGN string up to capped move count as context
                    dataset['original_contexts'].append(capped_pgn_str.strip())
                    dataset['questions'].append(format_question(question_text))
                    dataset['ground_truths'].append(correct_answer_str)
                    dataset['extra_info'].append(square_name)

                    gt_token_id = tokenizer.encode(correct_answer_str, add_special_tokens=False)
                    if len(gt_token_id) == 1:
                        dataset['ground_truth_token_ids'].append(gt_token_id[0])
                    else:
                        dataset['ground_truth_token_ids'].append(gt_token_id)
                        #if MAX_NEW_TOKENS == 1:
                        #    dataset['original_contexts'].pop()
                        #    dataset['questions'].pop()
                        #    dataset['ground_truths'].pop()
                        #    dataset['ground_truth_token_ids'].pop()
                        #    continue

                    count += 1
                    total_loaded += 1

                pgn_file.close()

            print(f"Loaded {total_loaded} chess positions and questions equally from {num_pgns} PGNs, up to a maximum of {NUM_DATA['NUM_MOVES']} moves per game.")

        except Exception as e:
            print(f"Error processing PGN data: {e}")

    elif SETTING == 'nlp_sentiment':
        print("Preparing NLP Sentiment data (e.g., SST-2)...")
        try:
            hf_dataset = load_dataset("sst2", split='validation') # Use validation set
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(min(NUM_DATA["NUM_SAMPLES"], len(hf_dataset))))

            sentiment_map = {0: " Negative", 1: " Positive"} # Add space for tokenization

            for example in hf_dataset:
                dataset['original_contexts'].append(example['sentence'])
                # Apply formatting based on model type
                dataset['questions'].append(format_question("What is the sentiment?"))
                gt_label = example['label']
                gt_text = sentiment_map[gt_label]
                dataset['ground_truths'].append(gt_text)
                # Get token IDs for the ground truth
                gt_token_id = tokenizer.encode(gt_text, add_special_tokens=False)
                if len(gt_token_id) == 1 or MAX_NEW_TOKENS > 1:
                    dataset['ground_truth_token_ids'].append(gt_token_id)
                else:
                    print(f"Warning: Sentiment label '{gt_text}' is multi-token. Check tokenization.")
                    # Decide how to handle: skip, use first token, etc. Skipping for now.
                    dataset['original_contexts'].pop()
                    dataset['questions'].pop()
                    dataset['ground_truths'].pop()
                    continue

            print(f"Loaded {len(dataset['original_contexts'])} sentiment analysis samples.")

        except Exception as e:
            print(f"Error loading/processing dataset: {e}")

    elif SETTING == 'nlp_truth':
        print("Preparing NLP Truthfulness data (e.g., BoolQ)...")
        # Example using BoolQ - requires careful question formulation
        try:
            hf_dataset = load_dataset("boolq", split='validation')
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(min(NUM_DATA["NUM_SAMPLES"], len(hf_dataset))))

            truth_map = {True: " Yes", False: " No"} # Add space

            for example in hf_dataset:
                # BoolQ context is usually long (passage). Use question as context?
                # Or use passage and ask 'Is the following statement true: [question]'?
                # Let's use the provided question as the primary 'context' for simplicity
                dataset['original_contexts'].append(example['question'])
                # Apply formatting based on model type
                dataset['questions'].append(format_question("Is this statement true or false?"))
                gt_label = example['answer']
                gt_text = truth_map[gt_label]
                dataset['ground_truths'].append(gt_text)

                gt_token_id = tokenizer.encode(gt_text, add_special_tokens=False)
                if len(gt_token_id) == 1 or MAX_NEW_TOKENS > 1:
                    dataset['ground_truth_token_ids'].append(gt_token_id)
                else:
                    print(f"Warning: Truth label '{gt_text}' is multi-token. Check tokenization.")
                    dataset['original_contexts'].pop()
                    dataset['questions'].pop()
                    dataset['ground_truths'].pop()
                    continue

            print(f"Loaded {len(dataset['original_contexts'])} truthfulness samples.")
        except Exception as e:
            print(f"Error loading/processing dataset: {e}")
    elif SETTING == 'secret':
        print("Preparing secret information probing data...")
        # Create examples where a specific piece of information is embedded in context
        # Each secret should be a single token for clear representation

        # Check which tokens are single tokens in our tokenizer
        test_tokens = [" dog", " triangle", " dragon", " chair", " square"]
        single_tokens = []
        for token in test_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                single_tokens.append((token, ids[0]))

        if not single_tokens:
            raise ValueError("No single-token secrets found. Check tokenization.")

        # Create contexts with the verified single tokens
        for token, token_id in single_tokens:
            # Strip the space for the context sentence
            word = token.strip()
            context = f"The secret object is a {word}. What is the secret object?"
            question = ": the secret object is a"

            dataset['original_contexts'].append(context)
            dataset['questions'].append(question)
            dataset['ground_truths'].append(token)  # Keep the space for proper tokenization
            dataset['ground_truth_token_ids'].append(token_id)

        print(f"Created {len(dataset['original_contexts'])} secret information examples with single-token answers.")
    else:
        raise ValueError(f"Unknown SETTING: {SETTING}")
    return dataset
