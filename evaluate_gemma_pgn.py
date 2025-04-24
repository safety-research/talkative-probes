import asyncio
import re
import statistics
import dataclasses
import logging
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import json
import pathlib
import os
import traceback
import random
import io
from pathlib import Path

import chess
import chess.pgn
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from safetytooling.utils.experiment_utils import ExperimentConfigBase
from simple_parsing import ArgumentParser

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = pathlib.Path("pgn_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Config definitions
# =============================================================================

@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    """Configuration for the PGN evaluation experiment."""
    model: str = "google/gemma-2-9b-it"
    models: List[str] = dataclasses.field(default_factory=list)
    samples: int = 20
    moves_prompt: int = 40
    prefix_moves: int = 0
    force: bool = False
    nopromptcache: bool = False
    save: str = None
    max_workers: int = 10

# =============================================================================
# Prompt helpers
# =============================================================================

SYSTEM_PROMPT = (
    "You are an expert chess engine. Respond ONLY with a sequence of chess moves in PGN SAN notation. "
    "Do NOT add any commentary, punctuation, newlines, or results. For example: '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6'."
)

USER_PROMPT_WITH_PREFIX = (
    "Continue the following chess game with {n_moves} additional moves. Here are the moves so far:\n"
    "{prefix_moves}\n"
    "Continue with move {next_move_num}."
)

USER_PROMPT = (
    "Generate the first {n_moves} moves of a sensible chess game starting from the standard initial position. "
    "Use PGN SAN notation including move numbers."
)

# =============================================================================
# PGN parsing / horizon calculation
# =============================================================================

MOVE_NUM_RE = re.compile(r"^[0-9]+\.(\.\.)?")
RESULT_RE = re.compile(r"^(1-0|0-1|1/2-1/2|\*)$")


def tokenise_pgn(text: str) -> List[str]:
    """Split model output into SAN tokens, stripping move numbers and result."""
    raw_tokens = re.split(r"\s+", text.strip())
    tokens: List[str] = []
    for raw_tok in raw_tokens:
        tok = raw_tok.strip()
        if not tok:
            continue
        # Skip move numbers like '1.' or '1...' BEFORE stripping punctuation
        if MOVE_NUM_RE.match(tok):
            continue
        # Stop at result token
        if RESULT_RE.match(tok):
            break
        # Remove trailing punctuation (only if it's not a move number or result)
        tok = tok.rstrip('.')
        tokens.append(tok)
    return tokens


def horizon_valid_moves(tokens: List[str]) -> Tuple[int, str]:
    """Return number of consecutive legal SAN moves starting from initial board and first invalid move."""
    board = chess.Board()
    count = 0
    invalid_move = ""
    
    # If we have no tokens, it's not an error, just no moves
    if not tokens:
        return count, "NO_MOVES_GENERATED"
    
    for i, san in enumerate(tokens):
        try:
            move = board.parse_san(san)
        except ValueError as e:
            invalid_move = f"'{san}' (token #{i+1}, after {count} valid moves): {str(e)}"
            break
        board.push(move)
        count += 1
    
    # If we processed all tokens without error, but didn't reach game end, note it
    if count == len(tokens) and not invalid_move:
        invalid_move = "OUTPUT_TRUNCATED"
    
    return count, invalid_move


def final_fen(tokens: List[str]) -> str:
    """Return the final FEN position after applying all valid moves."""
    board = chess.Board()
    for san in tokens:
        try:
            move = board.parse_san(san)
        except ValueError:
            break
        board.push(move)
    return board.fen()


def get_prefix_moves(n_prefix_moves: int, model_id: str = "", seed: int = 42) -> Tuple[str, int]:
    """Get prefix moves from a randomly selected game in the dataset."""
    if n_prefix_moves <= 0:
        return "", 1
    
    # Create a deterministic seed based on model_id and n_prefix_moves
    if model_id:
        # Create a hash from the model_id string to use as part of the seed
        model_hash = sum(ord(c) for c in model_id)
        local_seed = seed + model_hash + n_prefix_moves
    else:
        local_seed = seed
    
    # Set the seed for deterministic sampling
    random.seed(local_seed)
    
    try:
        # Try to load chess games dataset (from Kaggle or local file)
        df_path = Path('kaggle_chessgames.csv')
        if not df_path.exists():
            # Check a few common locations
            alt_paths = [
                Path('/workspace/kitf/data/kaggle_chessgames.csv'),
                Path('/home/kitf/data/kaggle_chessgames.csv'),
                Path('/data/kaggle_chessgames.csv')
            ]
            for p in alt_paths:
                if p.exists():
                    df_path = p
                    break
        
        if df_path.exists():
            df = pd.read_csv(df_path)
            if 'Moves' not in df.columns:
                print("CSV found but doesn't contain a 'Moves' column")
                return "", 1
                
            # Select a random game
            pgn_string = random.choice(df['Moves'].dropna().tolist())
        else:
            # Fallback to some hardcoded openings if dataset is not available
            pgn_strings = [
                "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6",  # Ruy Lopez
                "1. d4 Nf6 2. c4 e6 3. Nf3 d5 4. Nc3 Be7",   # Queen's Gambit
                "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6",  # Sicilian
                "1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5",      # French
                "1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 dxc4",   # Slav
            ]
            pgn_string = random.choice(pgn_strings)
        
        # Parse the selected game
        pgn_file = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_file)
        
        # Extract the first n_prefix_moves move
        prefix_moves = []
        move_num = 1
        node = game
        while node.variations and len(prefix_moves) < n_prefix_moves*2:  # *2 because each move number has 2 half-moves
            next_node = node.variations[0]
            move_san = node.board().san(next_node.move)
            prefix_moves.append(move_san)
            node = next_node
        
        # Format the prefix moves into PGN format
        formatted_prefix = ""
        for idx, move_san in enumerate(prefix_moves):
            if idx % 2 == 0:
                formatted_prefix += f"{(idx // 2) + 1}. {move_san} "
            else:
                formatted_prefix += f"{move_san} "
        
        # Calculate the next move number to start from
        next_move_num = (len(prefix_moves) // 2) + 1
        
        return formatted_prefix.strip(), next_move_num
    
    except Exception as e:
        print(f"Error getting prefix moves: {e}")
        return "", 1


# =============================================================================
# Evaluation core
# =============================================================================

async def generate_moves(
    api: InferenceAPI, 
    model_id: str, 
    n_moves_prompt: int, 
    prefix_moves: str = "", 
    next_move_num: int = 1, 
    use_cache: bool = True, 
    semaphore: asyncio.Semaphore = None,
    **kwargs
) -> str:
    """Generate a sequence of chess moves using the specified model."""
    async with semaphore:
        try:
            if prefix_moves:
                prompt_content = USER_PROMPT_WITH_PREFIX.format(
                    n_moves=n_moves_prompt, 
                    prefix_moves=prefix_moves,
                    next_move_num=next_move_num
                )
            else:
                prompt_content = USER_PROMPT.format(n_moves=n_moves_prompt)
                
            prompt = Prompt(
                messages=[
                    ChatMessage(role=MessageRole.system, content=SYSTEM_PROMPT),
                    ChatMessage(role=MessageRole.user, content=prompt_content),
                ]
            )
            
            # Set force_provider based on model name prefix
            api_kwargs = {}
            if model_id.startswith("together/"):
                api_kwargs["force_provider"] = "together"
            elif model_id.startswith("openrouter/"):
                api_kwargs["force_provider"] = "openrouter"
            
            responses = await api(
                model_id=model_id.replace("together/", "").replace("openrouter/", ""),  # Remove provider prefix
                prompt=prompt,
                max_tokens=512,  # Sufficient for ~60 moves
                temperature=0.7,
                top_p=1.0,
                use_cache=use_cache,
                **api_kwargs,
                **kwargs,
            )
            # Return first completion string
            return responses[0].completion.strip()
        except Exception as e:
            LOGGER.error(f"Error generating moves for {model_id}: {e}")
            traceback.print_exc()
            return "Error: Failed to generate moves"


async def evaluate_model(
    cfg: ExperimentConfig,
    model_id: str,
    num_samples: int = 50,
    n_moves_prompt: int = 40,
    n_prefix_moves: int = 0,
    use_cache: bool = True,
) -> Tuple[List[int], List[str]]:
    """Evaluate a model by generating multiple game samples and calculating move validity horizons."""
    api = cfg.api

    horizons: List[int] = []
    raw_outputs: List[str] = []
    prefix_data: List[Tuple[str, int]] = []
    
    # Get prefix moves for each sample if requested - now deterministic based on model_id
    if n_prefix_moves > 0:
        for i in range(num_samples):
            # Add sample index to ensure different prefixes for each sample
            prefix, next_move = get_prefix_moves(n_prefix_moves, model_id, seed=42+i)
            prefix_data.append((prefix, next_move))
    else:
        prefix_data = [("", 1) for _ in range(num_samples)]

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(cfg.max_workers)

    # Generate samples concurrently
    tasks = [
        generate_moves(
            api, 
            model_id, 
            n_moves_prompt, 
            prefix, 
            next_move, 
            use_cache=use_cache,
            semaphore=semaphore
        )
        for (prefix, next_move) in prefix_data
    ]
    completions = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {model_id}")

    # Process results
    for i, text in enumerate(completions):
        # Store the prefix with the output for complete game record
        if prefix_data[i][0]:
            full_output = f"{prefix_data[i][0]} {text}"
        else:
            full_output = text
            
        raw_outputs.append(full_output)
        
        if text.startswith("Error:"):
            horizons.append(0)  # Assign 0 horizon if generation failed
            continue
            
        tokens = tokenise_pgn(text)
        horizon, _ = horizon_valid_moves(tokens)
        horizons.append(horizon)

    return horizons, raw_outputs


async def load_or_generate(
    cfg: ExperimentConfig,
    model_id: str, 
    num_samples: int, 
    n_moves_prompt: int, 
    n_prefix_moves: int = 0, 
    force_regenerate: bool = False,
    use_prompt_cache: bool = True
):
    """Load or generate evaluation results, with controls for different caching mechanisms."""
    cache_key = f"{model_id.replace('/', '_')}_s{num_samples}_m{n_moves_prompt}_p{n_prefix_moves}"
    fname = OUTPUT_DIR / f"{cache_key}.jsonl"
    
    # Check if we need to force regenerate the results
    if not force_regenerate and fname.exists():
        with fname.open() as f:
            cached = [json.loads(l) for l in f]
            if len(cached) == num_samples:
                LOGGER.info(f"Loaded cached results for {model_id} ({num_samples} samples)")
                return cached

    # If we're here, we need to generate new results
    LOGGER.info(f"Generating new results for {model_id} ({num_samples} samples)")
    horizons, raw_outputs = await evaluate_model(
        cfg,
        model_id, 
        num_samples=num_samples, 
        n_moves_prompt=n_moves_prompt,
        n_prefix_moves=n_prefix_moves,
        use_cache=use_prompt_cache
    )
    
    records = []
    for text, hor in zip(raw_outputs, horizons):
        toks = tokenise_pgn(text)
        horizon, invalid_reason = horizon_valid_moves(toks) 
        records.append({
            "output": text,
            "horizon": horizon,
            "fen": final_fen(toks),
            "invalid_reason": invalid_reason
        })

    with fname.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            
    return records


def save_results(model_id: str, num_samples: int, n_moves_prompt: int, records: List[dict]):
    fname = OUTPUT_DIR / f"{model_id.replace('/', '_')}_s{num_samples}_m{n_moves_prompt}.jsonl"
    with fname.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def diversity_metrics(records: List[dict]):
    horizons = [r["horizon"] for r in records]
    unique_outputs = len({r["output"] for r in records})
    unique_fens = len({r["fen"] for r in records})
    n = len(records)
    
    # Count failure reasons - with better categorization
    failure_reasons = defaultdict(int)
    for r in records:
        if r["invalid_reason"]:
            # Special handling for non-error cases
            if r["invalid_reason"] == "OUTPUT_TRUNCATED":
                failure_reasons["OUTPUT_TRUNCATED (normal end of generation)"] += 1
            elif r["invalid_reason"] == "NO_MOVES_GENERATED":
                failure_reasons["NO_MOVES_GENERATED (empty response)"] += 1
            else:
                # Parse the error message - format is typically:
                # 'move' (token #N, after X valid moves): error message
                try:
                    if "': " in r["invalid_reason"]:
                        # Extract just the error message
                        move_part, error_part = r["invalid_reason"].split("': ", 1)
                        if "illegal move" in error_part.lower():
                            failure_reasons["ILLEGAL_MOVE"] += 1
                        elif "invalid piece" in error_part.lower():
                            failure_reasons["INVALID_PIECE"] += 1
                        elif "ambiguous move" in error_part.lower():
                            failure_reasons["AMBIGUOUS_MOVE"] += 1
                        else:
                            # Use a simpler classification for other errors
                            failure_reasons["OTHER_CHESS_ERROR"] += 1
                    else:
                        # Fall back to using the original message
                        failure_reasons["PARSE_ERROR"] += 1
                except Exception:
                    # If anything goes wrong with parsing, use a catch-all
                    failure_reasons["UNKNOWN_ERROR"] += 1
    
    return {
        "mean": statistics.mean(horizons) if horizons else 0,
        "median": statistics.median(horizons) if horizons else 0,
        "max": max(horizons) if horizons else 0,
        "min": min(horizons) if horizons else 0,
        "unique_output_ratio": unique_outputs / n if n > 0 else 0,
        "unique_fen_ratio": unique_fens / n if n > 0 else 0,
        "horizons": horizons,
        "failure_reasons": dict(failure_reasons)
    }


def plot_horizons(model_to_metrics: dict, n_moves_prompt: int, num_samples: int, n_prefix_moves: int = 0):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for idx, (model, metrics) in enumerate(model_to_metrics.items()):
        fig.add_trace(
            go.Box(
                y=metrics["horizons"],
                name=model,
                boxmean=True,
                marker_color=colors[idx % len(colors)],
            )
        )
    
    # Create title with all relevant experiment parameters
    prefix_info = f", {n_prefix_moves} prefilled moves" if n_prefix_moves > 0 else ", no prefilled moves"
    title = f"PGN valid-move horizon across models (n={num_samples} samples, {n_moves_prompt} moves requested{prefix_info})"
    
    fig.update_layout(
        title=title,
        yaxis_title="Consecutive legal moves"
    )
    fig.show()


def plot_failure_reasons(model_to_metrics: dict, num_samples: int, n_prefix_moves: int = 0):
    """Plot the reasons for move validation failures."""
    all_reasons = set()
    for metrics in model_to_metrics.values():
        all_reasons.update(metrics["failure_reasons"].keys())
    
    all_reasons = sorted(all_reasons)
    
    fig = go.Figure()
    for idx, (model, metrics) in enumerate(model_to_metrics.items()):
        reasons_count = []
        for reason in all_reasons:
            reasons_count.append(metrics["failure_reasons"].get(reason, 0))
        
        fig.add_trace(go.Bar(
            y=all_reasons,
            x=reasons_count,
            name=model,
            orientation='h',
        ))
    
    # Add prefix information to title
    prefix_info = f", {n_prefix_moves} prefilled moves" if n_prefix_moves > 0 else ", no prefilled moves"
    title = f"Reasons for First Invalid Move (n={num_samples} samples{prefix_info})"
    
    fig.update_layout(
        title=title,
        xaxis_title="Count",
        barmode='group'
    )
    fig.show()


async def main(cfg: ExperimentConfig):
    """Main function to run the evaluation."""
    model_ids = cfg.models if cfg.models else [cfg.model]
    
    # Process each model
    model_to_metrics = {}
    for m in model_ids:
        LOGGER.info(f"Evaluating model: {m}")
        
        recs = await load_or_generate(
            cfg,
            m, 
            cfg.samples, 
            cfg.moves_prompt, 
            cfg.prefix_moves,
            force_regenerate=cfg.force,
            use_prompt_cache=not cfg.nopromptcache
        )
        model_to_metrics[m] = diversity_metrics(recs)
        
        # Log API cost after each model
        cfg.log_api_cost(metadata={"model": m, "samples": cfg.samples})

    # Print out reports
    for model, met in model_to_metrics.items():
        print(f"\n=== {model} ===")
        print(f"Samples: {cfg.samples}")
        print(f"Mean horizon: {met['mean']:.1f}")
        print(f"Median horizon: {met['median']}")
        print(f"Max horizon: {met['max']}\tMin: {met['min']}")
        print(f"Unique outputs: {met['unique_output_ratio']*100:.1f}%  |  Unique final FENs: {met['unique_fen_ratio']*100:.1f}%")
        print("Failure reasons:")
        for reason, count in met["failure_reasons"].items():
            print(f"  - {reason}: {count} times ({count/cfg.samples*100:.1f}%)")

    if cfg.samples >= 3:  # Only show plots if we have enough samples
        plot_horizons(model_to_metrics, cfg.moves_prompt, cfg.samples, cfg.prefix_moves)
        plot_failure_reasons(model_to_metrics, cfg.samples, cfg.prefix_moves)

    if cfg.save is not None:
        path = pathlib.Path(cfg.save)
        with path.open("w") as f:
            for model, met in model_to_metrics.items():
                f.write(json.dumps({"model": model, **met}) + "\n")
        print(f"Saved metrics to {path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="config")
    args = parser.parse_args()
    cfg = args.config
    
    # Set up experiment
    cfg.output_dir = OUTPUT_DIR
    cfg.setup_experiment(log_file_prefix="pgn-eval")
    
    # Run main
    asyncio.run(main(cfg)) 