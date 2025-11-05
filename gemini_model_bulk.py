import os
import pandas as pd
import time
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, deque
import numpy as np
from tqdm import tqdm
from anthropic import Anthropic
from openai import OpenAI
import logging
import threading

# Set these to your API keys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_queries.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    def __init__(self, tokens_per_min, max_parallel_requests=5):
        self.tokens_per_min = tokens_per_min
        self.max_parallel_requests = max_parallel_requests
        self.tokens_used = 0
        self.last_reset = time.time()
        self.request_times = deque(maxlen=100)  # Track recent request timestamps
        self.lock = threading.Lock()
        self.semaphore = threading.Semaphore(max_parallel_requests)
    
    def add_tokens(self, tokens):
        """Add tokens to the counter"""
        with self.lock:
            # Reset counter if minute has passed
            current_time = time.time()
            if current_time - self.last_reset >= 60:
                self.tokens_used = 0
                self.last_reset = current_time
            
            self.tokens_used += tokens
    
    def wait_if_needed(self, estimated_tokens):
        """Wait if rate limit is about to be exceeded"""
        with self.lock:
            current_time = time.time()
            
            # Reset counter if minute has passed
            if current_time - self.last_reset >= 60:
                self.tokens_used = 0
                self.last_reset = current_time
                return False  # No need to wait
            
            # If adding these tokens would exceed the limit
            if self.tokens_used + estimated_tokens > self.tokens_per_min:
                seconds_to_wait = 60 - (current_time - self.last_reset) + 0.5  # Add a small buffer
                return seconds_to_wait
            
            return False  # No need to wait
    
    def acquire(self, estimated_tokens):
        """Acquire permission to make a request, waiting if necessary"""
        # First use the semaphore to limit concurrent requests
        self.semaphore.acquire()
        
        # Then check for rate limiting
        wait_time = self.wait_if_needed(estimated_tokens)
        if wait_time:
            logger.info(f"Rate limit approaching, waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            # After waiting, the minute counter should reset
            with self.lock:
                self.tokens_used = 0
                self.last_reset = time.time()
        
        # Record this request
        with self.lock:
            self.request_times.append(time.time())
        
        return estimated_tokens
    
    def release(self, tokens_used):
        """Release the semaphore and add tokens used"""
        self.add_tokens(tokens_used)
        self.semaphore.release()

# Create rate limiters
# OpenAI GPT-4o limit is 30k tokens per minute
openai_limiter = RateLimiter(tokens_per_min=28000, max_parallel_requests=3)
# Claude limit is ~100k tokens per minute (but we'll be conservative)
claude_limiter = RateLimiter(tokens_per_min=90000, max_parallel_requests=5)

def load_gene_lists_from_files(data_dir, top_n=100):
    """Load top genes from CSV files with reduced list size for multiple runs"""
    gene_lists = {}
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith('gene_scores_')]
    
    for file in csv_files:
        scheme_name = file.replace('gene_scores_', '').replace('.csv', '').replace('_', ' ').title()
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            gene_col = df.columns[0]
            score_col = df.columns[1]
            df = df.sort_values(score_col, ascending=False)
            top_genes = df.iloc[:top_n][gene_col].tolist()
            gene_lists[scheme_name] = top_genes
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    
    return gene_lists

def get_cart_prompt(gene_lists=None):
    """Get the CAR T prompt with optional gene lists"""
    cart_prompt = """As an expert in cancer biology and computational oncology, analyze these weighted scoring results for potential CAR T targets.

The scoring features include:
- Clinical Trials: Scores based on clinical trial progression (phase reached), availability of antibodies/drugs, and number of cancer-related studies
- HPA: Cell surface localization scores from Human Protein Atlas and UniProt (10=confirmed surface in both, 8=confirmed in UniProt/predicted in HPA, 7=confirmed in one source only, 5=predicted in HPA only)
- Heatmap: Z-score difference between log fold change in malignant vs. normal cells from bulk rna data
- Vital/Non-Vital: GTEx bulk RNA expression in vital tissues (brain, lung, heart) vs. non-vital tissues
- Protein Vital/Protein Non-Vital: Protein expression levels in vital vs. non-vital tissues from ProteomicsDB

Current weights are:
- Base weights: Clinical trials (1.0), HPA (2.0), Heatmap (5.0), Vital (2.0), Non-vital (1.0), Protein vital (2.0), Protein non-vital (1.0)
- Clinical trials emphasis: Clinical trials (5.0), HPA (2.0), Heatmap (3.0), Vital (1.0), Non-vital (0.5), Protein vital (1.0), Protein non-vital (0.5)
- HPA emphasis: Clinical trials (1.0), HPA (5.0), Heatmap (3.0), Vital (1.0), Non-vital (0.5), Protein vital (1.0), Protein non-vital (0.5)
- Heatmap emphasis: Clinical trials (1.0), HPA (2.0), Heatmap (10.0), Vital (1.0), Non-vital (0.5), Protein vital (1.0), Protein non-vital (0.5)
- Expression emphasis: Clinical trials (1.0), HPA (1.0), Heatmap (2.0), Vital (5.0), Non-vital (2.0), Protein vital (5.0), Protein non-vital (2.0)
- Balanced approach: Clinical trials (2.0), HPA (2.0), Heatmap (2.0), Vital (2.0), Non-vital (1.0), Protein vital (2.0), Protein non-vital (1.0)
- Equal weights: All features weighted equally (1.0)

Based on your expertise in CAR T development, what would be the optimal weighting scheme to prioritize genes that are:
1. Highly specific to tumor cells vs. normal tissue (high heatmap Z-score)
2. Accessible on the cell surface (high HPA score)
3. Limited expression in vital tissues (low vital/protein vital scores)
4. Have existing clinical evidence supporting targetability (clinical trials score)

IMPORTANT: Start your response with a clear specification of your recommended weights in this exact format:
"RECOMMENDED WEIGHTS: Clinical trials (X.X), HPA (X.X), Heatmap (X.X), Vital (X.X), Non-vital (X.X), Protein vital (X.X), Protein non-vital (X.X)"

Then explain your reasoning for the recommended weights and how they align with established principles for identifying ideal CAR T targets. Which of the top-ranked genes from your recommended weighting scheme appear most promising for further investigation?"""

    if gene_lists:
        gene_lists_text = "\n\nTop genes from different weighting schemes:\n"
        for name, genes in gene_lists.items():
            gene_lists_text += f"- {name}: {', '.join(genes)}\n"
        cart_prompt += gene_lists_text
    
    return cart_prompt

def query_openai(run_id, temperature=0.7):
    """Query OpenAI API with the CAR T prompt using rate limiting"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = get_cart_prompt(gene_lists)
    
    # Estimate token count (very rough approximation)
    estimated_tokens = len(prompt) // 4 + 500  # Input tokens + response buffer
    
    try:
        # Acquire token allocation from rate limiter
        openai_limiter.acquire(estimated_tokens)
        
        max_retries = 3
        backoff_time = 2
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert in cancer biology, computational oncology, and CAR T cell therapy development."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                
                result = response.choices[0].message.content
                
                # Update tokens used based on actual usage
                completion_tokens = response.usage.completion_tokens
                prompt_tokens = response.usage.prompt_tokens
                total_tokens = completion_tokens + prompt_tokens
                
                # Release with actual tokens used
                openai_limiter.release(total_tokens)
                
                return {
                    "run_id": run_id,
                    "model": "gpt-4o",
                    "temperature": temperature,
                    "response": result,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tokens_used": total_tokens
                }
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"OpenAI API Error (run {run_id}, attempt {attempt+1}/{max_retries}): {error_str}")
                
                # Handle rate limiting errors specifically
                if "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                    wait_time = backoff_time * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Rate limit hit, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                else:
                    # For other errors, shorter backoff
                    time.sleep(backoff_time)
        
        # If we get here, all retries failed
        logger.error(f"All retries failed for OpenAI query (run {run_id})")
        openai_limiter.release(0)  # Release the semaphore but don't count tokens
        return None
                
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI query wrapper (run {run_id}): {str(e)}")
        openai_limiter.release(0)  # Release the semaphore but don't count tokens
        return None

def query_claude(run_id, temperature=0.7):
    """Query Anthropic Claude API with the CAR T prompt using rate limiting"""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = get_cart_prompt(gene_lists)
    
    # Estimate token count (very rough approximation)
    estimated_tokens = len(prompt) // 4 + 1000  # Input tokens + response buffer
    
    try:
        # Acquire token allocation from rate limiter
        claude_limiter.acquire(estimated_tokens)
        
        max_retries = 3
        backoff_time = 2
        
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=4000,
                    temperature=temperature,
                    system="You are an expert in cancer biology, computational oncology, and CAR T cell therapy development.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result = response.content[0].text
                
                # Estimate tokens used since Claude doesn't report it directly in the same way
                approximate_tokens = len(prompt + result) // 4
                
                # Release with estimated tokens
                claude_limiter.release(approximate_tokens)
                
                return {
                    "run_id": run_id,
                    "model": "claude-3-7-sonnet",
                    "temperature": temperature,
                    "response": result,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tokens_used": approximate_tokens
                }
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Claude API Error (run {run_id}, attempt {attempt+1}/{max_retries}): {error_str}")
                
                # Handle rate limiting errors specifically
                if "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                    wait_time = backoff_time * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Rate limit hit, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                else:
                    # For other errors, shorter backoff
                    time.sleep(backoff_time)
        
        # If we get here, all retries failed
        logger.error(f"All retries failed for Claude query (run {run_id})")
        claude_limiter.release(0)  # Release the semaphore but don't count tokens
        return None
                
    except Exception as e:
        logger.error(f"Unexpected error in Claude query wrapper (run {run_id}): {str(e)}")
        claude_limiter.release(0)  # Release the semaphore but don't count tokens
        return None

def extract_weights(response_text):
    """Extract weight recommendations from response text"""
    # Try the exact formatted version first (standard pattern)
    pattern1 = r"RECOMMENDED WEIGHTS:\s*Clinical trials \((\d+\.\d+)\),\s*HPA \((\d+\.\d+)\),\s*Heatmap \((\d+\.\d+)\),\s*Vital \((\d+\.\d+)\),\s*Non-vital \((\d+\.\d+)\),\s*Protein vital \((\d+\.\d+)\),\s*Protein non-vital \((\d+\.\d+)\)"
    match = re.search(pattern1, response_text, re.IGNORECASE)
    
    if match:
        return {
            "Clinical_trials": float(match.group(1)),
            "HPA": float(match.group(2)),
            "Heatmap": float(match.group(3)),
            "Vital": float(match.group(4)),
            "Non_vital": float(match.group(5)),
            "Protein_vital": float(match.group(6)),
            "Protein_non_vital": float(match.group(7))
        }
    
    # Try alternative format with bold/markdown formatting
    pattern2 = r"\*\*RECOMMENDED WEIGHTS:?\*\*:?\s*Clinical trials \((\d+\.\d+)\),\s*HPA \((\d+\.\d+)\),\s*Heatmap \((\d+\.\d+)\),\s*Vital \((\d+\.\d+)\),\s*Non-vital \((\d+\.\d+)\),\s*Protein vital \((\d+\.\d+)\),\s*Protein non-vital \((\d+\.\d+)\)"
    match = re.search(pattern2, response_text)
    
    if match:
        return {
            "Clinical_trials": float(match.group(1)),
            "HPA": float(match.group(2)),
            "Heatmap": float(match.group(3)),
            "Vital": float(match.group(4)),
            "Non_vital": float(match.group(5)),
            "Protein_vital": float(match.group(6)),
            "Protein_non_vital": float(match.group(7))
        }
    
    # Try a more flexible pattern for each individual weight
    weights = {}
    patterns = {
        "Clinical_trials": r"Clinical trials\s*\(?(\d+\.?\d*)\)?",
        "HPA": r"HPA\s*\(?(\d+\.?\d*)\)?",
        "Heatmap": r"Heatmap\s*\(?(\d+\.?\d*)\)?",
        "Vital": r"(?<!\w)Vital\s*\(?(\d+\.?\d*)\)?",  # negative lookbehind to avoid matching "Protein vital"
        "Non_vital": r"Non-vital\s*\(?(\d+\.?\d*)\)?",
        "Protein_vital": r"Protein vital\s*\(?(\d+\.?\d*)\)?",
        "Protein_non_vital": r"Protein non-vital\s*\(?(\d+\.?\d*)\)?"
    }
    
    # Look for individual weights
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text)
        if match:
            try:
                weights[key] = float(match.group(1))
            except (ValueError, TypeError):
                # Skip if we can't convert to float
                continue
    
    # Return weights if we found at least 4 features
    if len(weights) >= 4:
        return weights
    
    # If all else fails, try a line-by-line approach to find weight numbers
    if len(weights) < 4:
        # Look for lines like "1. Heatmap (5.0):" or "1. **Heatmap (5.0)**:"
        for line in response_text.split('\n'):
            for key in patterns.keys():
                # Convert key to a readable format
                readable_key = key.replace('_', ' ')
                if readable_key in line:
                    # Extract numbers from the line
                    numbers = re.findall(r'(\d+\.\d+)', line)
                    if numbers:
                        try:
                            weights[key] = float(numbers[0])
                        except (ValueError, TypeError, IndexError):
                            continue
        
        # Return weights if we found at least 4 features
        if len(weights) >= 4:
            return weights
    
    return None

def run_multi_queries(n_runs=100):
    """Run multiple queries to both APIs and collect results with better rate limiting"""
    os.makedirs("results", exist_ok=True)
    os.makedirs("/data/ep924610/project_nb/paper_code/bulk_review/llm/raw", exist_ok=True)
    
    # Check if we have existing results to resume from
    openai_results = []
    claude_results = []
    
    # Check for existing OpenAI results
    existing_openai_files = set()
    for filename in os.listdir("/data/ep924610/project_nb/paper_code/bulk_review/llm/raw"):
        if filename.startswith("openai_run_") and filename.endswith(".json"):
            run_id = int(filename.replace("openai_run_", "").replace(".json", ""))
            existing_openai_files.add(run_id)
    
    # Check for existing Claude results
    existing_claude_files = set()
    for filename in os.listdir("/data/ep924610/project_nb/paper_code/bulk_review/llm/raw"):
        if filename.startswith("claude_run_") and filename.endswith(".json"):
            run_id = int(filename.replace("claude_run_", "").replace(".json", ""))
            existing_claude_files.add(run_id)
    
    logger.info(f"Found {len(existing_openai_files)} existing OpenAI results and {len(existing_claude_files)} existing Claude results")
    
    # Create sets of run_ids that need to be processed
    openai_todo = set(range(n_runs)) - existing_openai_files
    claude_todo = set(range(n_runs)) - existing_claude_files
    
    logger.info(f"Need to run {len(openai_todo)} OpenAI queries and {len(claude_todo)} Claude queries")
    
    # Run OpenAI queries for missing run_ids
    if openai_todo:
        logger.info(f"Starting {len(openai_todo)} OpenAI runs...")
        openai_pbar = tqdm(total=len(openai_todo), desc="OpenAI Queries")
        
        def process_openai(run_id):
            result = query_openai(run_id, temperature=0.7)
            if result:
                # Save individual result
                with open(f"/data/ep924610/project_nb/paper_code/bulk_review/llm/raw/openai_run_{result['run_id']}.json", "w") as f:
                    json.dump(result, f, indent=2)
                openai_results.append(result)
                openai_pbar.update(1)
                return result
            return None
        
        # Run in batches to better manage rate limits
        batch_size = 20
        openai_todo_list = sorted(list(openai_todo))
        
        for batch_start in range(0, len(openai_todo_list), batch_size):
            batch_end = min(batch_start + batch_size, len(openai_todo_list))
            batch = openai_todo_list[batch_start:batch_end]
            
            with ThreadPoolExecutor(max_workers=openai_limiter.max_parallel_requests) as executor:
                futures = [executor.submit(process_openai, run_id) for run_id in batch]
                for future in futures:
                    future.result()  # Just to catch any exceptions
            
            # Short pause between batches
            if batch_end < len(openai_todo_list):
                logger.info(f"Completed OpenAI batch {batch_start}-{batch_end}, pausing briefly...")
                time.sleep(5)  # Short pause between batches
        
        openai_pbar.close()
    
    # Run Claude queries for missing run_ids
    if claude_todo:
        logger.info(f"Starting {len(claude_todo)} Claude runs...")
        claude_pbar = tqdm(total=len(claude_todo), desc="Claude Queries")
        
        def process_claude(run_id):
            result = query_claude(run_id, temperature=0.7)
            if result:
                # Save individual result
                with open(f"/data/ep924610/project_nb/paper_code/bulk_review/llm/raw/claude_run_{result['run_id']}.json", "w") as f:
                    json.dump(result, f, indent=2)
                claude_results.append(result)
                claude_pbar.update(1)
                return result
            return None
        
        # Run in batches to better manage rate limits
        batch_size = 30  # Claude can handle more requests per batch
        claude_todo_list = sorted(list(claude_todo))
        
        for batch_start in range(0, len(claude_todo_list), batch_size):
            batch_end = min(batch_start + batch_size, len(claude_todo_list))
            batch = claude_todo_list[batch_start:batch_end]
            
            with ThreadPoolExecutor(max_workers=claude_limiter.max_parallel_requests) as executor:
                futures = [executor.submit(process_claude, run_id) for run_id in batch]
                for future in futures:
                    future.result()  # Just to catch any exceptions
            
            # Short pause between batches
            if batch_end < len(claude_todo_list):
                logger.info(f"Completed Claude batch {batch_start}-{batch_end}, pausing briefly...")
                time.sleep(3)  # Short pause between batches
        
        claude_pbar.close()
    
    # Load all results (including existing ones)
    all_openai_results = []
    all_claude_results = []
    
    for run_id in range(n_runs):
        # Try to load OpenAI result
        openai_file = f"/data/ep924610/project_nb/paper_code/bulk_review/llm/raw/openai_run_{run_id}.json"
        if os.path.exists(openai_file):
            try:
                with open(openai_file, "r") as f:
                    all_openai_results.append(json.load(f))
            except Exception as e:
                logger.error(f"Error loading {openai_file}: {str(e)}")
        
        # Try to load Claude result
        claude_file = f"/data/ep924610/project_nb/paper_code/bulk_review/llm/raw/claude_run_{run_id}.json"
        if os.path.exists(claude_file):
            try:
                with open(claude_file, "r") as f:
                    all_claude_results.append(json.load(f))
            except Exception as e:
                logger.error(f"Error loading {claude_file}: {str(e)}")
    
    # Save consolidated results
    with open("/data/ep924610/project_nb/paper_code/bulk_review/llm/openai_all_results.json", "w") as f:
        json.dump(all_openai_results, f, indent=2)
    
    with open("/data/ep924610/project_nb/paper_code/bulk_review/llm/claude_all_results.json", "w") as f:
        json.dump(all_claude_results, f, indent=2)
    
    logger.info(f"Completed {len(all_openai_results)} OpenAI runs and {len(all_claude_results)} Claude runs")
    return all_openai_results, all_claude_results

def analyze_results(openai_results, claude_results):
    """Analyze results and extract weight recommendations"""
    # Extract weights from responses
    openai_weights = []
    claude_weights = []
    
    for result in tqdm(openai_results, desc="Processing OpenAI results"):
        weights = extract_weights(result["response"])
        if weights:
            weights["model"] = "GPT-4o"
            weights["run_id"] = result["run_id"]
            openai_weights.append(weights)
    
    for result in tqdm(claude_results, desc="Processing Claude results"):
        weights = extract_weights(result["response"])
        if weights:
            weights["model"] = "Claude-3.7"
            weights["run_id"] = result["run_id"]
            claude_weights.append(weights)
    
    # Combine weights into a DataFrame
    all_weights = pd.DataFrame(openai_weights + claude_weights)
    
    # Save extracted weights
    all_weights.to_csv("/data/ep924610/project_nb/paper_code/bulk_review/llm/extracted_weights.csv", index=False)
    
    logger.info(f"Extracted weights from {len(openai_weights)} OpenAI responses and {len(claude_weights)} Claude responses")
    
    # Create summary statistics
    weight_features = ["Clinical_trials", "HPA", "Heatmap", "Vital", "Non_vital", "Protein_vital", "Protein_non_vital"]
    
    summary = all_weights.groupby("model")[weight_features].agg(['mean', 'median', 'std', 'min', 'max'])
    summary.to_csv("/data/ep924610/project_nb/paper_code/bulk_review/llm/weight_summary_stats.csv")
    
    # Count most common weight combinations by model
    def get_weight_combo(row):
        return tuple(round(row[feature], 1) for feature in weight_features)
    
    all_weights["weight_combo"] = all_weights.apply(get_weight_combo, axis=1)
    combo_counts = all_weights.groupby(["model", "weight_combo"]).size().reset_index(name="count")
    combo_counts = combo_counts.sort_values(["model", "count"], ascending=[True, False])
    
    # Save top combinations
    top_combos = pd.DataFrame()
    for model in ["GPT-4o", "Claude-3.7"]:
        model_combos = combo_counts[combo_counts["model"] == model].head(5)
        model_combos["weight_combo_str"] = model_combos["weight_combo"].apply(
            lambda x: ", ".join([f"{feat}={val}" for feat, val in zip(weight_features, x)])
        )
        top_combos = pd.concat([top_combos, model_combos])
    
    top_combos.to_csv("/data/ep924610/project_nb/paper_code/bulk_review/llm/top_weight_combinations.csv", index=False)
    
    return all_weights, summary, top_combos

def create_visualizations(all_weights, summary):
    """Create visualizations for weight analysis"""
    os.makedirs("/data/ep924610/project_nb/paper_code/bulk_review/llm/visualizations", exist_ok=True)
    
    # Feature names for plotting
    weight_features = ["Clinical_trials", "HPA", "Heatmap", "Vital", "Non_vital", "Protein_vital", "Protein_non_vital"]
    
    # 1. Distribution of weights by model
    plt.figure(figsize=(16, 10))
    for i, feature in enumerate(weight_features):
        plt.subplot(2, 4, i+1)
        sns.boxplot(x="model", y=feature, data=all_weights)
        plt.title(f"Distribution of {feature.replace('_', ' ')} Weights")
        plt.ylabel("Weight Value")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/data/ep924610/project_nb/paper_code/bulk_review/llm/visualizations/weight_distributions.png", dpi=300)
    
    # 2. Heatmap of average weights by model
    avg_weights = all_weights.groupby("model")[weight_features].mean()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(avg_weights, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Average Weights by Model")
    plt.tight_layout()
    plt.savefig("/data/ep924610/project_nb/paper_code/bulk_review/llm/visualizations/average_weights_heatmap.png", dpi=300)
    
    # 3. Radar chart for comparing models
    def radar_chart(avg_weights, weight_features):
        # Number of variables
        N = len(weight_features)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Initialise the plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], [f.replace('_', ' ') for f in weight_features], size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        max_val = avg_weights.values.max() * 1.1
        plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=10)
        plt.ylim(0, max_val)
        
        # Plot each model
        for model in avg_weights.index:
            values = avg_weights.loc[model].values.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Weight Comparison by Model", size=15)
        return fig
    
    radar_fig = radar_chart(avg_weights, weight_features)
    radar_fig.savefig("/data/ep924610/project_nb/paper_code/bulk_review/llm/visualizations/weight_radar_chart.png", dpi=300)
    
    # 4. Bar chart of top weight combinations
    top_combos = pd.DataFrame()
    for model in ["GPT-4o", "Claude-3.7"]:
        model_data = all_weights[all_weights["model"] == model]
        combos = model_data.groupby("weight_combo").size().reset_index(name="count")
        combos = combos.sort_values("count", ascending=False).head(5)
        combos["model"] = model
        top_combos = pd.concat([top_combos, combos])
    
    # Convert combinations to readable strings for the plot
    top_combos["combo_str"] = top_combos["weight_combo"].apply(
        lambda x: f"{x[2]}-{x[1]}-{x[0]}"  # Showing just Heatmap-HPA-ClinTrials
    )
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x="combo_str", y="count", hue="model", data=top_combos)
    plt.title("Top 5 Weight Combinations by Model")
    plt.xlabel("Weight Combinations (Heatmap-HPA-ClinTrials)")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/data/ep924610/project_nb/paper_code/bulk_review/llm/visualizations/top_weight_combinations.png", dpi=300)
    
    logger.info("Visualizations created in /data/ep924610/project_nb/paper_code/bulk_review/llm/visualizations/")

def generate_summary_report(all_weights, summary, top_combos):
    """Generate a summary report of the analysis"""
    report = []
    report.append("# CAR T Target Weight Recommendation Analysis")
    report.append(f"Analysis Date: {time.strftime('%Y-%m-%d')}")
    report.append(f"Total Runs: {len(all_weights)}")
    report.append(f"Models: GPT-4o ({len(all_weights[all_weights.model == 'GPT-4o'])}) and Claude-3.7 ({len(all_weights[all_weights.model == 'Claude-3.7'])})")
    report.append("\n## Summary Statistics")
    
    # Add mean weights table
    report.append("\n### Mean Weights by Model")
    means = all_weights.groupby("model")[["Clinical_trials", "HPA", "Heatmap", "Vital", "Non_vital", "Protein_vital", "Protein_non_vital"]].mean()
    report.append(means.to_markdown())
    
    # Add top weight combinations
    report.append("\n### Top Weight Combinations")
    weight_features = ["Clinical_trials", "HPA", "Heatmap", "Vital", "Non_vital", "Protein_vital", "Protein_non_vital"]
    
    for model in ["GPT-4o", "Claude-3.7"]:
        report.append(f"\n#### {model}")
        model_combos = top_combos[top_combos["model"] == model].head(5)
        
        for i, (_, row) in enumerate(model_combos.iterrows()):
            combo = row["weight_combo"]
            count = row["count"]
            combo_str = ", ".join([f"{feat}={val}" for feat, val in zip(weight_features, combo)])
            report.append(f"{i+1}. **{combo_str}** (frequency: {count})")
    
    # Add key observations
    report.append("\n## Key Observations")
    
    # Compare relative emphasis on each feature
    report.append("\n### Feature Emphasis")
    for feature in weight_features:
        gpt_mean = all_weights[all_weights.model == "GPT-4o"][feature].mean()
        claude_mean = all_weights[all_weights.model == "Claude-3.7"][feature].mean()
        
        report.append(f"- **{feature.replace('_', ' ')}**: GPT-4o: {gpt_mean:.2f}, Claude-3.7: {claude_mean:.2f}")
    
    # Find consensus (features both models emphasize)
    report.append("\n### Consensus Between Models")
    gpt_features = all_weights[all_weights.model == "GPT-4o"][weight_features].mean().sort_values(ascending=False)
    claude_features = all_weights[all_weights.model == "Claude-3.7"][weight_features].mean().sort_values(ascending=False)
    
    report.append("**Top 3 features by GPT-4o:**")
    for feature, value in gpt_features.head(3).items():
        report.append(f"- {feature.replace('_', ' ')}: {value:.2f}")
    
    report.append("\n**Top 3 features by Claude-3.7:**")
    for feature, value in claude_features.head(3).items():
        report.append(f"- {feature.replace('_', ' ')}: {value:.2f}")
    
    # Save the report
    with open("/data/ep924610/project_nb/paper_code/bulk_review/llm/weight_analysis_report.md", "w") as f:
        f.write("\n".join(report))
    
    logger.info("Summary report generated: /data/ep924610/project_nb/paper_code/bulk_review/llm/weight_analysis_report.md")



import os
import pandas as pd
import time
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
import traceback
import logging
from datetime import datetime
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gemini_analysis")



def load_gene_lists_from_files(data_dir, top_n=20):
    """Load top genes from CSV files with reduced list size for multiple runs"""
    gene_lists = {}
    
    try:
        # List all CSV files in the directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith('gene_scores_')]
        
        if not csv_files:
            logger.warning(f"No matching CSV files found in {data_dir}")
            return gene_lists
            
        for file in csv_files:
            # Extract weighting scheme name from filename
            scheme_name = file.replace('gene_scores_', '').replace('.csv', '').replace('_', ' ').title()
            
            try:
                # Load CSV file
                df = pd.read_csv(os.path.join(data_dir, file))
                
                # Assume first column is gene name, second is score
                gene_col = df.columns[0]
                score_col = df.columns[1]
                
                # Sort by score in descending order
                df = df.sort_values(score_col, ascending=False)
                
                # Get the top N genes
                top_genes = df.iloc[:top_n][gene_col].tolist()
                gene_lists[scheme_name] = top_genes
                logger.info(f"Loaded {len(top_genes)} genes from {file}")
                
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
    except Exception as e:
        logger.error(f"Error accessing data directory: {str(e)}")
    
    return gene_lists

class ExponentialBackoffRateLimiter:
    """Advanced rate limiter with exponential backoff for Gemini API"""
    def __init__(self, requests_per_minute=15, burst_capacity=5):
        self.requests_per_minute = requests_per_minute
        self.burst_capacity = burst_capacity
        self.request_times = []
        self.consecutive_failures = 0
        self.last_request_time = 0
        
    def wait_if_needed(self):
        """Wait if we're exceeding the rate limit with exponential backoff"""
        now = time.time()
        
        # Remove timestamps older than a minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Calculate base delay between requests
        base_delay = 60.0 / self.requests_per_minute
        
        # Add exponential backoff for consecutive failures
        backoff_delay = min(300, base_delay * (2 ** self.consecutive_failures))  # Max 5 minutes
        
        # Ensure minimum time between requests
        time_since_last = now - self.last_request_time
        if time_since_last < backoff_delay:
            sleep_time = backoff_delay - time_since_last
            logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds (failures: {self.consecutive_failures})")
            time.sleep(sleep_time)
        
        # If we're at capacity, wait until we're not
        if len(self.request_times) >= self.burst_capacity:
            oldest = min(self.request_times)
            sleep_time = max(0, 60 - (now - oldest))
            if sleep_time > 0:
                logger.info(f"Burst capacity reached, waiting {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(time.time())
        self.last_request_time = time.time()
    
    def on_success(self):
        """Reset failure counter on successful request"""
        self.consecutive_failures = 0
    
    def on_failure(self):
        """Increment failure counter"""
        self.consecutive_failures += 1

# Initialize rate limiter for Gemini (conservative settings)
gemini_limiter = ExponentialBackoffRateLimiter(requests_per_minute=15, burst_capacity=3)

def load_existing_results():
    """Load existing OpenAI and Claude results from JSON files"""
    openai_results = []
    claude_results = []
    
    try:
        # Load OpenAI results
        if os.path.exists("results/openai_all_results.json"):
            with open("results/openai_all_results.json", "r") as f:
                openai_results = json.load(f)
            logger.info(f"Loaded {len(openai_results)} existing OpenAI results")
        else:
            logger.warning("OpenAI results file not found")
        
        # Load Claude results
        if os.path.exists("results/claude_all_results.json"):
            with open("results/claude_all_results.json", "r") as f:
                claude_results = json.load(f)
            logger.info(f"Loaded {len(claude_results)} existing Claude results")
        else:
            logger.warning("Claude results file not found")
            
    except Exception as e:
        logger.error(f"Error loading existing results: {str(e)}")
    
    return openai_results, claude_results

def get_gemini_progress():
    """Check how many Gemini results we already have"""
    try:
        if os.path.exists("results/gemini_all_results.json"):
            with open("results/gemini_all_results.json", "r") as f:
                gemini_results = json.load(f)
            logger.info(f"Found {len(gemini_results)} existing Gemini results")
            return gemini_results
        return []
    except Exception as e:
        logger.error(f"Error checking Gemini progress: {str(e)}")
        return []

def get_cart_prompt(gene_lists=None):
    """Get the CAR T prompt with optional gene lists"""
    cart_prompt = """As an expert in cancer biology and computational oncology, analyze these weighted scoring results for potential CAR T targets.

The scoring features include:
- Clinical Trials: Scores based on clinical trial progression (phase reached), availability of antibodies/drugs, and number of cancer-related studies
- HPA: Cell surface localization scores from Human Protein Atlas and UniProt (10=confirmed surface in both, 8=confirmed in UniProt/predicted in HPA, 7=confirmed in one source only, 5=predicted in HPA only)
- Heatmap: Z-score difference between log fold change in malignant vs. normal cells from bulk rna data
- Vital/Non-Vital: GTEx bulk RNA expression in vital tissues (brain, lung, heart) vs. non-vital tissues
- Protein Vital/Protein Non-Vital: Protein expression levels in vital vs. non-vital tissues from ProteomicsDB

Current weights are:
- Base weights: Clinical trials (1.0), HPA (2.0), Heatmap (5.0), Vital (2.0), Non-vital (1.0), Protein vital (2.0), Protein non-vital (1.0)
- Clinical trials emphasis: Clinical trials (5.0), HPA (2.0), Heatmap (3.0), Vital (1.0), Non-vital (0.5), Protein vital (1.0), Protein non-vital (0.5)
- HPA emphasis: Clinical trials (1.0), HPA (5.0), Heatmap (3.0), Vital (1.0), Non-vital (0.5), Protein vital (1.0), Protein non-vital (0.5)
- Heatmap emphasis: Clinical trials (1.0), HPA (2.0), Heatmap (10.0), Vital (1.0), Non-vital (0.5), Protein vital (1.0), Protein non-vital (0.5)
- Expression emphasis: Clinical trials (1.0), HPA (1.0), Heatmap (2.0), Vital (5.0), Non-vital (2.0), Protein vital (5.0), Protein non-vital (2.0)
- Balanced approach: Clinical trials (2.0), HPA (2.0), Heatmap (2.0), Vital (2.0), Non-vital (1.0), Protein vital (2.0), Protein non-vital (1.0)
- Equal weights: All features weighted equally (1.0)

Based on your expertise in CAR T development, what would be the optimal weighting scheme to prioritize genes that are:
1. Highly specific to tumor cells vs. normal tissue (high heatmap Z-score)
2. Accessible on the cell surface (high HPA score)
3. Limited expression in vital tissues (low vital/protein vital scores)
4. Have existing clinical evidence supporting targetability (clinical trials score)

IMPORTANT: Start your response with a clear specification of your recommended weights in this exact format:
"RECOMMENDED WEIGHTS: Clinical trials (X.X), HPA (X.X), Heatmap (X.X), Vital (X.X), Non-vital (X.X), Protein vital (X.X), Protein non-vital (X.X)"

Then explain your reasoning for the recommended weights and how they align with established principles for identifying ideal CAR T targets. Which of the top-ranked genes from your recommended weighting scheme appear most promising for further investigation?"""

    if gene_lists:
        gene_lists_text = "\n\nTop genes from different weighting schemes:\n"
        for name, genes in gene_lists.items():
            gene_lists_text += f"- {name}: {', '.join(genes)}\n"
        cart_prompt += gene_lists_text
    
    return cart_prompt

def query_gemini_with_retry(run_id, gene_lists=None, temperature=0.7, max_retries=5):
    """Query Google Gemini API with retry logic and rate limiting"""
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Get the model
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro-preview-03-25",
        generation_config={
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
    )
    
    prompt = get_cart_prompt(gene_lists)
    
    for attempt in range(max_retries):
        try:
            # Apply rate limiting
            gemini_limiter.wait_if_needed()
            
            # Generate response
            response = model.generate_content([{"role": "user", "parts": [prompt]}])
            
            # Check if response is valid
            if not response.text or len(response.text.strip()) < 100:
                raise ValueError("Response too short or empty")
            
            # Success - reset failure counter
            gemini_limiter.on_success()
            
            result = {
                "run_id": run_id,
                "model": "Gemini-2.5-Pro",
                "temperature": temperature,
                "response": response.text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "attempt": attempt + 1
            }
            
            logger.info(f"Successfully completed run {run_id} on attempt {attempt + 1}")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Gemini API Error (run {run_id}, attempt {attempt + 1}/{max_retries}): {str(e)}")
            
            # Increment failure counter
            gemini_limiter.on_failure()
            
            # Check for specific error types
            if any(keyword in error_msg for keyword in ["quota", "rate", "limit", "429"]):
                # Rate limit or quota issue - wait longer
                wait_time = min(300, 60 * (2 ** attempt))  # Exponential backoff, max 5 minutes
                logger.info(f"Rate limit detected, waiting {wait_time} seconds before retry")
                time.sleep(wait_time)
            elif any(keyword in error_msg for keyword in ["500", "502", "503", "504", "internal"]):
                # Server error - wait and retry
                wait_time = min(60, 5 * (2 ** attempt))
                logger.info(f"Server error detected, waiting {wait_time} seconds before retry")
                time.sleep(wait_time)
            elif "safety" in error_msg or "blocked" in error_msg:
                # Safety filter - this won't resolve with retries
                logger.error(f"Safety filter triggered for run {run_id}, skipping")
                break
            else:
                # Other error - short wait
                time.sleep(min(30, 2 ** attempt))
            
            # If this was our last attempt, log the failure
            if attempt == max_retries - 1:
                logger.error(f"Failed to complete run {run_id} after {max_retries} attempts")
    
    return None

def run_gemini_queries(total_runs=1000, gene_lists=None, batch_size=25, max_threads=1):
    """Run multiple queries to Gemini API with improved retry logic"""
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)
    
    # Check for existing results
    existing_results = get_gemini_progress()
    existing_run_ids = set(r["run_id"] for r in existing_results)
    
    # Determine which runs we still need to do
    runs_to_do = [i for i in range(total_runs) if i not in existing_run_ids]
    
    if len(runs_to_do) == 0:
        logger.info("All Gemini runs already completed!")
        return existing_results
    
    logger.info(f"Already completed {len(existing_results)} Gemini runs. Need to run {len(runs_to_do)} more.")
    
    # Initialize results list with existing results
    gemini_results = existing_results.copy()
    
    # Create a progress bar for the runs
    pbar = tqdm(total=len(runs_to_do), desc="Gemini Queries", 
                initial=0, position=0, leave=True)
    
    # Track failed runs for retry
    failed_runs = []
    
    # Process runs in batches with single threading to respect rate limits
    for batch_start in range(0, len(runs_to_do), batch_size):
        batch_end = min(batch_start + batch_size, len(runs_to_do))
        batch_runs = runs_to_do[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: runs {batch_start} to {batch_end-1}")
        
        batch_results = []
        
        # Process each run in the batch sequentially
        for run_id in batch_runs:
            # Add some randomness to temperature
            temperature = 0.7
            
            result = query_gemini_with_retry(run_id, gene_lists, temperature)
            
            if result:
                batch_results.append(result)
                # Save individual result immediately
                with open(f"results/raw/gemini_run_{result['run_id']}.json", "w") as f:
                    json.dump(result, f, indent=2)
                pbar.update(1)
            else:
                failed_runs.append(run_id)
                logger.warning(f"Run {run_id} failed after all retries")
        
        # Add batch results to overall results
        gemini_results.extend(batch_results)
        
        # Save progress after each batch
        with open("results/gemini_all_results.json", "w") as f:
            json.dump(gemini_results, f, indent=2)
        
        logger.info(f"Batch complete. Successful: {len(batch_results)}, Failed: {len(batch_runs) - len(batch_results)}")
        
        # Wait between batches
        if batch_end < len(runs_to_do):
            logger.info("Waiting between batches...")
            time.sleep(30)
    
    # Retry failed runs if we don't have enough successful results
    if len(gemini_results) < total_runs and failed_runs:
        logger.info(f"Retrying {len(failed_runs)} failed runs to reach {total_runs} total results")
        
        retry_attempts = 0
        max_retry_rounds = 3
        
        while len(gemini_results) < total_runs and failed_runs and retry_attempts < max_retry_rounds:
            retry_attempts += 1
            logger.info(f"Retry round {retry_attempts}/{max_retry_rounds}")
            
            # Shuffle failed runs to try different ones first
            random.shuffle(failed_runs)
            
            still_failed = []
            
            for run_id in failed_runs:
                if len(gemini_results) >= total_runs:
                    break
                
                temperature = 0.7 + random.uniform(-0.1, 0.1)
                result = query_gemini_with_retry(run_id, gene_lists, temperature)
                
                if result:
                    gemini_results.append(result)
                    with open(f"results/raw/gemini_run_{result['run_id']}.json", "w") as f:
                        json.dump(result, f, indent=2)
                    pbar.update(1)
                    logger.info(f"Retry successful for run {run_id}")
                else:
                    still_failed.append(run_id)
                
                # Save progress
                with open("results/gemini_all_results.json", "w") as f:
                    json.dump(gemini_results, f, indent=2)
            
            failed_runs = still_failed
            
            if failed_runs and len(gemini_results) < total_runs:
                wait_time = 120 * retry_attempts  # Increasing wait time between retry rounds
                logger.info(f"Waiting {wait_time} seconds before next retry round")
                time.sleep(wait_time)
    
    pbar.close()
    
    final_count = len(gemini_results)
    if final_count >= total_runs:
        logger.info(f"Successfully completed {final_count} Gemini runs (target: {total_runs})")
    else:
        logger.warning(f"Completed {final_count}/{total_runs} Gemini runs. {len(failed_runs)} runs failed permanently.")
    
    return gemini_results

def extract_weights(response_text):
    """Extract weight recommendations from response text"""
    # Try the formatted version first
    pattern = r"RECOMMENDED WEIGHTS:\s*Clinical trials \((\d+\.?\d*)\),\s*HPA \((\d+\.?\d*)\),\s*Heatmap \((\d+\.?\d*)\),\s*Vital \((\d+\.?\d*)\),\s*Non-vital \((\d+\.?\d*)\),\s*Protein vital \((\d+\.?\d*)\),\s*Protein non-vital \((\d+\.?\d*)\)"
    match = re.search(pattern, response_text)
    
    if match:
        return {
            "Clinical_trials": float(match.group(1)),
            "HPA": float(match.group(2)),
            "Heatmap": float(match.group(3)),
            "Vital": float(match.group(4)),
            "Non_vital": float(match.group(5)),
            "Protein_vital": float(match.group(6)),
            "Protein_non_vital": float(match.group(7))
        }
    
    # Try a more relaxed pattern if the formatted version isn't found
    weights = {}
    patterns = {
        "Clinical_trials": r"Clinical trials[:\s]*\(?\s*(\d+\.?\d*)",
        "HPA": r"HPA[:\s]*\(?\s*(\d+\.?\d*)",
        "Heatmap": r"Heatmap[:\s]*\(?\s*(\d+\.?\d*)",
        "Vital": r"Vital[:\s]*\(?\s*(\d+\.?\d*)",
        "Non_vital": r"Non-vital[:\s]*\(?\s*(\d+\.?\d*)",
        "Protein_vital": r"Protein vital[:\s]*\(?\s*(\d+\.?\d*)",
        "Protein_non_vital": r"Protein non-vital[:\s]*\(?\s*(\d+\.?\d*)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text)
        if match:
            weights[key] = float(match.group(1))
    
    # Return weights if we found at least 4 features
    if len(weights) >= 4:
        return weights
    else:
        return None

def analyze_all_models():
    """Analyze results from all three models"""
    # Load results
    logger.info("Loading results from all models...")
    openai_results, claude_results = load_existing_results()
    gemini_results = get_gemini_progress()
    
    weight_features = ["Clinical_trials", "HPA", "Heatmap", "Vital", 
                       "Non_vital", "Protein_vital", "Protein_non_vital"]
    
    # Process results for each model
    def process_results(results, model_name):
        model_weights = []
        success_count = 0
        
        for result in results:
            try:
                weights = extract_weights(result["response"])
                if weights:
                    weights["model"] = model_name
                    weights["run_id"] = result["run_id"]
                    model_weights.append(weights)
                    success_count += 1
            except Exception as e:
                logger.error(f"Error processing {model_name} result: {str(e)}")
        
        logger.info(f"Extracted weights from {success_count}/{len(results)} {model_name} responses")
        return model_weights
    
    # Extract weights from each model
    openai_weights = process_results(openai_results, "GPT-4o")
    claude_weights = process_results(claude_results, "Claude-3.7")
    gemini_weights = process_results(gemini_results, "Gemini-2.5-Pro")
    
    # Combine all weights into a DataFrame
    all_weights = pd.DataFrame(openai_weights + claude_weights + gemini_weights)
    
    # Save extracted weights
    all_weights.to_csv("results/all_models_weights.csv", index=False)
    logger.info(f"Saved extracted weights from all models. Total: {len(all_weights)} rows")
    
    # Create summary statistics
    summary = all_weights.groupby("model")[weight_features].agg(['mean', 'median', 'std', 'min', 'max'])
    summary.to_csv("results/all_models_summary_stats.csv")
    
    # Count most common weight combinations by model
    def get_weight_combo(row):
        return tuple(round(row[feature], 1) for feature in weight_features)
    
    all_weights["weight_combo"] = all_weights.apply(get_weight_combo, axis=1)
    combo_counts = all_weights.groupby(["model", "weight_combo"]).size().reset_index(name="count")
    combo_counts = combo_counts.sort_values(["model", "count"], ascending=[True, False])
    
    # Save top combinations
    top_combos = pd.DataFrame()
    for model in all_weights["model"].unique():
        model_combos = combo_counts[combo_counts["model"] == model].head(5)
        model_combos["weight_combo_str"] = model_combos["weight_combo"].apply(
            lambda x: ", ".join([f"{feat}={val}" for feat, val in zip(weight_features, x)])
        )
        top_combos = pd.concat([top_combos, model_combos])
    
    top_combos.to_csv("results/all_models_top_combinations.csv", index=False)
    
    logger.info(f"Analysis complete for all models")
    return all_weights, summary, top_combos

def create_three_model_visualizations(all_weights, summary):
    """Create visualizations comparing all three models"""
    os.makedirs("results/visualizations", exist_ok=True)
    
    # Set color palette for models
    model_colors = {
        "GPT-4o": "#74aa9c",       # OpenAI green
        "Claude-3.7": "#ff6c00",   # Anthropic orange
        "Gemini-2.5-Pro": "#4285F4" # Google blue
    }
    
    # Feature names for plotting
    weight_features = ["Clinical_trials", "HPA", "Heatmap", "Vital", 
                       "Non_vital", "Protein_vital", "Protein_non_vital"]
    
    # 1. Distribution of weights by model
    logger.info("Creating distribution boxplots...")
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(weight_features):
        plt.subplot(2, 4, i+1)
        # Use the custom palette
        sns.boxplot(x="model", y=feature, data=all_weights, 
                   palette=model_colors, order=list(model_colors.keys()))
        plt.title(f"Distribution of {feature.replace('_', ' ')} Weights", fontsize=14)
        plt.ylabel("Weight Value", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/visualizations/three_model_weight_distributions.png", dpi=300)
    
    # 2. Heatmap of average weights by model
    logger.info("Creating average weights heatmap...")
    avg_weights = all_weights.groupby("model")[weight_features].mean()
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(avg_weights, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
    plt.title("Average Weights by Model", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/visualizations/three_model_average_weights_heatmap.png", dpi=300)
    
    # 3. Radar chart for comparing models
    logger.info("Creating radar chart...")
    def radar_chart(avg_weights, weight_features, model_colors):
        # Number of variables
        N = len(weight_features)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Initialise the plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], [f.replace('_', ' ') for f in weight_features], size=14)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        max_val = avg_weights.values.max() * 1.1
        plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=12)
        plt.ylim(0, max_val)
        
        # Plot each model with custom colors
        for model in avg_weights.index:
            values = avg_weights.loc[model].values.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=3, linestyle='solid', label=model, color=model_colors.get(model, "gray"))
            ax.fill(angles, values, alpha=0.2, color=model_colors.get(model, "gray"))
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.2, 0.1), fontsize=14)
        
        plt.title("Weight Comparison Across Models", size=20, pad=20)
        return fig
    
    radar_fig = radar_chart(avg_weights, weight_features, model_colors)
    radar_fig.savefig("results/visualizations/three_model_radar_chart.png", dpi=300)
    
    # 4. Stacked bar chart showing relative importance of features
    logger.info("Creating stacked bar chart of relative importance...")
    plt.figure(figsize=(14, 8))
    
    # Convert to percentage of total weights
    weights_percent = avg_weights.copy()
    for model in weights_percent.index:
        weights_percent.loc[model] = weights_percent.loc[model] / weights_percent.loc[model].sum() * 100
    
    # Plot stacked bars
    weights_percent.plot(kind='bar', stacked=True, figsize=(14, 8), 
                        colormap='tab10', width=0.7)
    plt.title("Relative Importance of Features by Model", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Percentage of Total Weight (%)", fontsize=14)
    plt.legend(title="Feature", fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0, fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/visualizations/three_model_relative_importance.png", dpi=300)
    
    # 5. Violin plots for distribution comparison
    logger.info("Creating violin plots...")
    for feature in weight_features:
        plt.figure(figsize=(12, 8))
        sns.violinplot(x="model", y=feature, data=all_weights, 
                      palette=model_colors, order=list(model_colors.keys()))
        plt.title(f"Distribution of {feature.replace('_', ' ')} Weights", fontsize=16)
        plt.ylabel("Weight Value", fontsize=14)
        plt.xlabel("Model", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/visualizations/violin_{feature}.png", dpi=300)
    
    # 6. Density plot of key features
    logger.info("Creating density plots...")
    for feature in ["Heatmap", "HPA", "Clinical_trials"]:  # Top 3 most important features
        plt.figure(figsize=(12, 8))
        for model in all_weights["model"].unique():
            model_data = all_weights[all_weights["model"] == model]
            sns.kdeplot(model_data[feature], label=model, color=model_colors[model], shade=True, alpha=0.3)
        
        plt.title(f"Density of {feature.replace('_', ' ')} Weights", fontsize=16)
        plt.xlabel("Weight Value", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/visualizations/density_{feature}.png", dpi=300)
    
    logger.info("Visualizations complete")

def generate_comparative_report(all_weights, summary, top_combos):
    """Generate a comprehensive report comparing all three models"""
    logger.info("Generating comparative report...")
    
    report = []
    report.append("# CAR T Target Weight Recommendation - Three Model Comparison")
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Count of samples by model
    model_counts = all_weights.groupby("model").size()
    report.append("\n## Model Samples")
    report.append(f"- GPT-4o: {model_counts.get('GPT-4o', 0)} samples")
    report.append(f"- Claude-3.7: {model_counts.get('Claude-3.7', 0)} samples")
    report.append(f"- Gemini-2.5-Pro: {model_counts.get('Gemini-2.5-Pro', 0)} samples")
    
    # Add mean weights table
    report.append("\n## Average Weights by Model")
    means = all_weights.groupby("model")[["Clinical_trials", "HPA", "Heatmap", "Vital", 
                                         "Non_vital", "Protein_vital", "Protein_non_vital"]].mean()
    report.append("```")
    report.append(means.to_string())
    report.append("```")
    
    # Add coefficient of variation to gauge consistency
    report.append("\n## Model Consistency (Coefficient of Variation)")
    report.append("*Lower values indicate more consistent recommendations*")
    
    cv = all_weights.groupby("model")[["Clinical_trials", "HPA", "Heatmap", "Vital", 
                                     "Non_vital", "Protein_vital", "Protein_non_vital"]].agg(
        lambda x: x.std() / x.mean() * 100  # CV as percentage
    )
    report.append("```")
    report.append(cv.to_string())
    report.append("```")
    
    # Top 3 features by model
    report.append("\n## Top 3 Features by Model")
    
    weight_features = ["Clinical_trials", "HPA", "Heatmap", "Vital", 
                      "Non_vital", "Protein_vital", "Protein_non_vital"]
    
    for model in all_weights["model"].unique():
        report.append(f"\n### {model}")
        model_means = means.loc[model].sort_values(ascending=False)
        top3 = model_means.head(3)
        
        for feature, value in top3.items():
            report.append(f"- **{feature.replace('_', ' ')}**: {value:.2f}")
    
    # Consensus analysis
    report.append("\n## Consensus Analysis")
    
    # Get rankings from each model
    rankings = {}
    for model in all_weights["model"].unique():
        model_means = means.loc[model]
        rankings[model] = model_means.rank(ascending=False).to_dict()
    
    # Calculate average rank for each feature
    feature_ranks = {}
    for feature in weight_features:
        feature_ranks[feature] = sum(rankings[model][feature] for model in rankings) / len(rankings)
    
    # Sort by average rank
    sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1])
    
    report.append("\n### Overall Feature Ranking (By Average Rank)")
    for feature, avg_rank in sorted_features:
        report.append(f"- **{feature.replace('_', ' ')}**: {avg_rank:.1f}")
    
    # Identify clear consensus (top 2 and bottom 2 features)
    report.append("\n### Consensus Features")
    report.append("**Top 2 Most Important Features:**")
    for feature, avg_rank in sorted_features[:2]:
        report.append(f"- {feature.replace('_', ' ')}")
    
    report.append("\n**2 Least Important Features:**")
    for feature, avg_rank in sorted_features[-2:]:
        report.append(f"- {feature.replace('_', ' ')}")
    
    # Model-specific unique insights
    report.append("\n## Model-Specific Insights")
    
    # Identify features where each model differs significantly from the others
    for model in all_weights["model"].unique():
        report.append(f"\n### {model} Distinctive Emphasis")
        
        other_models = [m for m in all_weights["model"].unique() if m != model]
        other_means = all_weights[all_weights["model"].isin(other_models)].groupby("model")[weight_features].mean().mean()
        model_means = means.loc[model]
        
        # Calculate percentage difference
        pct_diff = (model_means - other_means) / other_means * 100
        
        # Find features with substantial differences (more than 25%)
        significant_diffs = [(f.replace('_', ' '), d) for f, d in pct_diff.items() if abs(d) > 25]
        significant_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if significant_diffs:
            for feature, diff in significant_diffs:
                direction = "higher" if diff > 0 else "lower"
                report.append(f"- **{feature}**: {abs(diff):.1f}% {direction} than other models")
        else:
            report.append("- No features with significantly different emphasis")
    
    # Recommendations for consensus weights
    report.append("\n## Consensus Weight Recommendation")
    
    # Calculate consensus weights (weighted by number of samples per model)
    weights = {}
    for feature in weight_features:
        numerator = sum(means.loc[model, feature] * model_counts[model] for model in means.index)
        denominator = sum(model_counts[model] for model in means.index)
        weights[feature] = numerator / denominator
    
    report.append("Based on the combined analysis of all three models, the consensus weights are:")
    for feature, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        report.append(f"- **{feature.replace('_', ' ')}**: {weight:.2f}")
    
    # Save the report
    with open("results/three_model_analysis_report.md", "w") as f:
        f.write("\n".join(report))
    
    logger.info("Comparative report generated: results/three_model_analysis_report.md")

if __name__ == "__main__":
    # Make sure directories exist
    # Set API keys from environment (or replace with direct keys)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    
    # Load gene lists from CSV files
    data_dir = "/data/ep924610/project_nb/paper_code/bulk_review/llm_prompt_prep"  # Path to your CSV files
    gene_lists = load_gene_lists_from_files(data_dir, top_n=100)
    
    # Run N queries to each API
    n_runs = 1000  # Change to 100 for full experiment
    openai_results, claude_results = run_multi_queries(n_runs)
    
    # Analyze the results
    all_weights, summary, top_combos = analyze_results(openai_results, claude_results)
    
    # Create visualizations
    create_visualizations(all_weights, summary)
    
    # Generate summary report
    generate_summary_report(all_weights, summary, top_combos)
    
    logger.info("Analysis complete! Check the 'results' directory for outputs.")
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    # Set API key from environment (or replace with direct key)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    
    # Load gene lists with reduced size to avoid context issues
    try:
        gene_lists = None
        if os.path.exists("/data/ep924610/project_nb/paper_code/bulk_review/llm_prompt_prep"):
            gene_lists = load_gene_lists_from_files("/data/ep924610/project_nb/paper_code/bulk_review/llm_prompt_prep", top_n=100)
    except Exception as e:
        logger.warning(f"Could not load gene lists: {str(e)}")
        gene_lists = None
    
    # Check if we need to run Gemini queries
    existing_gemini = get_gemini_progress()
    if len(existing_gemini) < 1000:
        # Run queries to Gemini API (1000 runs total)
        logger.info(f"Running Gemini queries (need {1000 - len(existing_gemini)} more runs)")
        gemini_results = run_gemini_queries(total_runs=1000, gene_lists=gene_lists, 
                                          batch_size=25, max_threads=1)
    else:
        logger.info("All 1000 Gemini runs already completed!")
    
    # Analyze all model results
    all_weights, summary, top_combos = analyze_all_models()
    
    # Create visualizations comparing all three models
    create_three_model_visualizations(all_weights, summary)
    
    # Generate comparative report
    generate_comparative_report(all_weights, summary, top_combos)
    
    logger.info("Analysis complete! Check the 'results' directory for outputs.")

