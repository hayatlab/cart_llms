import os
import pandas as pd
import time
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
import traceback
import logging
from datetime import datetime

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

# API Key for Gemini (use environment variable or set directly)

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

class RateLimiter:
    """Simple rate limiter for Gemini API"""
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        
    def wait_if_needed(self):
        """Wait if we're exceeding the rate limit"""
        now = time.time()
        
        # Remove timestamps older than a minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # If we're at capacity, wait until we're not
        if len(self.request_times) >= self.requests_per_minute:
            oldest = min(self.request_times)
            sleep_time = max(0, 60 - (now - oldest))
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(time.time())

# Initialize rate limiter for Gemini (450 requests/minute for free tier)
gemini_limiter = RateLimiter(450)

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
#
def get_cart_prompt(gene_lists=None):
    """Get the CAR T prompt with optional gene lists"""
    cart_prompt = """As an expert in cancer biology and computational oncology, analyze these weighted scoring results for potential CAR T targets.

The scoring features include:
- Clinical Trials: Scores based on clinical trial progression (phase reached), availability of antibodies/drugs, and number of cancer-related studies
- HPA: Cell surface localization scores from Human Protein Atlas and UniProt (10=confirmed surface in both, 8=confirmed in UniProt/predicted in HPA, 7=confirmed in one source only, 5=predicted in HPA only)
- Heatmap: Z-score difference between expression in malignant vs. normal cells from single-cell data
- Vital/Non-Vital: GTEx bulk RNA expression in vital tissues (brain, lung, heart) vs. non-vital tissues
- Protein Vital/Protein Non-Vital: Protein expression levels in vital vs. non-vital tissues from ProteomicsDB

Current weights are:

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

def query_gemini(run_id, gene_lists=None, temperature=0.7):
    """Query Google Gemini API with the CAR T prompt"""
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
    
    # Apply rate limiting
    gemini_limiter.wait_if_needed()
    
    try:
        # Generate response
        response = model.generate_content(
            [
                {"role": "user", "parts": [prompt]}
            ]
        )
        
        result = response.text
        return {
            "run_id": run_id,
            "model": "Gemini-2.5-Pro",
            "temperature": temperature,
            "response": result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Gemini API Error (run {run_id}): {error_msg}")
        
        # If quota exceeded or rate limited, wait longer
        if "quota" in error_msg.lower() or "rate" in error_msg.lower():
            wait_time = 60  # Wait a minute
            logger.info(f"Rate limit or quota issue, waiting {wait_time} seconds")
            time.sleep(wait_time)
        else:
            # Regular error, shorter backoff
            time.sleep(2)
            
        return None

def run_gemini_queries(total_runs=1000, gene_lists=None, batch_size=50, max_threads=10):
    """Run multiple queries to Gemini API and collect results"""
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
    
    # Set number of parallel threads (respect rate limits)
    n_threads = min(max_threads, len(runs_to_do))
    
    # Initialize results list with existing results
    gemini_results = existing_results
    
    # Create a progress bar for the runs
    pbar = tqdm(total=len(runs_to_do), desc="Gemini Queries")
    
    # Process runs in batches
    for batch_start in range(0, len(runs_to_do), batch_size):
        batch_end = min(batch_start + batch_size, len(runs_to_do))
        batch_runs = runs_to_do[batch_start:batch_end]
        
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Submit the batch of runs
            futures = [executor.submit(query_gemini, run_id, gene_lists, 0.7 + 0.2 * (run_id % 4)/4) 
                      for run_id in batch_runs]
            
            # Process results as they complete
            for future in as_completed_with_timeout(futures, timeout=180):
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        # Save individual result
                        with open(f"results/raw/gemini_run_{result['run_id']}.json", "w") as f:
                            json.dump(result, f, indent=2)
                        pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing future: {str(e)}")
        
        # Add batch results to overall results
        gemini_results.extend(batch_results)
        
        # Save progress after each batch
        with open("results/gemini_all_results.json", "w") as f:
            json.dump(gemini_results, f, indent=2)
        
        # If we're not done, wait a bit between batches
        if batch_end < len(runs_to_do):
            logger.info(f"Completed batch {batch_start}-{batch_end}, waiting before next batch...")
            time.sleep(10)  # Wait between batches to be safe
    
    pbar.close()
    logger.info(f"Completed all {len(gemini_results)} Gemini runs")
    return gemini_results

def as_completed_with_timeout(futures, timeout=180):
    """Wrap concurrent.futures.as_completed with a timeout for each future"""
    for future in futures:
        yield future
        # Wait up to timeout seconds for future to complete
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            logger.warning(f"Future timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Future raised exception: {str(e)}")

def extract_weights(response_text):
    """Extract weight recommendations from response text"""
    # Try the formatted version first
    pattern = r"RECOMMENDED WEIGHTS:\s*Clinical trials \((\d+\.\d+)\),\s*HPA \((\d+\.\d+)\),\s*Heatmap \((\d+\.\d+)\),\s*Vital \((\d+\.\d+)\),\s*Non-vital \((\d+\.\d+)\),\s*Protein vital \((\d+\.\d+)\),\s*Protein non-vital \((\d+\.\d+)\)"
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
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    # Set API key from environment (or replace with direct key)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    
    # Load gene lists with reduced size to avoid context issues
    try:
        gene_lists = None
        if os.path.exists("/data/ep924610/project_nb/paper_code/llm_prompt_prep"):
            gene_lists = load_gene_lists_from_files("/data/ep924610/project_nb/paper_code/llm_prompt_prep", top_n=100)
    except Exception as e:
        logger.warning(f"Could not load gene lists: {str(e)}")
        gene_lists = None
    
    # Check if we need to run Gemini queries
    existing_gemini = get_gemini_progress()
    if len(existing_gemini) < 1000:
        # Run queries to Gemini API (1000 runs total)
        logger.info(f"Running Gemini queries (need {1000 - len(existing_gemini)} more runs)")
        gemini_results = run_gemini_queries(total_runs=1000, gene_lists=gene_lists, 
                                          batch_size=50, max_threads=2)
    else:
        logger.info("All 1000 Gemini runs already completed!")
    
    # Analyze all model results
    all_weights, summary, top_combos = analyze_all_models()
    
    # Create visualizations comparing all three models
    create_three_model_visualizations(all_weights, summary)
    
    # Generate comparative report
    generate_comparative_report(all_weights, summary, top_combos)
    
    logger.info("Analysis complete! Check the 'results' directory for outputs.")
