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
import google.generativeai as genai
import logging
import threading
import csv

# Set these to your API keys
OPENAI_API_KEY = "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"
ANTHROPIC_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyC1g55yOvI9wgdEnTKnjUVchxl65jMc6rw"  # Replace with your Gemini API key

# Output directory
OUTPUT_DIR = "/data/ep924610/project_nb/paper_code/new_prompt2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/raw", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/llm_queries.log", mode='a'),
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
openai_limiter = RateLimiter(tokens_per_min=28000, max_parallel_requests=3)
claude_limiter = RateLimiter(tokens_per_min=90000, max_parallel_requests=5)
gemini_limiter = RateLimiter(tokens_per_min=50000, max_parallel_requests=4)

def get_melanoma_cart_prompt():
    """Get the updated melanoma CAR T prompt that requests multiple reasons per gene"""
    melanoma_prompt = """As an expert in cancer immunotherapy and molecular oncology, analyze this list of 100 genes identified through computational screening for CAR T targets, focusing specifically on melanoma.

STRUCTURED EVALUATION REQUEST:

Review the following genes, considering the provided scoring features:
[GENES LIST: [ERBB3, RNF43, PMEL, MC1R, GPNMB, LZTS1, HPS4, IGF1R, SORT1, TNFRSF14, NRG2, MAK, ROBO1, NRG4, PTCH1, ADIPOR2, PKD2, CDH1, BLNK, TCHP, LTK, TNFSF13B, EPS8, CDH3, DDR2, LAT, FHIT, EPOR, PIGF, NKTR, BMPR2, ABCB4, MLLT10, DST, CD226, CHRNA5, NEO1, IFNAR2, MYO10, CTNS, F2RL1, ERBB2, PDCD1LG2, HTR2B, APC, TGFBR1, TRPM7, CNIH3, BCL2L12, RAD51B, EPHA2, WDPCP, PDE4D, FUT2, HCAR2, ADAM12, GPER1, PTK2, TNK2, ITGA6, FCGR2B, NPIPA1, GPR55, WLS, PPL, AXIN2, IL6R, HAS3, P2RX7, SLC22A1, RAF1, LAPTM4B, SLC38A6, BRAF, ZNRF3, RASGRP1, ATP10A, NF1, IL1RAP, PMP22, STIL, INSR, CATSPER2, CADM1, GRK4, TULP3, TMEM241, TMEM117, TRPV2, ABCA7, CTTN, SLC16A1, SLC39A4, LEPR, CRCP, SLC47A1, WDR83OS, MCOLN3, MR1, MME]
SCORING CRITERIA (Previously Used):
1. Clinical Trials: Scores based on clinical trial progression (phase reached), antibody/drug availability, and number of cancer-related studies
2. Cell Surface Localization: Human Protein Atlas and UniProt data (10=confirmed in both, 8=confirmed in UniProt/predicted in HPA, 7=confirmed in one source only, 5=predicted in HPA only)
3. Expression Difference: Z-score difference between malignant vs. normal cells from single-cell data
4. Vital vs. Non-Vital Tissue Expression: RNA expression in vital tissues (brain, lung, heart) vs. non-vital tissues
5. Protein Expression: Protein levels in vital vs. non-vital tissues from ProteomicsDB

REQUIRED OUTPUT:

Part 1: Top 10 Candidate Selection in CSV Format
Present the 10 most promising genes for CAR T development in melanoma in a clearly ranked order (1-10).

For each gene, provide:
- Rank (#1-10)
- Gene Symbol
- Brief description (1-2 sentences on function/role)
- ALL applicable nomination reasons from ONLY these five categories (list all that apply, between 1-5 reasons max):
  * High tumor-specific expression (>X% of melanoma cells)
  * Potential utility across multiple cancer types
  * Strong existing clinical development profile
  * Minimal expression in healthy vital tissues
  * Expression in significant patient subpopulations

IMPORTANT: Please use EXACTLY these five reason phrases and do not create additional reasons. Do not modify the wording.

Your output for Part 1 should be structured in a CSV-like format with the following columns:
Rank,Gene,Description,Reason1,Reason2,Reason3,Reason4,Reason5

Example:
1,GENE1,Description of gene function,High tumor-specific expression,Minimal expression in healthy vital tissues,,
2,GENE2,Description of gene function,Strong existing clinical development profile,Potential utility across multiple cancer types,Expression in significant patient subpopulations,,

Part 2: Individual Assessments
For each of the top 10 genes, provide a structured evaluation:

1. MELANOMA RELEVANCE:
   - Expression pattern in melanoma
   - Functional significance in melanoma pathology
   - Clinical correlations (if known)

2. SAFETY ASSESSMENT:
   - Expression in vital normal tissues
   - Known essential functions
   - Potential on-target/off-tumor toxicity concerns

3. TECHNICAL FEASIBILITY:
   - Cell surface localization confidence
   - Antibody/binding domain availability
   - Previous CAR development (if any)

4. BROADER APPLICABILITY:
   - Evidence for utility in other cancer types
   - Known mutations or variants affecting targetability

Base all assessments on established scientific knowledge and clinical evidence. Indicate any areas of uncertainty."""
    
    return melanoma_prompt


def query_openai(run_id, temperature=0.7):
    """Query OpenAI API with the melanoma CAR T prompt using rate limiting"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = get_melanoma_cart_prompt()
    
    # Estimate token count
    estimated_tokens = len(prompt) // 4 + 1000  # Input tokens + response buffer
    
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
                        {"role": "system", "content": "You are an expert in cancer immunotherapy, molecular oncology, and CAR T cell therapy development."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                
                result = response.choices[0].message.content
                
                # Update tokens used
                completion_tokens = response.usage.completion_tokens
                prompt_tokens = response.usage.prompt_tokens
                total_tokens = completion_tokens + prompt_tokens
                
                # Release with actual tokens used
                openai_limiter.release(total_tokens)
                
                return {
                    "run_id": run_id,
                    "model": "GPT-4o",
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
    """Query Anthropic Claude API with the melanoma CAR T prompt using rate limiting"""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = get_melanoma_cart_prompt()
    
    # Estimate token count
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
                    system="You are an expert in cancer immunotherapy, molecular oncology, and CAR T cell therapy development.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result = response.content[0].text
                
                # Estimate tokens used
                approximate_tokens = len(prompt + result) // 4
                
                # Release with estimated tokens
                claude_limiter.release(approximate_tokens)
                
                return {
                    "run_id": run_id,
                    "model": "Claude-3.7",
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

def query_gemini(run_id, temperature=0.7):
    """Query Google Gemini API with the melanoma CAR T prompt using rate limiting"""
    # Configure the Gemini API
    genai.configure(api_key=GEMINI_API_KEY)
    
    prompt = get_melanoma_cart_prompt()
    
    # Estimate token count
    estimated_tokens = len(prompt) // 4 + 1000  # Input tokens + response buffer
    
    try:
        # Acquire token allocation from rate limiter
        gemini_limiter.acquire(estimated_tokens)
        
        max_retries = 3
        backoff_time = 2
        
        for attempt in range(max_retries):
            try:
                # Initialize the model with gemini-2.5-pro-preview-03-25
                model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
                
                # Generate the content
                response = model.generate_content(
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": prompt}]
                        }
                    ],
                    generation_config={"temperature": temperature}
                )
                
                result = response.text
                
                # Estimate tokens used
                approximate_tokens = len(prompt + result) // 4
                
                # Release with estimated tokens
                gemini_limiter.release(approximate_tokens)
                
                return {
                    "run_id": run_id,
                    "model": "Gemini-2.5-Pro",
                    "temperature": temperature,
                    "response": result,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tokens_used": approximate_tokens
                }
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Gemini API Error (run {run_id}, attempt {attempt+1}/{max_retries}): {error_str}")
                
                # Handle rate limiting errors specifically
                if any(term in error_str.lower() for term in ["rate limit", "quota", "too many requests"]):
                    wait_time = backoff_time * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Rate limit hit, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                else:
                    # For other errors, shorter backoff
                    time.sleep(backoff_time)
        
        # If we get here, all retries failed
        logger.error(f"All retries failed for Gemini query (run {run_id})")
        gemini_limiter.release(0)  # Release the semaphore but don't count tokens
        return None
                
    except Exception as e:
        logger.error(f"Unexpected error in Gemini query wrapper (run {run_id}): {str(e)}")
        gemini_limiter.release(0)  # Release the semaphore but don't count tokens
        return None
    
def extract_gene_data(response_text, model_name=None, run_id=None):
    """
    Extract gene rankings and ALL applicable reasons from the response
    Returns a list of dictionaries with gene information
    Ensures each gene-reason combination is only counted once per response
    """
    gene_data = []
    
    # List of possible genes to look for
    gene_list = ["ERBB3", "RNF43", "PMEL", "MC1R", "GPNMB", "LZTS1", "HPS4", "IGF1R", 
                 "SORT1", "TNFRSF14", "NRG2", "MAK", "ROBO1", "NRG4", "PTCH1", "ADIPOR2", 
                 "PKD2", "CDH1", "BLNK", "TCHP", "LTK", "TNFSF13B", "EPS8", "CDH3", 
                 "DDR2", "LAT", "FHIT", "EPOR", "PIGF", "NKTR", "BMPR2", "ABCB4", 
                 "MLLT10", "DST", "CD226", "CHRNA5", "NEO1", "IFNAR2", "MYO10", "CTNS", 
                 "F2RL1", "ERBB2", "PDCD1LG2", "HTR2B", "APC", "TGFBR1", "TRPM7", "CNIH3", 
                 "BCL2L12", "RAD51B", "EPHA2", "WDPCP", "PDE4D", "FUT2", "HCAR2", "ADAM12", 
                 "GPER1", "PTK2", "TNK2", "ITGA6", "FCGR2B", "NPIPA1", "GPR55", "WLS", 
                 "PPL", "AXIN2", "IL6R", "HAS3", "P2RX7", "SLC22A1", "RAF1", "LAPTM4B", 
                 "SLC38A6", "BRAF", "ZNRF3", "RASGRP1", "ATP10A", "NF1", "IL1RAP", "PMP22", 
                 "STIL", "INSR", "CATSPER2", "CADM1", "GRK4", "TULP3", "TMEM241", "TMEM117", 
                 "TRPV2", "ABCA7", "CTTN", "SLC16A1", "SLC39A4", "LEPR", "CRCP", "SLC47A1", 
                 "WDR83OS", "MCOLN3", "MR1", "MME"]
    
    # List of standard nomination reasons - use EXACTLY these phrases
    standard_reasons = [
        "High tumor-specific expression",
        "Potential utility across multiple cancer types",
        "Strong existing clinical development profile",
        "Minimal expression in healthy vital tissues",
        "Expression in significant patient subpopulations"
    ]
    
    # Track what genes we've found to avoid duplicates
    found_genes = set()
    # Track gene-reason pairs to avoid counting the same reason twice for a gene
    gene_reason_pairs = set()
    
    # Check if this is a Gemini model (handles CSV differently)
    is_gemini = model_name and "Gemini" in model_name
    
    # First, try to extract data from markdown-formatted CSV (common in Gemini)
    csv_method_worked = False
    
    if is_gemini and "```csv" in response_text:
        # Extract the CSV content from markdown
        csv_matches = re.findall(r'```csv\s*(.*?)\s*```', response_text, re.DOTALL)
        if csv_matches:
            csv_content = csv_matches[0]
            # Process each line of the CSV
            for line in csv_content.strip().split('\n'):
                # Skip the header row
                if line.startswith("Rank,Gene,Description") or "Description" in line:
                    continue
                
                # Parse the CSV line
                parts = line.split(',')
                if len(parts) >= 3:  # Needs at least rank, gene, description
                    try:
                        rank_text = parts[0].strip().replace('.', '')
                        if not rank_text.isdigit():
                            continue
                            
                        rank = int(rank_text)
                        gene = parts[1].strip()
                        description = parts[2].strip()
                        
                        # Skip if we already processed this gene
                        if gene in found_genes:
                            continue
                            
                        # Make sure it's a valid gene and within top 10
                        if gene not in gene_list or rank > 10:
                            continue
                            
                        # Process each reason field
                        reasons = []
                        for i in range(3, min(8, len(parts))):
                            field = parts[i].strip() if i < len(parts) else ""
                            if field and field not in ["", ",", ",,", ",,,"]:
                                # Try to match with standard reasons
                                for std_reason in standard_reasons:
                                    # First check if the field matches a standard reason
                                    if field == std_reason or field.startswith(std_reason) or std_reason.lower() in field.lower():
                                        # Only add if we haven't already added this reason for this gene
                                        if (gene, std_reason) not in gene_reason_pairs:
                                            reasons.append(std_reason)
                                            gene_reason_pairs.add((gene, std_reason))
                                        break
                        
                        found_genes.add(gene)
                        gene_data.append({
                            "rank": rank,
                            "gene": gene,
                            "description": description,
                            "reasons": reasons,
                            "run_id": run_id
                        })
                        csv_method_worked = True
                        
                    except (ValueError, IndexError):
                        pass
    
    # If the Gemini CSV parsing didn't work, try standard CSV format
    if not csv_method_worked:
        # Try to extract data directly from CSV format lines
        csv_pattern = r'(\d+)\.?\s*,\s*([A-Z0-9]+)\s*,([^,]*?)(?:,\s*([^,\n]*?))?(?:,\s*([^,\n]*?))?(?:,\s*([^,\n]*?))?(?:,\s*([^,\n]*?))?(?:,\s*([^,\n]*?))?(?:\s*$|\n)'
        csv_matches = re.findall(csv_pattern, response_text)
        
        for match in csv_matches:
            try:
                rank_text = match[0].strip()
                if not rank_text.isdigit():
                    continue
                    
                rank = int(rank_text)
                gene = match[1].strip()
                description = match[2].strip()
                
                # Skip if we already processed this gene
                if gene in found_genes:
                    continue
                    
                # Make sure it's a valid gene and within top 10
                if gene not in gene_list or rank > 10:
                    continue
                
                # Process each potential reason field
                reasons = []
                for i in range(3, min(8, len(match))):
                    field = match[i].strip() if match[i] else ""
                    if field:
                        # Try to match with standard reasons
                        for std_reason in standard_reasons:
                            # Check if the field matches a standard reason
                            if field.startswith(std_reason) or std_reason.lower() in field.lower():
                                # Only add if we haven't already added this reason for this gene
                                if (gene, std_reason) not in gene_reason_pairs:
                                    reasons.append(std_reason)
                                    gene_reason_pairs.add((gene, std_reason))
                                break
                
                found_genes.add(gene)
                gene_data.append({
                    "rank": rank,
                    "gene": gene,
                    "description": description,
                    "reasons": reasons,
                    "run_id": run_id
                })
                csv_method_worked = True
                
            except (ValueError, IndexError):
                pass
    
    # If CSV parsing still didn't work, try a more flexible approach
    if not csv_method_worked:
        print(f"CSV parsing didn't work, using fallback extraction")
        # Look for each gene and try to find rank and reasons
        for gene in gene_list:
            # Stop if we've already found 10 genes
            if len(found_genes) >= 10:
                break
                
            # Only search for genes we haven't found yet
            if gene in found_genes:
                continue
                
            # Find all occurrences of the gene in the text
            gene_matches = list(re.finditer(rf'\b{gene}\b', response_text))
            for gene_match in gene_matches:
                gene_pos = gene_match.start()
                
                # Look for rank info within 50 chars before or 200 chars after gene mention
                search_range = max(0, gene_pos-50), min(len(response_text), gene_pos+200)
                search_text = response_text[search_range[0]:search_range[1]]
                
                # Try various rank patterns
                rank_patterns = [
                    rf'(?:Rank|#)\s*(\d+)[^A-Z]*{gene}',  # Rank: 1 GENE or #1 GENE
                    rf'^(\d+)[.)].*?{gene}',  # 1. GENE or 1) GENE at start of line
                    rf'(\d+)[.)]?\s*,\s*{gene}',  # 1, GENE (CSV format)
                    rf'{gene}\s*\(Rank\s*(\d+)\)',  # GENE (Rank 1)
                    rf'{gene}.*?ranked.*?(\d+)',  # GENE... ranked... 1
                    rf'top.*?(\d+).*?{gene}'  # top... 1... GENE
                ]
                
                rank = None
                for pattern in rank_patterns:
                    rank_match = re.search(pattern, search_text, re.MULTILINE|re.IGNORECASE)
                    if rank_match:
                        try:
                            rank = int(rank_match.group(1))
                            break
                        except (ValueError, IndexError):
                            continue
                
                if not rank or rank > 10:
                    continue  # Skip if no valid rank found or rank > 10
                
                # Look for a description - usually follows the gene mention
                description = ""
                desc_patterns = [
                    rf'{gene}\s*[:,-]\s*([^:,\n]*?)(?:$|\n|\.|Reason)',
                    rf'{gene}\s*,\s*"([^"]*?)"',  # Gene, "Description"
                    rf'{gene}.*?(?:is|as)\s+([^.]*?\.)' # Gene... is/as... Description.
                ]
                
                for pattern in desc_patterns:
                    desc_match = re.search(pattern, search_text)
                    if desc_match:
                        description = desc_match.group(1).strip()
                        break
                
                # Look for standard reasons in vicinity of gene
                reasons = []
                # Search in a wide context around the gene mention
                context_range = max(0, gene_pos-300), min(gene_pos+700, len(response_text))
                context_text = response_text[context_range[0]:context_range[1]]
                
                # First search in the context for exact reason phrases
                for reason in standard_reasons:
                    if reason in context_text and (gene, reason) not in gene_reason_pairs:
                        reasons.append(reason)
                        gene_reason_pairs.add((gene, reason))
                
                # If no exact matches, look for substrings or similar phrases
                if not reasons:
                    reason_keywords = {
                        "High tumor-specific expression": ["tumor-specific", "high expression", "overexpression", "specific expression", "highly expressed", "melanoma-specific"],
                        "Potential utility across multiple cancer types": ["multiple cancer", "other cancer", "across cancer", "utility across", "various cancer", "different cancer"],
                        "Strong existing clinical development profile": ["clinical development", "clinical trial", "antibody available", "therapeutic development", "existing clinical"],
                        "Minimal expression in healthy vital tissues": ["minimal expression", "low expression", "vital tissue", "healthy tissue", "normal organ", "limited expression"],
                        "Expression in significant patient subpopulations": ["subpopulation", "patient population", "subset of patient", "significant patient", "many patients", "patient subset"]
                    }
                    
                    for reason, keywords in reason_keywords.items():
                        if (gene, reason) not in gene_reason_pairs:  # Only consider if not already added
                            for keyword in keywords:
                                if keyword.lower() in context_text.lower():
                                    reasons.append(reason)
                                    gene_reason_pairs.add((gene, reason))
                                    break
                
                # If we found rank and gene
                if rank is not None and gene:
                    found_genes.add(gene)
                    gene_data.append({
                        "rank": rank,
                        "gene": gene,
                        "description": description,
                        "reasons": reasons,
                        "run_id": run_id
                    })
    
    # Special handling for PMEL, GPNMB, MC1R which are commonly top genes that might be missed
    frequently_missed = ["PMEL", "GPNMB", "MC1R", "ERBB3", "PDCD1LG2"]
    for gene in frequently_missed:
        if gene not in found_genes:
            # Do a more aggressive search for this gene
            gene_match = re.search(rf'(?:^|\W)({gene})(?:\W|$)', response_text)
            if gene_match:
                # Look for context around this gene
                gene_pos = gene_match.start()
                context = response_text[max(0, gene_pos-200):min(len(response_text), gene_pos+500)]
                
                # Try to find a rank
                rank_match = re.search(r'(?:Rank|#)?\s*?(\d+)[^A-Z0-9]*', context)
                rank = int(rank_match.group(1)) if rank_match else len(gene_data) + 1
                
                # Look for reasons
                reasons = []
                for reason in standard_reasons:
                    # Only add if this gene-reason pair hasn't been seen
                    if reason in context or reason.lower() in context.lower():
                        if (gene, reason) not in gene_reason_pairs:
                            reasons.append(reason)
                            gene_reason_pairs.add((gene, reason))
                
                # Add with minimal information if we have at least a rank and the gene
                if rank <= 10:
                    gene_data.append({
                        "rank": rank,
                        "gene": gene,
                        "description": "",
                        "reasons": reasons,
                        "run_id": run_id
                    })
                    found_genes.add(gene)
    
    # Sort by rank
    gene_data.sort(key=lambda x: x["rank"])
    
    return gene_data[:10]  # Return top 10 or fewer
def process_reason_stats(all_gene_data):
    """Process reason statistics, ensuring no duplicates"""
    import pandas as pd
    
    # Create a DataFrame to store the reason statistics
    reason_stats = []
    
    # Track unique gene-model-reason-run_id combinations to avoid duplicates
    unique_combinations = set()
    
    for data in all_gene_data:
        if data is not None:
            model = data.get('model', 'unknown')
            run_id = data.get('run_id', 'unknown')
            
            for gene_info in data.get('genes', []):
                gene = gene_info.get('gene', '')
                
                # Only process valid genes
                if not gene:
                    continue
                    
                for reason in gene_info.get('reasons', []):
                    # Create a unique identifier for this combination
                    unique_id = (model, gene, reason, run_id)
                    
                    # Skip if we've already seen this exact combination
                    if unique_id in unique_combinations:
                        continue
                        
                    unique_combinations.add(unique_id)
                    
                    # Add to statistics
                    reason_stats.append({
                        'model': model,
                        'gene': gene,
                        'reason': reason,
                        'count': 1,  # Count each unique occurrence exactly once
                        'run_id': run_id
                    })
    
    # Convert to DataFrame
    reason_stats_df = pd.DataFrame(reason_stats)
    
    # Save the reason statistics
    reason_stats_df.to_csv(f"{OUTPUT_DIR}/gene_reason_statistics.csv", index=False)
    
    return reason_stats_df
def run_multi_queries(n_runs=1000):
    """Run multiple queries to all three APIs and collect results with rate limiting"""
    # Check if we have existing results to resume from
    openai_results = []
    claude_results = []
    gemini_results = []
    
    # Check for existing OpenAI results
    existing_openai_files = set()
    if os.path.exists(f"{OUTPUT_DIR}/raw"):
        for filename in os.listdir(f"{OUTPUT_DIR}/raw"):
            if filename.startswith("openai_run_") and filename.endswith(".json"):
                run_id = int(filename.replace("openai_run_", "").replace(".json", ""))
                existing_openai_files.add(run_id)
    
    # Check for existing Claude results
    existing_claude_files = set()
    if os.path.exists(f"{OUTPUT_DIR}/raw"):
        for filename in os.listdir(f"{OUTPUT_DIR}/raw"):
            if filename.startswith("claude_run_") and filename.endswith(".json"):
                run_id = int(filename.replace("claude_run_", "").replace(".json", ""))
                existing_claude_files.add(run_id)
    
    # Check for existing Gemini results
    existing_gemini_files = set()
    if os.path.exists(f"{OUTPUT_DIR}/raw"):
        for filename in os.listdir(f"{OUTPUT_DIR}/raw"):
            if filename.startswith("gemini_run_") and filename.endswith(".json"):
                run_id = int(filename.replace("gemini_run_", "").replace(".json", ""))
                existing_gemini_files.add(run_id)
    
    logger.info(f"Found {len(existing_openai_files)} existing OpenAI, {len(existing_claude_files)} Claude, and {len(existing_gemini_files)} Gemini results")
    
    # Calculate how many new runs are needed for each model
    n_openai_needed = max(0, n_runs - len(existing_openai_files))
    n_claude_needed = max(0, n_runs - len(existing_claude_files))
    n_gemini_needed = max(0, n_runs - len(existing_gemini_files))
    
    logger.info(f"Need to generate {n_openai_needed} new OpenAI, {n_claude_needed} Claude, and {n_gemini_needed} Gemini results to reach {n_runs}")
    
    # Generate new unique run IDs for OpenAI
    openai_todo = set()
    current_run_id = 0
    while len(openai_todo) < n_openai_needed:
        if current_run_id not in existing_openai_files:
            openai_todo.add(current_run_id)
        current_run_id += 1
    
    # Generate new unique run IDs for Claude
    claude_todo = set()
    current_run_id = 0
    while len(claude_todo) < n_claude_needed:
        if current_run_id not in existing_claude_files:
            claude_todo.add(current_run_id)
        current_run_id += 1
    
    # Generate new unique run IDs for Gemini
    gemini_todo = set()
    current_run_id = 0
    while len(gemini_todo) < n_gemini_needed:
        if current_run_id not in existing_gemini_files:
            gemini_todo.add(current_run_id)
        current_run_id += 1
    
    logger.info(f"Generated {len(openai_todo)} OpenAI, {len(claude_todo)} Claude, and {len(gemini_todo)} Gemini run IDs to process")
    
    # Run OpenAI queries for new run_ids
    if openai_todo:
        logger.info(f"Starting {len(openai_todo)} OpenAI runs...")
        openai_pbar = tqdm(total=len(openai_todo), desc="OpenAI Queries")
        
        def process_openai(run_id):
            result = query_openai(run_id, temperature=0.7 + 0.2 * (run_id % 4) / 4)
            if result:
                # Save individual result
                with open(f"{OUTPUT_DIR}/raw/openai_run_{result['run_id']}.json", "w") as f:
                    json.dump(result, f, indent=2)
                openai_pbar.update(1)
                return result
            return None
        
        # Run in batches to better manage rate limits
        batch_size = 20
        openai_todo_list = sorted(list(openai_todo))
        
        for batch_start in range(0, len(openai_todo_list), batch_size):
            batch_end = min(batch_start + batch_size, len(openai_todo_list))
            batch = openai_todo_list[batch_start:batch_end]
            
            batch_results = []
            with ThreadPoolExecutor(max_workers=openai_limiter.max_parallel_requests) as executor:
                futures = [executor.submit(process_openai, run_id) for run_id in batch]
                for future in futures:
                    result = future.result()
                    if result:
                        batch_results.append(result)
            
            # Add successful results
            openai_results.extend(batch_results)
            
            # Short pause between batches
            if batch_end < len(openai_todo_list):
                logger.info(f"Completed OpenAI batch {batch_start}-{batch_end}, pausing briefly...")
                time.sleep(5)  # Short pause between batches
        
        openai_pbar.close()
    
    # Run Claude queries for new run_ids
    if claude_todo:
        logger.info(f"Starting {len(claude_todo)} Claude runs...")
        claude_pbar = tqdm(total=len(claude_todo), desc="Claude Queries")
        
        def process_claude(run_id):
            result = query_claude(run_id, temperature=0.7 + 0.2 * (run_id % 4) / 4)
            if result:
                # Save individual result
                with open(f"{OUTPUT_DIR}/raw/claude_run_{result['run_id']}.json", "w") as f:
                    json.dump(result, f, indent=2)
                claude_pbar.update(1)
                return result
            return None
        
        # Run in batches to better manage rate limits
        batch_size = 30  # Claude can handle more requests per batch
        claude_todo_list = sorted(list(claude_todo))
        
        for batch_start in range(0, len(claude_todo_list), batch_size):
            batch_end = min(batch_start + batch_size, len(claude_todo_list))
            batch = claude_todo_list[batch_start:batch_end]
            
            batch_results = []
            with ThreadPoolExecutor(max_workers=claude_limiter.max_parallel_requests) as executor:
                futures = [executor.submit(process_claude, run_id) for run_id in batch]
                for future in futures:
                    result = future.result()
                    if result:
                        batch_results.append(result)
            
            # Add successful results
            claude_results.extend(batch_results)
            
            # Short pause between batches
            if batch_end < len(claude_todo_list):
                logger.info(f"Completed Claude batch {batch_start}-{batch_end}, pausing briefly...")
                time.sleep(3)  # Short pause between batches
        
        claude_pbar.close()
    
    # Run Gemini queries for new run_ids
    if gemini_todo:
        logger.info(f"Starting {len(gemini_todo)} Gemini runs...")
        gemini_pbar = tqdm(total=len(gemini_todo), desc="Gemini Queries")
        
        def process_gemini(run_id):
            logger.info(f"Processing Gemini run_id: {run_id}")
            result = query_gemini(run_id, temperature=0.7 + 0.2 * (run_id % 4) / 4)
            if result:
                logger.info(f"Got successful result for Gemini run_id: {run_id}")
                # Save individual result
                with open(f"{OUTPUT_DIR}/raw/gemini_run_{result['run_id']}.json", "w") as f:
                    json.dump(result, f, indent=2)
                gemini_pbar.update(1)
                return result
            logger.warning(f"Failed to get result for Gemini run_id: {run_id}")
            return None
        
        # Run in batches to better manage rate limits
        batch_size = 25
        gemini_todo_list = sorted(list(gemini_todo))
        logger.info(f"Processing {len(gemini_todo_list)} Gemini run IDs in batches of {batch_size}")
        
        for batch_start in range(0, len(gemini_todo_list), batch_size):
            batch_end = min(batch_start + batch_size, len(gemini_todo_list))
            batch = gemini_todo_list[batch_start:batch_end]
            logger.info(f"Processing Gemini batch {batch_start}-{batch_end} with {len(batch)} run IDs")
            
            batch_results = []
            with ThreadPoolExecutor(max_workers=gemini_limiter.max_parallel_requests) as executor:
                futures = [executor.submit(process_gemini, run_id) for run_id in batch]
                for future in futures:
                    try:
                        result = future.result()
                        if result:
                            batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing Gemini batch: {str(e)}")
            
            # Add successful results
            gemini_results.extend(batch_results)
            logger.info(f"Added {len(batch_results)} Gemini results from this batch. Total so far: {len(gemini_results)}")
            
            # Short pause between batches
            if batch_end < len(gemini_todo_list):
                logger.info(f"Completed Gemini batch {batch_start}-{batch_end}, pausing briefly...")
                time.sleep(4)  # Short pause between batches
        
        gemini_pbar.close()
    
    # Load existing results
    existing_openai_results = []
    existing_claude_results = []
    existing_gemini_results = []
    
    # Load existing OpenAI results
    for run_id in existing_openai_files:
        openai_file = f"{OUTPUT_DIR}/raw/openai_run_{run_id}.json"
        try:
            with open(openai_file, "r") as f:
                existing_openai_results.append(json.load(f))
        except Exception as e:
            logger.error(f"Error loading {openai_file}: {str(e)}")
    
    # Load existing Claude results
    for run_id in existing_claude_files:
        claude_file = f"{OUTPUT_DIR}/raw/claude_run_{run_id}.json"
        try:
            with open(claude_file, "r") as f:
                existing_claude_results.append(json.load(f))
        except Exception as e:
            logger.error(f"Error loading {claude_file}: {str(e)}")
    
    # Load existing Gemini results
    for run_id in existing_gemini_files:
        gemini_file = f"{OUTPUT_DIR}/raw/gemini_run_{run_id}.json"
        try:
            with open(gemini_file, "r") as f:
                existing_gemini_results.append(json.load(f))
        except Exception as e:
            logger.error(f"Error loading {gemini_file}: {str(e)}")
    
    # Combine existing and new results
    all_openai_results = existing_openai_results + openai_results
    all_claude_results = existing_claude_results + claude_results
    all_gemini_results = existing_gemini_results + gemini_results
    
    logger.info(f"Final counts - OpenAI: {len(all_openai_results)}, Claude: {len(all_claude_results)}, Gemini: {len(all_gemini_results)}")
    
    # Save consolidated results
    with open(f"{OUTPUT_DIR}/openai_all_results.json", "w") as f:
        json.dump(all_openai_results, f, indent=2)
    
    with open(f"{OUTPUT_DIR}/claude_all_results.json", "w") as f:
        json.dump(all_claude_results, f, indent=2)
    
    with open(f"{OUTPUT_DIR}/gemini_all_results.json", "w") as f:
        json.dump(all_gemini_results, f, indent=2)
    
    return all_openai_results, all_claude_results, all_gemini_results

def debug_extraction(model_results, model_name):
    """Debug the extraction function with model outputs"""
    logger.info(f"Debugging {model_name} extraction...")
    
    for i, result in enumerate(model_results[:3]):  # Check first 3 results
        response_text = result["response"]
        logger.info(f"\nAnalyzing {model_name} response {i}:")
        
        # Print the first 500 characters to see the format
        logger.info(f"Response sample: {response_text[:500]}...")
        
        # Try to extract gene data
        gene_data = extract_gene_data(response_text, model_name)
        
        logger.info(f"Extracted {len(gene_data)} genes:")
        for gene_entry in gene_data[:3]:  # Show first 3 for brevity
            logger.info(f"  Rank {gene_entry['rank']}: {gene_entry['gene']} - Reasons: {', '.join(gene_entry['reasons'])}")
        
        # Print simple counts of keywords to help debugging
        gene_counts = {}
        for gene in ["GPNMB", "PMEL", "ERBB3", "MC1R", "CDH3", "PDCD1LG2"]:
            gene_counts[gene] = response_text.count(gene)
        
        logger.info(f"Keyword counts in response: {gene_counts}")
        
def process_model_results(model_results, model_name):
    """Process model results and save to CSV files with run_id preserved"""
    all_gene_data = []
    all_reason_data = []  # This is the new list to track individual gene-reason-run_id triplets
    
    # Process each model response
    for result in tqdm(model_results, desc=f"Processing {model_name} results"):
        try:
            run_id = result["run_id"]
            response_text = result["response"]
            
            # Extract gene data
            gene_data = extract_gene_data(response_text, model_name, run_id)
            
            # Add metadata to gene data
            for entry in gene_data:
                entry["model"] = model_name
                entry["run_id"] = run_id
                all_gene_data.append(entry)
                
                # For each reason this gene has, create a reason entry
                for reason in entry.get("reasons", []):
                    all_reason_data.append({
                        "model": model_name,
                        "run_id": run_id,
                        "gene": entry["gene"],
                        "rank": entry["rank"],
                        "reason": reason
                    })
                
        except Exception as e:
            logger.error(f"Error processing {model_name} run {result.get('run_id', 'unknown')}: {str(e)}")
    
    # Save gene data to CSV
    gene_csv_file = f"{OUTPUT_DIR}/{model_name.lower()}_gene_data.csv"
    with open(gene_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Run ID', 'Rank', 'Gene', 'Description', 'Reasons'])
        for entry in all_gene_data:
            writer.writerow([
                entry['model'],
                entry['run_id'],
                entry['rank'],
                entry['gene'],
                entry['description'],
                '|'.join(entry['reasons'])
            ])
    
    # Save reason data to CSV - preserves the gene-reason-run_id linkage
    reason_csv_file = f"{OUTPUT_DIR}/{model_name.lower()}_reason_data.csv"
    with open(reason_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Run ID', 'Gene', 'Rank', 'Reason'])
        for entry in all_reason_data:
            writer.writerow([
                entry['model'],
                entry['run_id'],
                entry['gene'],
                entry['rank'],
                entry['reason']
            ])
    
    logger.info(f"Saved {len(all_gene_data)} {model_name} gene entries to {gene_csv_file}")
    logger.info(f"Saved {len(all_reason_data)} {model_name} reason entries to {reason_csv_file}")
    
    return all_gene_data, all_reason_data

def analyze_gene_rankings(all_models_data):
    """Analyze gene rankings across all models and save summary statistics"""
    # Combine all data
    combined_data = []
    for model_data in all_models_data:
        combined_data.extend(model_data)
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame(combined_data)
    
    # Calculate gene frequency by model and rank
    gene_rank_stats = df.groupby(['model', 'gene', 'rank']).size().reset_index(name='count')
    
    # Calculate average rank per gene by model
    avg_ranks = df.groupby(['model', 'gene'])['rank'].mean().reset_index(name='avg_rank')
    
    # Calculate reason frequency by gene and model using raw reason data
    # We no longer need to extract this from the gene data since we have it directly
    # Instead, this will be passed separately to the visualization functions
    
    # Save statistics
    gene_rank_stats.to_csv(f"{OUTPUT_DIR}/gene_rank_statistics.csv", index=False)
    avg_ranks.to_csv(f"{OUTPUT_DIR}/gene_average_ranks.csv", index=False)
    
    # Create cross-model comparison
    # Top genes by frequency across all models
    top_genes = df['gene'].value_counts().head(20).index.tolist()
    
    # Create statistics for these top genes
    top_gene_stats = []
    for gene in top_genes:
        gene_data = df[df['gene'] == gene]
        for model in ['GPT-4o', 'Claude-3.7', 'Gemini-2.5-Pro']:
            model_data = gene_data[gene_data['model'] == model]
            if not model_data.empty:
                avg_rank = model_data['rank'].mean()
                frequency = len(model_data)
                
                # Get top reasons
                if 'reasons' in model_data.iloc[0]:
                    all_reasons = []
                    for reasons_list in model_data['reasons']:
                        all_reasons.extend(reasons_list)
                    reason_counts = Counter(all_reasons)
                    top_reasons = ', '.join([f"{r} ({c})" for r, c in reason_counts.most_common(3)])
                else:
                    top_reasons = "N/A"
                
                top_gene_stats.append({
                    'gene': gene,
                    'model': model,
                    'frequency': frequency,
                    'avg_rank': avg_rank,
                    'top_reasons': top_reasons
                })
    
    # Save cross-model comparison
    pd.DataFrame(top_gene_stats).to_csv(f"{OUTPUT_DIR}/cross_model_gene_comparison.csv", index=False)
    
    logger.info(f"Saved gene analysis statistics to {OUTPUT_DIR}")
    
    # Calculate aggregated reason counts for backward compatibility
    reason_counts = pd.read_csv(f"{OUTPUT_DIR}/all_reason_data.csv")
    reason_counts = reason_counts.groupby(['model', 'gene', 'reason']).size().reset_index(name='count')
    reason_counts.to_csv(f"{OUTPUT_DIR}/gene_reason_statistics.csv", index=False)
    
    return gene_rank_stats, avg_ranks, reason_counts

def create_visualizations(gene_rank_stats, avg_ranks, reason_counts, reason_data_detailed=None):
    """Create focused visualizations for gene analysis with requested specifications"""
    # Convert reason_counts to DataFrame if it's not already one
    reason_counts_df = reason_counts
    if not isinstance(reason_counts_df, pd.DataFrame):
        logger.warning("reason_counts is not a DataFrame, attempting to load from file")
        try:
            reason_counts_df = pd.read_csv(f"{OUTPUT_DIR}/gene_reason_statistics.csv")
        except Exception as e:
            logger.error(f"Failed to load reason statistics file: {str(e)}")
            # Create an empty DataFrame with the right structure
            reason_counts_df = pd.DataFrame(columns=['model', 'gene', 'reason', 'count'])
    
    # Set a more attractive custom color palette for the models
    model_colors = {
        "GPT-4o": "#74aa9c",       # OpenAI green
        "Claude-3.7": "#ff6c00",    # Anthropic orange
        "Gemini-2.5-Pro": "#4285F4" # Google blue
    }
    
    # Define models
    models = ["GPT-4o", "Claude-3.7", "Gemini-2.5-Pro"]
    
    # Standard reasons in the order we want them to appear in the heatmap
    standard_reasons = [
        "High tumor-specific expression",
        "Potential utility across multiple cancer types",
        "Strong existing clinical development profile", 
        "Minimal expression in healthy vital tissues",
        "Expression in significant patient subpopulations"
    ]
    
    # Simplified reason names for display
    short_reasons = {
        "High tumor-specific expression": "Tumor-specific",
        "Potential utility across multiple cancer types": "Multi-cancer utility",
        "Strong existing clinical development profile": "Clinical development",
        "Minimal expression in healthy vital tissues": "Low vital tissue expr",
        "Expression in significant patient subpopulations": "Patient subpops"
    }
    
    # [All your existing visualization code...]
    
    # NEW: Create a bar chart showing top genes with 100+ occurrences grouped by nomination reasons
    plt.figure(figsize=(20, 12))

    # Get all genes with at least 100 occurrences across all models
    top_genes_overall = gene_rank_stats.groupby('gene')['count'].sum()
    top_genes_100plus = top_genes_overall[top_genes_overall >= 0].index.tolist()

    if not top_genes_100plus:
        # Fallback if no genes have 100+ occurrences
        top_genes_100plus = top_genes_overall.sort_values(ascending=False).head(15).index.tolist()
        logger.warning(f"No genes with 100+ occurrences found, using top 15 genes instead: {top_genes_100plus}")

    # Filter reason data for these genes with 100+ occurrences and using standard reasons
    top_genes_reasons = reason_counts_df[
        (reason_counts_df['gene'].isin(top_genes_100plus)) & 
        (reason_counts_df['reason'].isin(standard_reasons))
    ]

    # Create separate subplots for each reason
    fig, axs = plt.subplots(len(standard_reasons), 1, figsize=(25, 5*len(standard_reasons)))

    for i, reason in enumerate(standard_reasons):
        # Filter for just this reason
        reason_data = top_genes_reasons[top_genes_reasons['reason'] == reason]
        
        # Get total counts per gene for this reason (across all models)
        gene_totals = reason_data.groupby('gene')['count'].sum().reset_index()
        gene_totals = gene_totals.sort_values('count', ascending=False)
        genes_with_data = gene_totals['gene'].tolist()
        print(f"[DEBUG] Reason: {reason} | Genes with data: {genes_with_data}")

        if not genes_with_data:
            # No data for this reason
            axs[i].text(
                0.5, 0.5, f"No genes with data for '{short_reasons.get(reason, reason)}'",
                ha='center', va='center', fontsize=14, transform=axs[i].transAxes
            )
            axs[i].set_title(f"{short_reasons.get(reason, reason)}", fontsize=14)
            continue

        # Get model-specific data for these genes
        model_data = []
        for model in models:
            model_reason_data = reason_data[reason_data['model'] == model]
            for gene in genes_with_data:
                count = model_reason_data[model_reason_data['gene'] == gene]['count'].sum()
                model_data.append({'model': model, 'gene': gene, 'count': count})

        model_df = pd.DataFrame(model_data)
        # Pivot for stacked barplot
        pivot_data = model_df.pivot(index='gene', columns='model', values='count').fillna(0)
        pivot_data = pivot_data.reindex(genes_with_data)  # preserve the order
        pivot_data = pivot_data[models]  # ensure correct model order

        print(f"[DEBUG] Pivot data for '{reason}':\n{pivot_data}")

        ax = axs[i]
        bar_container = pivot_data.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=[model_colors.get(m, '#AAAAAA') for m in models],
            width=0.8,
            edgecolor='black'
        )

        short_reason = short_reasons.get(reason, reason)
        ax.set_title(f"{short_reason}", fontsize=14)
        ax.set_ylabel("Count", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xticks(range(len(genes_with_data)))
        ax.set_xticklabels(genes_with_data, rotation=45, ha='right')

        # Add single value label per bar (total per gene)
        totals = pivot_data.sum(axis=1)
        y_max = totals.max() if len(totals) else 1
        for idx, total in enumerate(totals):
            ax.text(
                idx, total + y_max*0.02,
                str(int(total)),
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

        # Only show x-labels on bottom plot
        if i < len(standard_reasons) - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Gene", fontsize=12)
        # Add legend to the top plot only
        if i == 0:
            ax.legend(title="Model", loc='upper right')
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    plt.suptitle("Genes with 100+ Occurrences by Nomination Reason", fontsize=18, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(f"{OUTPUT_DIR}/visualizations/top_genes_by_reason.svg", dpi=300)
    plt.close()
    
    logger.info(f"Saved gene analysis visualizations to {OUTPUT_DIR}/visualizations/")
def create_reason_cooccurrence_plots(gene_rank_stats, reason_data):
    """Create co-occurrence plots using detailed run-level reason data"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from matplotlib_venn import venn2, venn3
    
    # Ensure output directory exists
    os.makedirs(f"{OUTPUT_DIR}/visualizations/cooccurrence", exist_ok=True)
    
    # Convert reason_data to DataFrame if it's not already one
    reason_df = reason_data
    if not isinstance(reason_df, pd.DataFrame):
        logger.warning("reason_data is not a DataFrame, attempting to load from file")
        try:
            reason_df = pd.read_csv(f"{OUTPUT_DIR}/all_reason_data.csv")
        except Exception as e:
            logger.error(f"Failed to load detailed reason data: {str(e)}")
            return False
    
    # Define models and reasons
    models = ["GPT-4o", "Claude-3.7", "Gemini-2.5-Pro"]
    standard_reasons = [
        "High tumor-specific expression",
        "Potential utility across multiple cancer types",
        "Strong existing clinical development profile", 
        "Minimal expression in healthy vital tissues",
        "Expression in significant patient subpopulations"
    ]
    
    # Short reason names for better display
    short_reasons = {
        "High tumor-specific expression": "Tumor-specific",
        "Potential utility across multiple cancer types": "Multi-cancer utility",
        "Strong existing clinical development profile": "Clinical development",
        "Minimal expression in healthy vital tissues": "Low vital tissue expr",
        "Expression in significant patient subpopulations": "Patient subpops"
    }
    
    # Color palette for the reasons
    reason_colors = {
        "High tumor-specific expression": "#FF9999",
        "Potential utility across multiple cancer types": "#66B2FF",
        "Strong existing clinical development profile": "#99FF99", 
        "Minimal expression in healthy vital tissues": "#FFCC99",
        "Expression in significant patient subpopulations": "#CC99FF"
    }
    
    # Process each model and gene separately
    for model in models:
        # Get top 10 genes for this model
        model_top_genes = gene_rank_stats[gene_rank_stats['model'] == model].groupby('gene')['count'].sum().sort_values(ascending=False).head(10).index.tolist()
        
        for gene in model_top_genes:
            try:
                print(f"Creating co-occurrence plot for {gene} in {model}...")
                
                # Filter the reason data for this gene-model combination
                gene_reasons = reason_df[
                    (reason_df['model'] == model) & 
                    (reason_df['gene'] == gene) &
                    (reason_df['reason'].isin(standard_reasons))
                ]
                
                if gene_reasons.empty:
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, f"No standard reasons found for {gene} in {model}", 
                          ha='center', va='center', fontsize=14)
                    plt.title(f"{gene} in {model}", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f"{OUTPUT_DIR}/visualizations/cooccurrence/{model}_{gene}.svg", dpi=300)
                    plt.close()
                    continue
                
                # Group by run_id to find which reasons co-occur within the same run
                run_to_reasons = {}
                for _, row in gene_reasons.iterrows():
                    run_id = row['run_id']
                    reason = row['reason']
                    
                    if run_id not in run_to_reasons:
                        run_to_reasons[run_id] = set()
                    
                    run_to_reasons[run_id].add(reason)
                
                # Create sets of run_ids for each reason
                reason_to_runs = {}
                for reason in standard_reasons:
                    reason_to_runs[reason] = set()
                    
                for run_id, reasons in run_to_reasons.items():
                    for reason in reasons:
                        reason_to_runs[reason].add(run_id)
                
                # Create a co-occurrence matrix
                cooccur_matrix = np.zeros((len(standard_reasons), len(standard_reasons)))
                
                # Fill the matrix
                for i, r1 in enumerate(standard_reasons):
                    for j, r2 in enumerate(standard_reasons):
                        if i == j:
                            # Diagonal - count of runs with this reason
                            cooccur_matrix[i, j] = len(reason_to_runs[r1])
                        else:
                            # Off-diagonal - count of runs with both reasons
                            cooccur_matrix[i, j] = len(reason_to_runs[r1].intersection(reason_to_runs[r2]))
                
                # Create figure with multiple panels
                fig = plt.figure(figsize=(16, 12))
                gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 0.8])
                
                # 1. Heatmap of co-occurrence matrix
                ax_heatmap = fig.add_subplot(gs[0, 0])
                sns.heatmap(
                    cooccur_matrix, 
                    annot=True, 
                    fmt='d',
                    cmap="YlGnBu",
                    xticklabels=[short_reasons[r] for r in standard_reasons],
                    yticklabels=[short_reasons[r] for r in standard_reasons],
                    ax=ax_heatmap
                )
                
                ax_heatmap.set_title(f"Reason Co-occurrence for {gene} in {model}", fontsize=14)
                ax_heatmap.set_xlabel("Reason", fontsize=12)
                ax_heatmap.set_ylabel("Reason", fontsize=12)
                
                # Improve readability
                plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha='right')
                
                # 2. Frequency bar chart
                ax_bar = fig.add_subplot(gs[0, 1])
                
                # Count frequency of each reason
                reason_counts = {r: len(runs) for r, runs in reason_to_runs.items()}
                
                # Sort by frequency
                sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
                labels = [short_reasons[r] for r, _ in sorted_reasons]
                counts = [c for _, c in sorted_reasons]
                
                bars = ax_bar.barh(labels, counts, color=[reason_colors.get(r, '#AAAAAA') for r, _ in sorted_reasons])
                
                # Add count labels
                for bar in bars:
                    width = bar.get_width()
                    ax_bar.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                             f'{int(width)}', ha='left', va='center', fontsize=10)
                
                ax_bar.set_title(f"Reason Frequencies for {gene}", fontsize=14)
                ax_bar.set_xlabel("Count (unique runs)", fontsize=12)
                ax_bar.grid(axis='x', linestyle='--', alpha=0.6)
                
                # 3. Venn diagram for top 3 reasons
                ax_venn = fig.add_subplot(gs[1, 0])
                
                # Get top 3 reasons
                top_reasons = [r for r, _ in sorted_reasons[:min(3, len(sorted_reasons))]]
                
                if len(top_reasons) >= 3:
                    try:
                        # Use venn3 for 3 sets
                        set1 = reason_to_runs[top_reasons[0]]
                        set2 = reason_to_runs[top_reasons[1]]
                        set3 = reason_to_runs[top_reasons[2]]
                        
                        venn3(
                            [set1, set2, set3],
                            [short_reasons[top_reasons[0]], short_reasons[top_reasons[1]], short_reasons[top_reasons[2]]],
                            ax=ax_venn,
                            set_colors=[reason_colors[r] for r in top_reasons]
                        )
                    except Exception as e:
                        ax_venn.text(0.5, 0.5, f"Error creating Venn diagram: {str(e)}",
                                   ha='center', va='center', fontsize=12)
                elif len(top_reasons) == 2:
                    try:
                        # Use venn2 for 2 sets
                        set1 = reason_to_runs[top_reasons[0]]
                        set2 = reason_to_runs[top_reasons[1]]
                        
                        venn2(
                            [set1, set2],
                            [short_reasons[top_reasons[0]], short_reasons[top_reasons[1]]],
                            ax=ax_venn,
                            set_colors=[reason_colors[r] for r in top_reasons]
                        )
                    except Exception as e:
                        ax_venn.text(0.5, 0.5, f"Error creating Venn diagram: {str(e)}",
                                   ha='center', va='center', fontsize=12)
                else:
                    ax_venn.text(0.5, 0.5, "Only one reason found, cannot create Venn diagram",
                               ha='center', va='center', fontsize=14)
                
                ax_venn.set_title(f"Reason Overlap for {gene} in {model}", fontsize=14)
                
                # 4. Common reason combinations
                ax_combos = fig.add_subplot(gs[1, 1])
                
                # Count how often each combination of reasons appears
                combo_counts = {}
                for reasons_set in run_to_reasons.values():
                    if len(reasons_set) > 1:  # Only interested in combinations
                        # Sort reasons to ensure consistent keys
                        combo = tuple(sorted([short_reasons[r] for r in reasons_set]))
                        combo_counts[combo] = combo_counts.get(combo, 0) + 1
                
                if combo_counts:
                    # Sort by frequency
                    sorted_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    # Display top 5 combinations
                    combo_labels = [' + '.join(combo) for combo, _ in sorted_combos[:5]]
                    combo_values = [count for _, count in sorted_combos[:5]]
                    
                    ax_combos.barh(combo_labels, combo_values, color='lightgreen')
                    
                    # Add count labels
                    for i, v in enumerate(combo_values):
                        ax_combos.text(v + 0.1, i, str(v), va='center')
                    
                    ax_combos.set_title("Common Reason Combinations", fontsize=14)
                    ax_combos.set_xlabel("Count", fontsize=12)
                else:
                    ax_combos.text(0.5, 0.5, "No reason combinations found",
                                 ha='center', va='center', fontsize=14)
                    ax_combos.set_title("Common Reason Combinations", fontsize=14)
                
                plt.tight_layout()
                plt.savefig(f"{OUTPUT_DIR}/visualizations/cooccurrence/{model}_{gene}.svg", dpi=300)
                plt.close()
                
                print(f"  Successfully created co-occurrence plot for {gene} in {model}")
                
            except Exception as e:
                logger.error(f"Error creating co-occurrence plot for {gene} in {model}: {str(e)}")
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=12)
                plt.title(f"Error: {gene} in {model}", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{OUTPUT_DIR}/visualizations/cooccurrence/{model}_{gene}.svg", dpi=300)
                plt.close()
    
    # Create a PDF with all plots
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(f"{OUTPUT_DIR}/visualizations/all_cooccurrence_plots.pdf") as pdf:
            for model in models:
                model_top_genes = gene_rank_stats[gene_rank_stats['model'] == model].groupby('gene')['count'].sum().sort_values(ascending=False).head(10).index.tolist()
                
                for gene in model_top_genes:
                    try:
                        img = plt.imread(f"{OUTPUT_DIR}/visualizations/cooccurrence/{model}_{gene}.svg")
                        plt.figure(figsize=(16, 12))
                        plt.imshow(img)
                        plt.axis('off')
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                    except Exception as e:
                        logger.error(f"Error adding {gene} in {model} to PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Error creating PDF of co-occurrence plots: {str(e)}")
    
    logger.info(f"Created co-occurrence plots for top genes in {OUTPUT_DIR}/visualizations/cooccurrence/")
    return True
def generate_summary_report(gene_rank_stats, avg_ranks, reason_counts):
    """Generate a comprehensive summary report of the gene analysis"""
    # Convert reason_counts to DataFrame if it's not already one
    reason_counts_df = reason_counts
    if not isinstance(reason_counts_df, pd.DataFrame):
        logger.warning("reason_counts is not a DataFrame in generate_summary_report, attempting to load from file")
        try:
            reason_counts_df = pd.read_csv(f"{OUTPUT_DIR}/gene_reason_statistics.csv")
        except Exception as e:
            logger.error(f"Failed to load reason statistics file: {str(e)}")
            # Create an empty DataFrame with the right structure
            reason_counts_df = pd.DataFrame(columns=['model', 'gene', 'reason', 'count'])
    
    report = []
    report.append("# Melanoma CAR T Target Analysis with Multiple Nomination Reasons")
    report.append(f"Analysis Date: {time.strftime('%Y-%m-%d')}")
    
    # Gene ranking statistics
    models = ["GPT-4o", "Claude-3.7", "Gemini-2.5-Pro"]
    
    # Top genes by model
    report.append("\n## Top 10 Genes by Model")
    
    for model in models:
        report.append(f"\n### {model}")
        
        # Get top 10 genes for this model
        model_data = gene_rank_stats[gene_rank_stats['model'] == model]
        top_genes = model_data.groupby('gene')['count'].sum().sort_values(ascending=False).head(10)
        
        report.append("| Rank | Gene | Frequency | Average Position |")
        report.append("| ---- | ---- | --------- | ---------------- |")
        
        for i, (gene, freq) in enumerate(top_genes.items(), 1):
            # Get average rank for this gene
            avg_rank = avg_ranks[(avg_ranks['model'] == model) & (avg_ranks['gene'] == gene)]['avg_rank'].values
            avg_rank_val = avg_rank[0] if len(avg_rank) > 0 else "N/A"
            
            report.append(f"| {i} | {gene} | {freq} | {avg_rank_val:.2f} |")
    
    # Reason distribution analysis
    report.append("\n## Nomination Reason Distribution")
    
    reason_list = [
        "High tumor-specific expression",
        "Potential utility across multiple cancer types",
        "Strong existing clinical development profile",
        "Minimal expression in healthy vital tissues",
        "Expression in significant patient subpopulations"
    ]
    
    for model in models:
        report.append(f"\n### {model} Reason Distribution")
        
        # Get reason counts for this model
        model_reasons = reason_counts_df[reason_counts_df['model'] == model]
        # Filter for standard reasons
        standard_reasons = model_reasons[model_reasons['reason'].isin(reason_list)]
        reason_totals = standard_reasons.groupby('reason')['count'].sum().sort_values(ascending=False)
        
        # Calculate percentages
        total_reasons = reason_totals.sum()
        
        report.append("| Reason | Count | Percentage |")
        report.append("| ------ | ----- | ---------- |")
        
        for reason, count in reason_totals.items():
            percentage = (count / total_reasons * 100) if total_reasons > 0 else 0
            report.append(f"| {reason} | {count} | {percentage:.1f}% |")
    
    # Top genes across all models
    report.append("\n## Top Genes Across All Models")
    
    # Calculate overall gene frequency
    overall_gene_freq = gene_rank_stats.groupby('gene')['count'].sum().sort_values(ascending=False).head(15)
    
    report.append("| Rank | Gene | Total Frequency | Avg Rank (GPT-4o) | Avg Rank (Claude-3.7) | Avg Rank (Gemini-2.5-Pro) |")
    report.append("| ---- | ---- | --------------- | ----------------- | --------------------- | -------------------------- |")
    
    for i, (gene, freq) in enumerate(overall_gene_freq.items(), 1):
        # Get average ranks for each model
        avg_gpt = avg_ranks[(avg_ranks['model'] == 'GPT-4o') & (avg_ranks['gene'] == gene)]['avg_rank'].values
        avg_gpt_val = f"{avg_gpt[0]:.2f}" if len(avg_gpt) > 0 else "N/A"
        
        avg_claude = avg_ranks[(avg_ranks['model'] == 'Claude-3.7') & (avg_ranks['gene'] == gene)]['avg_rank'].values
        avg_claude_val = f"{avg_claude[0]:.2f}" if len(avg_claude) > 0 else "N/A"
        
        avg_gemini = avg_ranks[(avg_ranks['model'] == 'Gemini-2.5-Pro') & (avg_ranks['gene'] == gene)]['avg_rank'].values
        avg_gemini_val = f"{avg_gemini[0]:.2f}" if len(avg_gemini) > 0 else "N/A"
        
        report.append(f"| {i} | {gene} | {freq} | {avg_gpt_val} | {avg_claude_val} | {avg_gemini_val} |")
    
    # Multiple reasons analysis
    report.append("\n## Multiple Reasons Analysis")
    report.append("\nThis section examines how often genes were assigned multiple nomination reasons")
    
    # Calculate how many genes have multiple reasons
    gene_reason_data = reason_counts_df.groupby(['model', 'gene']).size().reset_index(name='reason_count')
    
    for model in models:
        report.append(f"\n### {model}")
        
        model_data = gene_reason_data[gene_reason_data['model'] == model]
        reason_count_dist = model_data['reason_count'].value_counts().sort_index()
        
        report.append("| Number of Reasons | Count of Genes | Percentage |")
        report.append("| ----------------- | -------------- | ---------- |")
        
        total_genes = len(model_data)
        for n_reasons, count in reason_count_dist.items():
            percentage = (count / total_genes * 100) if total_genes > 0 else 0
            report.append(f"| {n_reasons} | {count} | {percentage:.1f}% |")
    
    # Reason co-occurrence
    report.append("\n## Reason Co-occurrence Analysis")
    report.append("\nThis section looks at which nomination reasons frequently appear together")
    
    # Standard reasons to focus on
    standard_reasons = [
        "High tumor-specific expression",
        "Potential utility across multiple cancer types",
        "Strong existing clinical development profile",
        "Minimal expression in healthy vital tissues",
        "Expression in significant patient subpopulations"
    ]
    
    # Function to analyze reason co-occurrence for a model
    def analyze_cooccurrence(model_name):
        try:
            # Filter data for this model and standard reasons
            model_data = reason_counts_df[(reason_counts_df['model'] == model_name) & 
                                        (reason_counts_df['reason'].isin(standard_reasons))]
            
            # For each gene, get all its reasons
            gene_to_reasons = {}
            for _, row in model_data.iterrows():
                gene = row['gene']
                reason = row['reason']
                if gene not in gene_to_reasons:
                    gene_to_reasons[gene] = []
                gene_to_reasons[gene].append(reason)
            
            # Count co-occurrences
            cooccurrence = {}
            for gene, reasons in gene_to_reasons.items():
                if len(reasons) < 2:
                    continue
                    
                for i, r1 in enumerate(reasons):
                    for r2 in reasons[i+1:]:
                        pair = tuple(sorted([r1, r2]))
                        cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
            
            return cooccurrence
        except Exception as e:
            logger.error(f"Error in analyze_cooccurrence for {model_name}: {str(e)}")
            return {}
    
    for model in models:
        report.append(f"\n### {model} Reason Co-occurrence")
        
        cooccurrence = analyze_cooccurrence(model)
        if cooccurrence:
            # Sort by frequency
            sorted_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
            
            report.append("| Reason Pair | Co-occurrence Count |")
            report.append("| ----------- | ------------------- |")
            
            for (r1, r2), count in sorted_pairs[:10]:  # Show top 10
                report.append(f"| {r1} + {r2} | {count} |")
        else:
            report.append("No significant co-occurrence patterns found")
    
    # Save the report
    with open(f"{OUTPUT_DIR}/multi_reason_gene_analysis.md", "w") as f:
        f.write("\n".join(report))
    
    logger.info(f"Summary report generated: {OUTPUT_DIR}/multi_reason_gene_analysis.md")

def process_existing_files():
    """Process existing JSON files without making new API calls"""
    openai_results = []
    claude_results = []
    gemini_results = []
    
    # Try to load existing consolidated results
    try:
        with open(f"{OUTPUT_DIR}/openai_all_results.json", "r") as f:
            openai_results = json.load(f)
        logger.info(f"Loaded {len(openai_results)} OpenAI results from consolidated file")
    except FileNotFoundError:
        logger.info("OpenAI consolidated results file not found, will look for individual files.")
        
    # Try to load existing consolidated results
    try:
        with open(f"{OUTPUT_DIR}/claude_all_results.json", "r") as f:
            claude_results = json.load(f)
        logger.info(f"Loaded {len(claude_results)} Claude results from consolidated file")
    except FileNotFoundError:
        logger.info("Claude consolidated results file not found, will look for individual files.")
        
    try:
        with open(f"{OUTPUT_DIR}/gemini_all_results.json", "r") as f:
            gemini_results = json.load(f)
        logger.info(f"Loaded {len(gemini_results)} Gemini results from consolidated file")
    except FileNotFoundError:
        logger.info("Gemini consolidated results file not found, will look for individual files.")
    
    # If consolidated files didn't exist or were empty, load from individual files
    if not openai_results or not claude_results or not gemini_results:
        logger.info("Loading from individual JSON files...")
        
        # Check for individual OpenAI files
        if not openai_results and os.path.exists(f"{OUTPUT_DIR}/raw"):
            openai_files = [f for f in os.listdir(f"{OUTPUT_DIR}/raw") if f.startswith("openai_run_") and f.endswith(".json")]
            for file in tqdm(openai_files, desc="Loading OpenAI files"):
                try:
                    with open(os.path.join(f"{OUTPUT_DIR}/raw", file), "r") as f:
                        result = json.load(f)
                        openai_results.append(result)
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
            logger.info(f"Loaded {len(openai_results)} OpenAI results from individual files")
        
        # Check for individual Claude files
        if not claude_results and os.path.exists(f"{OUTPUT_DIR}/raw"):
            claude_files = [f for f in os.listdir(f"{OUTPUT_DIR}/raw") if f.startswith("claude_run_") and f.endswith(".json")]
            for file in tqdm(claude_files, desc="Loading Claude files"):
                try:
                    with open(os.path.join(f"{OUTPUT_DIR}/raw", file), "r") as f:
                        result = json.load(f)
                        claude_results.append(result)
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
            logger.info(f"Loaded {len(claude_results)} Claude results from individual files")
        
        # Check for individual Gemini files
        if not gemini_results and os.path.exists(f"{OUTPUT_DIR}/raw"):
            gemini_files = [f for f in os.listdir(f"{OUTPUT_DIR}/raw") if f.startswith("gemini_run_") and f.endswith(".json")]
            for file in tqdm(gemini_files, desc="Loading Gemini files"):
                try:
                    with open(os.path.join(f"{OUTPUT_DIR}/raw", file), "r") as f:
                        result = json.load(f)
                        gemini_results.append(result)
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
            logger.info(f"Loaded {len(gemini_results)} Gemini results from individual files")
    
    return openai_results, claude_results, gemini_results

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/raw", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)
    
    # Set API keys from environment (or replace with direct keys)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    
    # Run N queries to each API (will automatically use existing results + make additional API calls to reach n_runs)
    n_runs = 1000
    logger.info(f"Running queries to reach {n_runs} results for each model...")
    openai_results, claude_results, gemini_results = run_multi_queries(n_runs)
    
    # Debug extraction for each model
    debug_extraction(openai_results[:3], "GPT-4o")
    debug_extraction(claude_results[:3], "Claude-3.7")
    debug_extraction(gemini_results[:3], "Gemini-2.5-Pro")
    
    # Process results for each model - now returns both gene_data and reason_data
    openai_gene_data, openai_reason_data = process_model_results(openai_results, "GPT-4o")
    claude_gene_data, claude_reason_data = process_model_results(claude_results, "Claude-3.7")
    gemini_gene_data, gemini_reason_data = process_model_results(gemini_results, "Gemini-2.5-Pro")
    
    # Combine all reason data
    all_reason_data = pd.concat([
        pd.DataFrame(openai_reason_data),
        pd.DataFrame(claude_reason_data),
        pd.DataFrame(gemini_reason_data)
    ])
    
    # Save combined reason data
    all_reason_data.to_csv(f"{OUTPUT_DIR}/all_reason_data.csv", index=False)
    
    # Analyze gene rankings
    gene_rank_stats, avg_ranks, reason_counts = analyze_gene_rankings([openai_gene_data, claude_gene_data, gemini_gene_data])
    
    # Create visualizations with both aggregated counts and detailed reason data
    create_visualizations(gene_rank_stats, avg_ranks, reason_counts, all_reason_data)
    
    # Generate summary report
    generate_summary_report(gene_rank_stats, avg_ranks, reason_counts)
    
    # Create better co-occurrence visualizations using the detailed reason data
    create_reason_cooccurrence_plots(gene_rank_stats, all_reason_data)
    
    # Run any additional analyses
    
    logger.info(f"Analysis complete! Check {OUTPUT_DIR} for results.")
    def create_upset_plots(reason_data):
        """Create upset plots showing the intersection of reasons across runs for each model and gene"""
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from upsetplot import plot as upset_plot
        from upsetplot import from_memberships
        import os
        
        # Ensure the upsetplot package is installed
        try:
            import upsetplot
        except ImportError:
            print("Error: upsetplot package not installed. Install with pip install upsetplot")
            return False
        
        # Ensure output directory exists
        os.makedirs(f"{OUTPUT_DIR}/visualizations/upset_plots", exist_ok=True)
        
        # Convert reason_data to DataFrame if it's not already one
        reason_df = pd.DataFrame(reason_data) if not isinstance(reason_data, pd.DataFrame) else reason_data
        
        # Define models and standard reasons in your specified order
        models = ["GPT-4o", "Claude-3.7", "Gemini-2.5-Pro"]
        standard_reasons = [
            "High tumor-specific expression",
            "Potential utility across multiple cancer types",
            "Strong existing clinical development profile", 
            "Minimal expression in healthy vital tissues",
            "Expression in significant patient subpopulations"
        ]
        
        # Short reason names for display
        short_reasons = {
            "High tumor-specific expression": "Tumor-specific",
            "Potential utility across multiple cancer types": "Multi-cancer utility",
            "Strong existing clinical development profile": "Clinical development",
            "Minimal expression in healthy vital tissues": "Low vital tissue expr",
            "Expression in significant patient subpopulations": "Patient subpops"
        }
        
        # For each model, create upset plots for top genes
        for model in models:
            print(f"Creating upset plots for {model}...")
            
            # Get top 10 genes by frequency for this model
            model_data = reason_df[reason_df['model'] == model]
            if model_data.empty:
                print(f"  No data found for {model}")
                continue
                
            top_genes = model_data['gene'].value_counts().head(10).index.tolist()
            
            for gene in top_genes:
                try:
                    print(f"  Processing {gene}...")
                    
                    # Get reason data for this gene
                    gene_data = model_data[model_data['gene'] == gene]
                    
                    if gene_data.empty:
                        print(f"    No reason data found for {gene}")
                        continue
                    
                    # Create a binary matrix where:
                    # - Each row is a run_id
                    # - Each column is a reason category
                    # - Value is 1 if that run_id has that reason, 0 otherwise
                    
                    # Get all unique run_ids for this gene
                    run_ids = gene_data['run_id'].unique()
                    
                    # Create an empty DataFrame with runs as rows and reasons as columns
                    matrix_data = pd.DataFrame(0, index=run_ids, columns=[short_reasons[r] for r in standard_reasons])
                    
                    # Fill in the matrix
                    for _, row in gene_data.iterrows():
                        run_id = row['run_id']
                        reason = row['reason']
                        if reason in standard_reasons:
                            matrix_data.loc[run_id, short_reasons[reason]] = 1
                    
                    # Remove any columns (reasons) that have no occurrences
                    # Remove any columns (reasons) that have no occurrences
                    matrix_data = matrix_data.loc[:, matrix_data.sum() > 0]

                    # Convert indicators to boolean (fix for upsetplot)
                    matrix_data = matrix_data.astype(bool)

                    from upsetplot import from_indicators

                    # Convert the indicator DataFrame to the required Series
                    upset_data = from_indicators(matrix_data.columns, matrix_data)

                    fig = plt.figure(figsize=(12, 8))
                    upset_plot(upset_data, fig=fig, sort_by='cardinality', show_counts=True,)
                    
                    # Add title and adjust layout
                    plt.suptitle(f"Intersection of Reasons for {gene} in {model}", fontsize=16)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
                    
                    # Save the plot
                    plt.savefig(f"{OUTPUT_DIR}/visualizations/upset_plots/{model}_{gene}_upset.svg", dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"    Created upset plot for {gene}")
                    import numpy as np
                    import matplotlib.pyplot as plt

                    # Filter to standard reasons
                    filtered = gene_data[gene_data['reason'].isin(standard_reasons)]
                    if filtered.empty:
                        print(f"    No nomination reasons available for cumulative plot for {gene}")
                    else:
                        # Sort run_ids
                        run_id_order = sorted(filtered['run_id'].unique())
                        run_id_to_x = {rid: i for i, rid in enumerate(run_id_order)}
                        n_runs = len(run_id_order)

                        # Prepare a matrix: rows=runs, cols=reasons, value=1 if present
                        reason_order = standard_reasons
                        reason_to_col = {reason: i for i, reason in enumerate(reason_order)}
                        mat = np.zeros((n_runs, len(reason_order)), dtype=int)
                        for _, row in filtered.iterrows():
                            x = run_id_to_x[row['run_id']]
                            y = reason_to_col[row['reason']]
                            mat[x, y] = 1

                        # Compute cumulative sum for each reason (column)
                        cum_mat = np.cumsum(mat, axis=0)

                        # Plot
                        fig, ax = plt.subplots(figsize=(8, 8))
                        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
                        for i, reason in enumerate(reason_order):
                            ax.plot(run_id_order, cum_mat[:, i], label=short_reasons[reason], lw=2, color=colors[i])

                        ax.set_xlabel("Run ID")
                        ax.set_ylabel("Cumulative count")
                        ax.set_title(f"Cumulative Nomination Reasons by Run: {gene} ({model})")
                        ax.legend(title="Nomination Reason", fontsize=9)
                        plt.tight_layout()
                        plt.savefig(f"{OUTPUT_DIR}/visualizations/xy_plots/{model}_{gene}_run_reason_cumulative.svg", dpi=300)
                        plt.close()
                    
                except Exception as e:
                    print(f"    Error creating upset plot for {gene}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
            print(f"Completed upset plots for {model}")
        
        # Create a summary PDF with all plots
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(f"{OUTPUT_DIR}/visualizations/all_upset_plots.pdf") as pdf:
                for model in models:
                    # Get top 10 genes for this model using the same logic as above
                    model_data = reason_df[reason_df['model'] == model]
                    if model_data.empty:
                        continue
                        
                    top_genes = model_data['gene'].value_counts().head(10).index.tolist()
                    
                    for gene in top_genes:
                        try:
                            filepath = f"{OUTPUT_DIR}/visualizations/upset_plots/{model}_{gene}_upset.svg"
                            if os.path.exists(filepath):
                                img = plt.imread(filepath)
                                plt.figure(figsize=(12, 8))
                                plt.imshow(img)
                                plt.axis('off')
                                plt.tight_layout()
                                pdf.savefig()
                                plt.close()
                        except Exception as e:
                            print(f"Error adding {gene} to PDF: {str(e)}")
                            
        except Exception as e:
            print(f"Error creating summary PDF: {str(e)}")
        
        print(f"Created upset plots in {OUTPUT_DIR}/visualizations/upset_plots/")
        return True
    create_upset_plots(all_reason_data)
    import pandas as pd
    import matplotlib.pyplot as plt
    from upsetplot import from_indicators, plot as upset_plot
    import os
    from upsetplot import UpSet
    def plot_combined_upset_top_genes(reason_df, output_dir, top_n=10):
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        from upsetplot import UpSet, from_indicators

        standard_reasons = [
            "High tumor-specific expression",
            "Potential utility across multiple cancer types",
            "Strong existing clinical development profile", 
            "Minimal expression in healthy vital tissues",
            "Expression in significant patient subpopulations"
        ]
        short_reasons = {
            "High tumor-specific expression": "Tumor-specific",
            "Potential utility across multiple cancer types": "Multi-cancer utility",
            "Strong existing clinical development profile": "Clinical development",
            "Minimal expression in healthy vital tissues": "Low vital tissue expr",
            "Expression in significant patient subpopulations": "Patient subpops"
        }

        os.makedirs(f"{output_dir}/visualizations/upset_plots", exist_ok=True)

        top_genes = (
            reason_df['gene']
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )

        for gene in top_genes:
            print(f"Creating combined UpSet plot for {gene}...")
            gene_data = reason_df[
                (reason_df['gene'] == gene) &
                (reason_df['reason'].isin(standard_reasons))
            ].copy()
            if gene_data.empty:
                print(f"  No data for {gene}")
                continue

            gene_data['run_key'] = gene_data['model'] + "_" + gene_data['run_id'].astype(str)
            indicator_df = pd.DataFrame(
                0, 
                index=gene_data['run_key'].unique(), 
                columns=[short_reasons[r] for r in standard_reasons]
            )
            for _, row in gene_data.iterrows():
                indicator_df.loc[row['run_key'], short_reasons[row['reason']]] = 1
            indicator_df['model'] = indicator_df.index.str.split('_').str[0]
            reasons_short = [
                'Tumor-specific',
                'Multi-cancer utility',
                'Clinical development',
                'Low vital tissue expr',
                'Patient subpops'
            ]
            indicator_df[reasons_short] = indicator_df[reasons_short].astype(bool)
            indicator_df['model'] = indicator_df.index.str.split('_').str[0]
            indicator_df_for_upset = indicator_df[reasons_short + ['model']].copy()

            upset_data = from_indicators(reasons_short, indicator_df_for_upset)
            fig = plt.figure(figsize=(13, 8))
            upset = UpSet(
                upset_data,
                subset_size='count',
                show_counts=True,  # <-- SHOW COUNTS ABOVE EACH BAR
                sort_by='cardinality'
            )
            upset.add_stacked_bars(by='model')
            upset.plot(fig=fig)
            plt.suptitle(f"Combined UpSet Plot for {gene} (stacked by Model)", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            outfile = f"{output_dir}/visualizations/upset_plots/ALLMODELS_{gene}_upset_stacked.svg"
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved {outfile}")
    # Example usage:
    plot_combined_upset_top_genes(all_reason_data, OUTPUT_DIR, top_n=50)
    def plot_gene_nominations_by_model_unique_runs(reason_data, output_dir, top_n=30):
        """
        Plot a grouped barplot showing all unique genes that got nominated and
        the number of unique runs in which each gene was nominated in each model.
        (one bar per model for each gene, genes on the x-axis).
        Only top_n most frequent genes (by total unique runs) are shown for clarity.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import os

        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        # Ensure DataFrame
        reason_df = reason_data if isinstance(reason_data, pd.DataFrame) else pd.DataFrame(reason_data)

        # Only count one nomination per (model, run_id, gene)
        unique_nominations = (
            reason_df
            .drop_duplicates(subset=['model', 'run_id', 'gene'])
        )

        # Count number of unique runs for each (model, gene)
        count_df = (
            unique_nominations
            .groupby(['model', 'gene'])['run_id']
            .nunique()
            .reset_index(name='unique_run_count')
        )

        # Get top_n genes by total unique runs across all models
        top_genes = (
            count_df
            .groupby('gene')['unique_run_count']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )

        # Filter for those genes
        plot_df = count_df[count_df['gene'].isin(top_genes)]

        # Order genes by total unique runs
        gene_order = (
            plot_df
            .groupby('gene')['unique_run_count']
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        model_palette = {
            "GPT-4o": "#74aa9c",
            "Claude-3.7": "#ff6c00",
            "Gemini-2.5-Pro": "#4285F4"
        }

        plt.figure(figsize=(max(12, 0.3*len(gene_order)), 7))
        ax = sns.barplot(
            data=plot_df,
            x="gene",
            y="unique_run_count",
            hue="model",
            order=gene_order,
            palette=model_palette,
            edgecolor="black"
        )
        ax.set_xlabel("Gene", fontsize=12)
        ax.set_ylabel("Number of Runs Nominated", fontsize=12)
        ax.set_title(f"Unique Run Nominations by Model (Top {top_n} Genes)", fontsize=15)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Model")
        plt.tight_layout()

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=8, padding=2)

        plot_path = f"{output_dir}/visualizations/gene_nominations_by_model_unique_runs.svg"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved barplot: {plot_path}")
    plot_gene_nominations_by_model_unique_runs(all_reason_data, OUTPUT_DIR, top_n=100)
    def plot_top_genes_stacked_barplot(reason_data, output_dir, top_n=10):
        """
        Plot a stacked barplot for the top N genes by average number of nominations,
        with stacks for each model, including only genes nominated at least once by all models.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import os

        # Make sure output directory exists
        outdir = os.path.join(output_dir, "visualizations")
        os.makedirs(outdir, exist_ok=True)

        # Convert to DataFrame if needed
        df = reason_data if isinstance(reason_data, pd.DataFrame) else pd.DataFrame(reason_data)

        # Only count one nomination per (model, run_id, gene)
        unique_noms = df.drop_duplicates(subset=['model', 'run_id', 'gene'])

        # Get the number of unique runs per (model, gene)
        counts = (
            unique_noms
            .groupby(['model', 'gene'])['run_id']
            .nunique()
            .reset_index(name='run_count')
        )

        # Find genes nominated at least once in EACH model
        models = ['GPT-4o', 'Claude-3.7', 'Gemini-2.5-Pro']
        genes_per_model = [
            set(counts[counts['model'] == m]['gene']) for m in models
        ]
        genes_in_all_models = set.intersection(*genes_per_model)
        
        # Restrict to those genes only
        counts_all = counts[counts['gene'].isin(genes_in_all_models)]

        # Get average number of nominations per gene (across models)
        avg_counts = (
            counts_all.groupby('gene')['run_count']
            .mean()
            .reset_index(name='avg_run_count')
            .sort_values('avg_run_count', ascending=False)
        )

        # Top N genes by average
        top_genes = avg_counts.head(top_n)['gene'].tolist()

        # Prepare data for stacked barplot (gene as index, columns as models)
        plot_df = (
            counts_all[counts_all['gene'].isin(top_genes)]
            .pivot(index='gene', columns='model', values='run_count')
            .fillna(0)
            .loc[top_genes]  # preserve order
        )

        # Plot
        model_palette = {
            "GPT-4o": "#74aa9c",
            "Claude-3.7": "#ff6c00",
            "Gemini-2.5-Pro": "#4285F4"
        }
        plot_df = plot_df[models]  # ensure correct column order

        ax = plot_df.plot(
            kind='bar',
            stacked=True,
            color=[model_palette[m] for m in models],
            figsize=(max(10, 0.8*len(top_genes)), 7),
            edgecolor="black"
        )
        
        ax.set_xlabel("Gene", fontsize=12)
        ax.set_ylabel("Number of Unique Runs Nominated", fontsize=12)
        ax.set_title(f"Top {top_n} Genes by Average Nominations (Stacked by Model)", fontsize=15)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Model")
        plt.tight_layout()

        # Add value labels (optional)
        # for container in ax.containers:
        #     ax.bar_label(container, fmt='%d', fontsize=8, padding=2)
        totals = plot_df.sum(axis=1)
        for i, (gene, total) in enumerate(totals.items()):
            ax.text(i, total+0.1, f"{int(total)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        outpath = os.path.join(outdir, f"top_{top_n}_genes_stacked_by_model.svg")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"Saved stacked barplot: {outpath}")
    plot_top_genes_stacked_barplot(all_reason_data, OUTPUT_DIR, top_n=15)

    def plot_simple_alluvial(
        reason_data, output_dir, top_n_genes=10, alpha_link=0.35
    ):
        import os
        import pandas as pd
        import plotly.graph_objects as go
        import matplotlib.colors as mcolors
        import numpy as np

        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        reason_df = pd.DataFrame(reason_data) if not isinstance(reason_data, pd.DataFrame) else reason_data
        dedup = reason_df.drop_duplicates(subset=['model', 'run_id', 'gene', 'reason'])

        desired_model_order = ['GPT-4o', 'Claude-3.7', 'Gemini-2.5-Pro']
        models = [m for m in desired_model_order if m in dedup['model'].unique()]
        n_models = len(models)

        gene_models = dedup.groupby('gene')['model'].nunique()
        genes_all_models = gene_models[gene_models == n_models].index.tolist()

        gene_run_count = (
            dedup[dedup['gene'].isin(genes_all_models)]
            .drop_duplicates(subset=['model', 'run_id', 'gene'])
            .groupby('gene')['run_id']
            .nunique()
            .sort_values(ascending=False)
        )
        top_genes = gene_run_count.head(top_n_genes).index.tolist()

        # --- Set custom gene order ---
        desired_gene_order = [
            "GPNMB", "EPHA2", "PMEL", "ERBB3", "IGF1R", "ERBB2", "MC1R", "CDH3", "PDCD1LG2", "ITGA6"
        ]
        genes = [g for g in desired_gene_order if g in top_genes]
        n_genes = len(genes)

        filtered = dedup[dedup['gene'].isin(genes)]

        # --- Custom reason order and display names ---
        reason_display_map = {
            "High tumor-specific expression": "Tumor specific",
            "Potential utility across multiple cancer types": "Multi cancer utility",
            "Strong existing clinical development profile": "Clinical development profile",
            "Minimal expression in healthy vital tissues": "Expression in healthy tissues",
            "Expression in significant patient subpopulations": "Patient subpopulations"
        }
        desired_reason_order = [
            "High tumor-specific expression",
            "Potential utility across multiple cancer types",
            "Strong existing clinical development profile",
            "Minimal expression in healthy vital tissues",
            "Expression in significant patient subpopulations"
        ]
        present_reasons = [r for r in desired_reason_order if r in filtered['reason'].values]
        present_reason_labels = [reason_display_map[r] for r in present_reasons]
        reason_label_lookup = dict(zip(present_reasons, present_reason_labels))
        n_reasons = len(present_reasons)

        all_nodes = models + genes + present_reason_labels
        node_indices = {name: i for i, name in enumerate(all_nodes)}

        model_colors = {
            "GPT-4o": "#74aa9c",
            "Claude-3.7": "#ff6c00",
            "Gemini-2.5-Pro": "#4285F4"
        }
        reason_colors = {
            "High tumor-specific expression": "#FF9999",
            "Potential utility across multiple cancer types": "#66B2FF",
            "Strong existing clinical development profile": "#99FF99",
            "Minimal expression in healthy vital tissues": "#FFCC99",
            "Expression in significant patient subpopulations": "#CC99FF"
        }
        def with_alpha(hex_color, alpha):
            rgb = mcolors.to_rgb(hex_color)
            return f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{alpha})'

        # node_colors = (
        #     [model_colors.get(m, '#CCCCCC') for m in models] +
        #     ['#C0C0C0'] * len(genes) +
        #     [reason_colors.get(r, '#AAAAAA') for r in present_reasons]
        # )
        node_colors = (
            [model_colors.get(m, '#CCCCCC') for m in models] +
            ['white'] * len(genes) +
            [reason_colors.get(r, '#AAAAAA') for r in present_reasons]
        )
        sources, targets, values, link_colors = [], [], [], []

        # Model→Gene links
        mg_links = (
            filtered
            .drop_duplicates(subset=['model', 'run_id', 'gene'])
            .groupby(['model', 'gene'])
            .size()
            .reset_index(name='count')
        )
        for _, row in mg_links.iterrows():
            sources.append(node_indices[row['model']])
            targets.append(node_indices[row['gene']])
            values.append(row['count'])
            color = with_alpha(model_colors.get(row['model'], "#999999"), alpha_link)
            link_colors.append(color)

        # Gene→Reason links (use display label for target)
        gr_links = (
            filtered
            .drop_duplicates(subset=['gene', 'run_id', 'reason'])
            .groupby(['gene', 'reason'])
            .size()
            .reset_index(name='count')
        )
        for _, row in gr_links.iterrows():
            sources.append(node_indices[row['gene']])
            display_reason = reason_label_lookup[row['reason']]
            targets.append(node_indices[display_reason])
            values.append(row['count'])
            color = with_alpha(reason_colors.get(row['reason'], "#BBBBBB"), alpha_link)
            link_colors.append(color)

        # Arrange node positions manually for symmetry and centering
        def centered_spread(n, y_min=0.1, y_max=0.9):
            if n == 1:
                return [0.5]
            return np.linspace(y_min, y_max, n)

        model_x = [0.08] * n_models
        model_y = centered_spread(n_models, y_min=0.15, y_max=0.85)

        gene_x = [0.5] * n_genes
        gene_y = centered_spread(n_genes, y_min=0.07, y_max=0.93)

        reason_x = [0.92] * n_reasons
        reason_y = centered_spread(n_reasons, y_min=0.15, y_max=0.85)

        node_x = list(model_x) + list(gene_x) + list(reason_x)
        node_y = list(model_y) + list(gene_y) + list(reason_y)

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=18,
                        thickness=42,
                        line=dict(color="black", width=1.2),
                        label=all_nodes,
                        color=node_colors,
                        x=node_x,
                        y=node_y,
                        hovertemplate='%{label}<extra></extra>',
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                    ),
                    textfont=dict(size=21, color="black"),
                )
            ]
        )

        fig.update_layout(
            title_text=f"Model → Gene → Reason Alluvial (Top {top_n_genes} Genes, intersection)",
            font_size=24,
            font_color="black",
            margin=dict(l=60, r=60, t=80, b=40),
            plot_bgcolor='white',
            width=2000,
            height=900
        )
        
        html_path = f"{output_dir}/visualizations/model_gene_reason_alluvial_simple.html"
        fig.write_html(html_path)
        print(f"Interactive alluvial plot saved: {html_path}")

        try:
            png_path = f"{output_dir}/visualizations/model_gene_reason_alluvial_simple.svg"
            fig.write_image(png_path, scale=2)
            print(f"Static PNG saved: {png_path}")
        except Exception as e:
            print("Static PNG could not be saved (install kaleido for static export).")

        return fig
    plot_simple_alluvial(all_reason_data, OUTPUT_DIR, top_n_genes=10)
    def make_gene_nominations_csv(reason_data, output_path):
        """
        Create a CSV with:
        - Rows: genes
        - Columns: number of nominations in each model
        - Columns: number of times each reason was given for the gene
        - average_nominations: average nominations per model
        - total_nominations: total nominations summed across models
        """
        import pandas as pd

        df = pd.DataFrame(reason_data) if not isinstance(reason_data, pd.DataFrame) else reason_data.copy()

        # Unique models and reasons
        models = sorted(df['model'].unique())
        reasons = sorted(df['reason'].unique())

        # Count #nominations per gene per model (unique run_id)
        gene_model_counts = (
            df.drop_duplicates(subset=['model', 'run_id', 'gene'])
            .groupby(['gene', 'model'])['run_id']
            .nunique()
            .unstack(fill_value=0)
        )

        # Make sure all models are present as columns
        for model in models:
            if model not in gene_model_counts.columns:
                gene_model_counts[model] = 0
        gene_model_counts = gene_model_counts[models]  # sort columns

        # Total and average nominations
        gene_model_counts['total_nominations'] = gene_model_counts.sum(axis=1)
        gene_model_counts['average_nominations'] = gene_model_counts[models].mean(axis=1)

        # For each reason, count number of times it was given for each gene (across all runs/models)
        gene_reason_counts = (
            df.drop_duplicates(subset=['gene', 'run_id', 'reason'])
            .groupby(['gene', 'reason'])['run_id']
            .nunique()
            .unstack(fill_value=0)
        )
        # Make sure all reasons are present as columns
        for reason in reasons:
            if reason not in gene_reason_counts.columns:
                gene_reason_counts[reason] = 0
        gene_reason_counts = gene_reason_counts[reasons]

        # Combine into one DataFrame
        out = pd.concat([gene_model_counts, gene_reason_counts], axis=1)
        out = out.reset_index()

        # Save to CSV
        out.to_csv(output_path, index=False)
        print(f"CSV saved: {output_path}")
        return out
    make_gene_nominations_csv(all_reason_data, "/data/ep924610/project_nb/paper_code/new_prompt2/gene_nominations_summary.csv")
    def plot_reason_stackedbar_error_top(reason_data, output_dir):
        """
        Plot a stacked barplot per model (GPT-4o, Claude-3.7, Gemini-2.5-Pro)
        showing the percentage of times each nomination reason was given,
        with error bars (std) at the top of each stack.
        """

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import os

        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)

        # Standard order and color for reasons
        reason_order = [
            "High tumor-specific expression",
            "Potential utility across multiple cancer types",
            "Strong existing clinical development profile",
            "Minimal expression in healthy vital tissues",
            "Expression in significant patient subpopulations"
        ]
        reason_colors = {
            "High tumor-specific expression": "#264653",
            "Potential utility across multiple cancer types": "#2a9d8f",
            "Strong existing clinical development profile": "#e76f51",
            "Minimal expression in healthy vital tissues": "#f4a261",
            "Expression in significant patient subpopulations": "#e9c46a"
        }
        reason_labels = {
            "High tumor-specific expression": "Tumor-specific",
            "Potential utility across multiple cancer types": "Multi-cancer utility",
            "Strong existing clinical development profile": "Clinical development",
            "Minimal expression in healthy vital tissues": "Low vital expr.",
            "Expression in significant patient subpopulations": "Patient subpops"
        }

        model_order = ["GPT-4o", "Claude-3.7", "Gemini-2.5-Pro"]

        # Prepare data: for each model, for each run, what % of nominations were each reason?
        df = pd.DataFrame(reason_data) if not isinstance(reason_data, pd.DataFrame) else reason_data
        # Only want one per (model, run_id, gene, reason)
        df = df.drop_duplicates(subset=['model', 'run_id', 'gene', 'reason'])

        # Count for each (model, run_id, reason) how many times that reason was given in that run
        counts = (
            df.groupby(['model', 'run_id', 'reason'])
            .size()
            .reset_index(name='count')
        )

        # For each (model, run_id), total number of reasons given (for normalization)
        total_per_run = (
            counts.groupby(['model', 'run_id'])['count']
            .sum()
            .reset_index(name='total')
        )

        # Merge total into counts to compute per-run percentage
        counts = counts.merge(total_per_run, on=['model', 'run_id'])
        counts['pct'] = counts['count'] / counts['total'] * 100

        # For (model, reason): get all per-run percentages
        pct_table = (
            counts.groupby(['model', 'reason'])
            .agg(
                mean_pct=('pct', 'mean'),
                std_pct=('pct', 'std'),
                n_runs=('pct', 'count')
            )
            .reset_index()
        )

        # For stacked bar, need mean_pct and std_pct in correct model and reason order
        mean_matrix = np.zeros((len(model_order), len(reason_order)))
        std_matrix = np.zeros((len(model_order), len(reason_order)))

        for i, model in enumerate(model_order):
            for j, reason in enumerate(reason_order):
                row = pct_table[
                    (pct_table['model'] == model) &
                    (pct_table['reason'] == reason)
                ]
                if not row.empty:
                    mean_matrix[i, j] = row['mean_pct'].values[0]
                    std_matrix[i, j] = row['std_pct'].values[0] if not np.isnan(row['std_pct'].values[0]) else 0
                else:
                    mean_matrix[i, j] = 0
                    std_matrix[i, j] = 0

        # Plot!
        x = np.arange(len(model_order))
        fig, ax = plt.subplots(figsize=(11, 7))
        bottoms = np.zeros(len(model_order))

        for j, reason in enumerate(reason_order):
            means = mean_matrix[:, j]
            stds = std_matrix[:, j]
            color = reason_colors[reason]
            bars = ax.bar(
                x, means, bottom=bottoms, color=color,
                label=reason_labels[reason], edgecolor="black", linewidth=0.7
            )
            bar_tops = bottoms + means
            ax.errorbar(
                x, bar_tops, yerr=stds, fmt='none', ecolor='black', capsize=6, lw=1.3, alpha=0.8, zorder=5
            )
            bottoms = bar_tops

        ax.set_xticks(x)
        ax.set_xticklabels(model_order, fontsize=13)
        ax.set_ylabel("Nomination Reason (% of all reasons per run)", fontsize=14)
        ax.set_xlabel("Model", fontsize=14)
        ax.set_title("Distribution of Nomination Reasons by Model\n(error bars: run-to-run std)", fontsize=16)
        ax.legend(title="Nomination Reason", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.tight_layout()
        plot_path = f"{output_dir}/visualizations/reason_stackedbar_error_top.svg"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Stacked bar plot of nomination reasons saved to {plot_path}")
    plot_reason_stackedbar_error_top(all_reason_data, OUTPUT_DIR)
    logger.info(f"Analysis complete! Check {OUTPUT_DIR} for results.")

