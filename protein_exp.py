import pandas as pd
import requests
import time
from tqdm import tqdm

# Step 1: Read your CSV file
df = pd.read_csv('/data/ep924610/project_nb/paper_code/heatmap_results/skin_results.csv')
gene_symbols = df['Gene'].unique().tolist()

# Step 2: Convert gene symbols to UniProt IDs
def get_uniprot_id(gene_symbol):
    """Convert a gene symbol to UniProt ID using UniProt API"""
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_symbol}+AND+reviewed:true+AND+organism_id:9606"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['primaryAccession']
    return None

# Create a dictionary mapping gene symbols to UniProt IDs
print("Converting gene symbols to UniProt IDs...")
uniprot_dict = {}
for gene in tqdm(gene_symbols):
    uniprot_id = get_uniprot_id(gene)
    if uniprot_id:
        uniprot_dict[gene] = uniprot_id
    time.sleep(0.5)  # Avoid overwhelming the API

# Step 3: Query protein expression API for each protein
def get_protein_expression(uniprot_id):
    """Get protein expression data for a given UniProt ID"""
    base_url = "https://www.proteomicsdb.org/proteomicsdb/logic/api/proteinexpression.xsodata"
    params = {
        "PROTEINFILTER": uniprot_id,
        "MS_LEVEL": 1,
        "TISSUE_ID_SELECTION": "",
        "TISSUE_CATEGORY_SELECTION": "tissue;fluid",
        "SCOPE_SELECTION": 1,
        "GROUP_BY_TISSUE": 1,
        "CALCULATION_METHOD": 0,
        "EXP_ID": -1
    }
    
    query = f"/InputParams(PROTEINFILTER='{params['PROTEINFILTER']}',MS_LEVEL={params['MS_LEVEL']},TISSUE_ID_SELECTION='{params['TISSUE_ID_SELECTION']}',TISSUE_CATEGORY_SELECTION='{params['TISSUE_CATEGORY_SELECTION']}',SCOPE_SELECTION={params['SCOPE_SELECTION']},GROUP_BY_TISSUE={params['GROUP_BY_TISSUE']},CALCULATION_METHOD={params['CALCULATION_METHOD']},EXP_ID={params['EXP_ID']})/Results"
    url = base_url + query + "?$select=UNIQUE_IDENTIFIER,TISSUE_ID,TISSUE_NAME,NORMALIZED_INTENSITY&$format=json"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# Step 4: Calculate average expression scores
vital_tissues = ['brain', 'heart', 'lung']
results = []

print("Fetching protein expression data...")
for gene, uniprot_id in tqdm(uniprot_dict.items()):
    expression_data = get_protein_expression(uniprot_id)
    if not expression_data:
        continue
        
    vital_expressions = []
    non_vital_expressions = []
    
    for entry in expression_data.get('d', {}).get('results', []):
        tissue_name = entry.get('TISSUE_NAME', '').lower()
        expression = entry.get('NORMALIZED_INTENSITY', 0)
        
        if any(vital in tissue_name for vital in vital_tissues):
            vital_expressions.append(expression)
        else:
            non_vital_expressions.append(expression)
    
    avg_vital = sum(vital_expressions) / len(vital_expressions) if vital_expressions else 0
    avg_non_vital = sum(non_vital_expressions) / len(non_vital_expressions) if non_vital_expressions else 0
    
    results.append({
        'Gene': gene,
        'UniProt_ID': uniprot_id,
        'Avg_Vital_Expression': avg_vital,
        'Avg_NonVital_Expression': avg_non_vital,
        'Vital_NonVital_Ratio': avg_vital / avg_non_vital if avg_non_vital > 0 else 0
    })
    
    time.sleep(1)  # Avoid overwhelming the API

# Create a DataFrame with the results
results_df = pd.DataFrame(results)
results_df.to_csv('/data/ep924610/project_nb/paper_code/protein_expression/protein_expression_results.csv', index=False)
print("Done! Results saved to protein_expression_results.csv")