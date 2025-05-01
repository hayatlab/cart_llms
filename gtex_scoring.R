# Load necessary libraries
library(gtexr)
library(dplyr)
library(tidyr)

# Read the list of genes
genes_list <- read.csv("/data/ep924610/project_nb/paper_code/heatmap_results/skin_results.csv")$Gene

# Prepare a data frame to store the results
gene_expression_results <- data.frame(
  Gene = character(),
  Tissue = character(),
  MedianExpression = numeric(),
  stringsAsFactors = FALSE
)

# Iterate over each gene symbol
for (Gene in genes_list) {
  cat("Processing gene:", Gene, "\n")
  
  # Convert the gene symbol to its corresponding GENCODE ID
  gene_gencodeId <- tryCatch({
    gencode_result <- get_genes(Gene)
    if (is.data.frame(gencode_result) && "gencodeId" %in% names(gencode_result)) {
      gencode_result %>% pull(gencodeId)
    } else {
      warning("get_genes() didn't return expected structure")
      NULL
    }
  }, error = function(e) {
    warning(paste("Error getting GENCODE ID for gene:", Gene, "-", e$message))
    return(NULL)
  })

  if (length(gene_gencodeId) > 0) {
    # Get the median gene expression for the gene
    median_expression_data <- tryCatch({
      get_median_gene_expression(gencodeIds = gene_gencodeId, itemsPerPage = 500)
    }, error = function(e) {
      warning(paste("Error getting expression data for gene:", Gene, "-", e$message))
      return(NULL)
    })
    
    if (!is.null(median_expression_data) && is.data.frame(median_expression_data)) {
      # Process the data
      gene_data <- tryCatch({
        if (all(c("geneSymbol", "tissueSiteDetailId", "median") %in% names(median_expression_data))) {
          # Extract and rename needed columns
          df <- median_expression_data[, c("geneSymbol", "tissueSiteDetailId", "median")]
          df <- df[df$geneSymbol == Gene, ]
          names(df) <- c("Gene", "Tissue", "MedianExpression")
          df
        } else {
          warning("Required columns not found in expression data")
          NULL
        }
      }, error = function(e) {
        warning(paste("Error processing data for gene:", Gene, "-", e$message))
        return(NULL)
      })
      
      # Add data to results if valid
      if (!is.null(gene_data) && nrow(gene_data) > 0) {
        gene_expression_results <- bind_rows(gene_expression_results, gene_data)
        cat("Added data for gene:", Gene, "\n")
      } else {
        warning(paste("No data after filtering for gene:", Gene))
      }
    } else {
      warning(paste("No valid expression data found for gene:", Gene))
    }
  } else {
    warning(paste("No GENCODE ID found for gene:", Gene))
  }
  
  # Add a brief pause to not overwhelm the API
  Sys.sleep(0.1)
}

# Pivot the data so that rows are gene symbols and columns are tissues
# Replace spread with pivot_wider
# Pivot the data with explicit handling of duplicates
gene_expression_pivoted <- gene_expression_results %>%
  pivot_wider(
    names_from = Tissue, 
    values_from = MedianExpression,
    values_fn = mean  # Use mean for duplicates
  )

# Convert all tissue columns to numeric properly
numeric_cols <- names(gene_expression_pivoted)[-1]  # All except Gene column
gene_expression_pivoted <- gene_expression_pivoted %>%
  mutate(across(all_of(numeric_cols), as.numeric))

# Rest of your code continues...

if("geneSymbol" %in% names(gene_expression_pivoted) && !"Gene" %in% names(gene_expression_pivoted)) {
  gene_expression_pivoted <- gene_expression_pivoted %>%
    rename(Gene = geneSymbol)
}

# Convert all tissue columns to numeric (necessary for calculating the average)
gene_expression_pivoted[, -1] <- sapply(gene_expression_pivoted[, -1], as.numeric)

# Calculate the average median expression across all tissues for each gene
gene_expression_pivoted$AverageMedianExpression <- rowMeans(gene_expression_pivoted[, -1], na.rm = TRUE)

vital_cols <- grep("Lung|Heart|Brain", names(gene_expression_pivoted), 
                   ignore.case = TRUE, value = TRUE)
non_vital_cols <- setdiff(names(gene_expression_pivoted)[-1], vital_cols)

gene_expression_pivoted$VitalAverage <- rowMeans(gene_expression_pivoted[, vital_cols], na.rm = TRUE)
gene_expression_pivoted$NonVitalAverage <- rowMeans(gene_expression_pivoted[, non_vital_cols], na.rm = TRUE)

gene_expression_pivoted <- gene_expression_pivoted %>% 
  select(-AverageMedianExpression)  # Remove original average


write.csv(gene_expression_pivoted, "/data/ep924610/project_nb/paper_code/gtex_results/median_expression_results.csv", row.names = FALSE)

head(gene_expression_pivoted)

# Top 5 genes by vital average
cat("Top 5 genes by vital average:\n")
gene_expression_pivoted %>% 
  arrange(desc(VitalAverage)) %>% 
  select(Gene, VitalAverage) %>% 
  tail(5) %>% 
  print()

# Top 5 genes by non-vital average
cat("\nTop 5 genes by non-vital average:\n")
gene_expression_pivoted %>% 
  arrange(desc(NonVitalAverage)) %>% 
  select(Gene, NonVitalAverage) %>% 
  tail(5) %>% 
  print()

# GPNMB values
cat("\nGPNMB values:\n")
gene_expression_pivoted %>% 
  filter(Gene == "GPNMB") %>% 
  select(Gene, VitalAverage, NonVitalAverage) %>% 
  print()