# Function to quietly install and load packages
install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing package:", pkg, "\n")
      
      # Check if it's a Bioconductor package
      bioc_packages <- c("TCGAbiolinks", "SummarizedExperiment", "DESeq2")
      
      if (pkg %in% bioc_packages) {
        # Install Bioconductor packages
        if (!requireNamespace("BiocManager", quietly = TRUE)) {
          install.packages("BiocManager", repos = "https://cran.r-project.org/", 
                         quiet = TRUE, dependencies = TRUE)
        }
        BiocManager::install(pkg, quiet = TRUE, ask = FALSE, update = FALSE)
      } else {
        # Install CRAN packages
        install.packages(pkg, repos = "https://cran.r-project.org/", 
                        quiet = TRUE, dependencies = TRUE)
      }
      
      # Load the package
      library(pkg, character.only = TRUE, quietly = TRUE)
    }
  }
}

# List of required packages
required_packages <- c(
  "TCGAbiolinks",
  "SummarizedExperiment", 
  "DESeq2",
  "dplyr",
  "ggplot2",
  "readr",
  "data.table"
)

# Install and load all required packages
cat("Checking and installing required packages...\n")
install_and_load(required_packages)
cat("All packages loaded successfully!\n")

# Set working directory
output_dir <- "/data/ep924610/project_nb/paper_code/bulk_review"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
setwd(output_dir)

# Download TCGA SKCM data
cat("Downloading TCGA SKCM data...\n")
query_skcm <- GDCquery(
  project = "TCGA-SKCM",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",
  sample.type = c("Primary Tumor", "Solid Tissue Normal")
)

# Download the data
GDCdownload(query_skcm)
skcm_data <- GDCprepare(query_skcm)

# Check sample types in SKCM
sample_info <- colData(skcm_data)
table(sample_info$sample_type)

# Check if we have normal samples in SKCM
normal_skcm <- sum(sample_info$sample_type == "Solid Tissue Normal")
cat("Number of normal samples in SKCM:", normal_skcm, "\n")

# Function to read GTEx GCT file
read_gtex_gct <- function(file_path) {
  cat("Reading GTEx data from:", file_path, "\n")
  
  # Read the file, skipping the first two lines (GCT format header)
  gtex_data <- fread(file_path, skip = 2, header = TRUE)
  
  # Extract gene information and expression data
  gene_info <- gtex_data[, 1:2]  # First two columns: gene_id and gene_name
  expression_data <- gtex_data[, -c(1:2)]  # Remaining columns are samples
  
  # Set gene names as row names
  gene_names <- gene_info$Name
  expression_matrix <- as.matrix(expression_data)
  rownames(expression_matrix) <- gene_names
  
  return(expression_matrix)
}

# Load GTEx skin data
gtex_file <- file.path(output_dir, "gene_reads_v10_skin_sun_exposed_lower_leg.gct.gz")

if(file.exists(gtex_file)) {
  cat("Loading GTEx skin data...\n")
  gtex_expression <- read_gtex_gct(gtex_file)
  cat("GTEx data loaded. Dimensions:", nrow(gtex_expression), "genes x", ncol(gtex_expression), "samples\n")
  
  # Get SKCM tumor samples
  tumor_samples <- skcm_data[, skcm_data$sample_type == "Primary Tumor"]
  skcm_expression <- assay(tumor_samples)
  
  cat("SKCM tumor data dimensions:", nrow(skcm_expression), "genes x", ncol(skcm_expression), "samples\n")
  
  # Find common genes between SKCM and GTEx
  # Extract ENSEMBL gene IDs from SKCM data (they might have version numbers)
  skcm_gene_ids <- rownames(skcm_expression)
  gtex_gene_ids <- rownames(gtex_expression)
  
  # Remove version numbers from ENSEMBL IDs if present
  skcm_gene_ids_clean <- gsub("\\..*", "", skcm_gene_ids)
  gtex_gene_ids_clean <- gsub("\\..*", "", gtex_gene_ids)
  
  # Find common genes
  common_genes <- intersect(skcm_gene_ids_clean, gtex_gene_ids_clean)
  cat("Number of common genes:", length(common_genes), "\n")
  
  if(length(common_genes) < 1000) {
    cat("Warning: Very few common genes found. Checking gene ID formats...\n")
    cat("SKCM gene ID examples:", head(skcm_gene_ids), "\n")
    cat("GTEx gene ID examples:", head(gtex_gene_ids), "\n")
  }
  
  # Match and subset data
  skcm_match_idx <- match(common_genes, skcm_gene_ids_clean)
  gtex_match_idx <- match(common_genes, gtex_gene_ids_clean)
  
  # Remove NAs
  valid_idx <- !is.na(skcm_match_idx) & !is.na(gtex_match_idx)
  skcm_match_idx <- skcm_match_idx[valid_idx]
  gtex_match_idx <- gtex_match_idx[valid_idx]
  common_genes <- common_genes[valid_idx]
  
  # Subset expression matrices
  skcm_subset <- skcm_expression[skcm_match_idx, ]
  gtex_subset <- gtex_expression[gtex_match_idx, ]
  
  # Ensure row names match
  rownames(skcm_subset) <- common_genes
  rownames(gtex_subset) <- common_genes
  
  # Convert GTEx data to integer counts (round if necessary)
  gtex_subset <- round(gtex_subset)
  mode(gtex_subset) <- "integer"
  
  # Combine datasets
  combined_counts <- cbind(skcm_subset, gtex_subset)
  
  # Create sample metadata
  tumor_meta <- data.frame(
    sample_id = colnames(skcm_subset),
    condition = "Tumor",
    stringsAsFactors = FALSE
  )
  
  normal_meta <- data.frame(
    sample_id = colnames(gtex_subset),
    condition = "Normal",
    stringsAsFactors = FALSE
  )
  
  sample_metadata <- rbind(tumor_meta, normal_meta)
  rownames(sample_metadata) <- sample_metadata$sample_id
  
  cat("Final combined dataset dimensions:", nrow(combined_counts), "genes x", ncol(combined_counts), "samples\n")
  
} else {
  cat("GTEx file not found. Using SKCM data only.\n")
  # Use SKCM data only
  combined_counts <- assay(skcm_data)
  sample_metadata <- data.frame(
    sample_id = colnames(skcm_data),
    condition = ifelse(skcm_data$sample_type == "Primary Tumor", "Tumor", "Normal"),
    stringsAsFactors = FALSE
  )
  rownames(sample_metadata) <- sample_metadata$sample_id
}

cat("Final dataset: Tumor samples =", sum(sample_metadata$condition == "Tumor"), 
    ", Normal samples =", sum(sample_metadata$condition == "Normal"), "\n")

# Remove genes with low counts (keep genes with at least 10 reads in at least 10% of samples)
min_samples <- ceiling(0.1 * ncol(combined_counts))
keep_genes <- rowSums(combined_counts >= 10) >= min_samples
combined_counts <- combined_counts[keep_genes, ]

cat("Number of genes after filtering:", nrow(combined_counts), "\n")

# Ensure counts are integers
combined_counts <- round(combined_counts)
mode(combined_counts) <- "integer"

# Create DESeq2 object
dds <- DESeqDataSetFromMatrix(
  countData = combined_counts,
  colData = sample_metadata,
  design = ~ condition
)

# Set reference level
dds$condition <- relevel(dds$condition, ref = "Normal")

# Run DESeq2 analysis
cat("Running DESeq2 analysis...\n")
dds <- DESeq(dds)

# Get results without any filtering
results_all <- results(dds, 
                      contrast = c("condition", "Tumor", "Normal"),
                      alpha = 0.99,  # No significance cutoff
                      lfcThreshold = 0,  # No fold change cutoff
                      cooksCutoff = FALSE,
                      independentFiltering = FALSE)

# Convert to data frame and remove NAs
results_df <- as.data.frame(results_all)
results_df <- results_df[complete.cases(results_df), ]

# Add gene symbols
results_df$gene_id <- rownames(results_df)

# Calculate z-score of log fold changes
results_df$log2FC_zscore <- scale(results_df$log2FoldChange)[, 1]

# Order by z-score (descending)
results_df <- results_df[order(results_df$log2FC_zscore, decreasing = TRUE), ]

# Select columns for output
output_df <- results_df[, c("gene_id", "log2FoldChange", "log2FC_zscore", "pvalue", "padj")]
colnames(output_df) <- c("Gene_ID", "Log2FoldChange", "Log2FC_ZScore", "PValue", "AdjustedPValue")

# Save results to CSV
write_csv(output_df, file.path(output_dir, "SKCM_DEG_results_with_zscores.csv"))

cat("Results saved to:", file.path(output_dir, "SKCM_DEG_results_with_zscores.csv"), "\n")
cat("Total genes analyzed:", nrow(output_df), "\n")
cat("Z-score range:", round(min(output_df$Log2FC_ZScore), 3), "to", round(max(output_df$Log2FC_ZScore), 3), "\n")

# Create volcano plot
cat("Creating volcano plot...\n")

# Prepare data for volcano plot
volcano_data <- results_df
volcano_data$significant <- ifelse(volcano_data$padj < 0.05 & abs(volcano_data$log2FoldChange) > 1, 
                                  "Significant", "Not Significant")
volcano_data$neg_log10_pval <- -log10(volcano_data$pvalue)

# Handle infinite values
volcano_data$neg_log10_pval[is.infinite(volcano_data$neg_log10_pval)] <- max(volcano_data$neg_log10_pval[!is.infinite(volcano_data$neg_log10_pval)]) + 10

# Create volcano plot
volcano_plot <- ggplot(volcano_data, aes(x = log2FoldChange, y = neg_log10_pval)) +
  geom_point(aes(color = significant), alpha = 0.6, size = 0.8) +
  scale_color_manual(values = c("Not Significant" = "gray", "Significant" = "red")) +
  labs(
    title = "Volcano Plot: TCGA SKCM Tumor vs GTEx Normal Skin",
    x = "Log2 Fold Change",
    y = "-Log10(P-value)",
    color = "Significance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 12),
    legend.title = element_text(size = 10),
    legend.position = "top"
  ) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "blue", alpha = 0.7) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue", alpha = 0.7)

# Save volcano plot
ggsave(file.path(output_dir, "SKCM_volcano_plot.png"), 
       plot = volcano_plot, 
       width = 10, height = 8, dpi = 300)

ggsave(file.path(output_dir, "SKCM_volcano_plot.pdf"), 
       plot = volcano_plot, 
       width = 10, height = 8)

cat("Volcano plot saved to:", file.path(output_dir, "SKCM_volcano_plot.png"), "\n")

# Create a z-score distribution plot
zscore_plot <- ggplot(output_df, aes(x = Log2FC_ZScore)) +
  geom_histogram(bins = 50, fill = "skyblue", alpha = 0.7, color = "black") +
  labs(
    title = "Distribution of Log2 Fold Change Z-Scores",
    x = "Log2 Fold Change Z-Score",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 12)
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red")

ggsave(file.path(output_dir, "SKCM_zscore_distribution.png"), 
       plot = zscore_plot, 
       width = 10, height = 6, dpi = 300)

# Print summary statistics
cat("\n=== ANALYSIS SUMMARY ===\n")
cat("Top 10 upregulated genes (highest z-scores):\n")
print(head(output_df[, c("Gene_ID", "Log2FoldChange", "Log2FC_ZScore")], 10))

cat("\nTop 10 downregulated genes (lowest z-scores):\n")
print(tail(output_df[, c("Gene_ID", "Log2FoldChange", "Log2FC_ZScore")], 10))

cat("\nFiles created:\n")
cat("1. SKCM_DEG_results_with_zscores.csv - Complete results with z-scores\n")
cat("2. SKCM_volcano_plot.png/.pdf - Volcano plot\n")
cat("3. SKCM_zscore_distribution.png - Z-score distribution plot\n")

cat("\nAnalysis completed successfully!\n")