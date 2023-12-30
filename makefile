# Define variables for directories and files to make the Makefile more maintainable
RAW_DATA_DIR = data/raw
PROCESSED_DATA_DIR = data/processed
FIGURES_DIR = results/figures
MODELS_DIR = results/models
TABLES_DIR = results/tables

# Run all the scripts
all : eda preprocessing_featsengineering model_fitting feature_importance evaluate_models

eda : scripts/eda.py 
	python scripts/eda.py --training-data $(RAW_DATA_DIR)/train.csv --plot-to $(FIGURES_DIR)

preprocessing_featsengineering : scripts/preprocessing_featsengineering.py 
	python scripts/preprocessing_featsengineering.py --train-data $(RAW_DATA_DIR)/train.csv --test-data $(RAW_DATA_DIR)/test.csv --test-labels $(RAW_DATA_DIR)/test_labels.csv --data-to $(PROCESSED_DATA_DIR) --plot-to $(FIGURES_DIR)

model_fitting : scripts/model_fitting.py 
	python scripts/model_fitting.py --original-train $(PROCESSED_DATA_DIR)/train.csv --pipeline-to $(MODELS_DIR) --result-to $(TABLES_DIR)

feature_importance : scripts/feature_importance.py 
	python scripts/feature_importance.py --original-train $(PROCESSED_DATA_DIR)/train.csv --pipeline-to $(MODELS_DIR) --result-to $(FIGURES_DIR)

evaluate_models : scripts/evaluate_models.py 
	python scripts/evaluate_models.py --test-data $(PROCESSED_DATA_DIR)/test.csv --model-dir $(MODELS_DIR) --output-dir $(TABLES_DIR)

# Clean up the generated files
clean:
	rm -rf $(PROCESSED_DATA_DIR)
	rm -rf $(FIGURES_DIR)
	rm -rf $(MODELS_DIR)
	rm -rf $(TABLES_DIR)
