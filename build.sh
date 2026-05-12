#!/usr/bin/env bash
set -e

pip install -r requirements.txt

echo "Downloading model files from Google Drive..."

mkdir -p analysis_and_inference/models/ensemble/outputs
mkdir -p analysis_and_inference/models/random_forest/outputs
mkdir -p analysis_and_inference/models/lasso_log_reg/outputs
mkdir -p analysis_and_inference/models/ridge_log_reg/outputs
mkdir -p analysis_and_inference/models/svm/outputs
mkdir -p analysis_and_inference/models/split_and_features

gdown 1Dax7Yiy_zVxqZU10AdN7-gHyLcUc1MnV -O analysis_and_inference/models/ensemble/outputs/ensemble_soft_vote_tuned.pkl
gdown 1Pa3ekT6ih5gh0xWpbL8xnmdUYuNSpTM1 -O analysis_and_inference/models/random_forest/outputs/random_forest_tuned.pkl
gdown 1YzsmibAZkHbxI96RHLYh1nNDii2z0tQC -O analysis_and_inference/models/lasso_log_reg/outputs/lasso_log_reg_tuned.pkl
gdown 1AvI63BbyNedbwPP8RHGfw73RrCZEnBQW -O analysis_and_inference/models/ridge_log_reg/outputs/ridge_log_reg_tuned.pkl
gdown 1xudBRR8h6tVGBtGsn_d2mbda1MVlD5-0 -O analysis_and_inference/models/svm/outputs/svm_tuned.pkl
gdown 13ZXb6knmObX0MbC32wSA8oIikMvEXyQ0 -O analysis_and_inference/models/split_and_features/features.pkl
gdown 195K8STKsVcv2ZIpZJPXtAMfRwVo4YnhM -O analysis_and_inference/models/split_and_features/split.pkl

echo "All model files downloaded."
