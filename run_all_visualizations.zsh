#!/bin/zsh

# time stretching
# uv run python scripts/visualize_embeddings.py --use_class_means --aug time_stretching --center_projection
# uv run python scripts/visualize_embeddings.py --use_class_means --aug time_stretching
uv run python scripts/visualize_embeddings.py --aug time_stretching --center_projection
# uv run python scripts/visualize_embeddings.py --aug time_stretching

# gain
# uv run python scripts/visualize_embeddings.py --use_class_means --aug gain --center_projection
# uv run python scripts/visualize_embeddings.py --use_class_means --aug gain
uv run python scripts/visualize_embeddings.py --aug gain --center_projection
# uv run python scripts/visualize_embeddings.py --aug gain

# pitch shifting
# uv run python scripts/visualize_embeddings.py --use_class_means --aug pitch_shifting --center_projection
# uv run python scripts/visualize_embeddings.py --use_class_means --aug pitch_shifting
uv run python scripts/visualize_embeddings.py --aug pitch_shifting --center_projection
# uv run python scripts/visualize_embeddings.py --aug pitch_shifting

# Music sounds only:
# echo "Generating visualizations for music sounds only..."
# uv run python scripts/visualize_embeddings.py --use_class_means --aug time_stretching --center_projection --class-filter m
# uv run python scripts/visualize_embeddings.py --use_class_means --aug gain --center_projection --class-filter m
# uv run python scripts/visualize_embeddings.py --use_class_means --aug pitch_shifting --center_projection --class-filter m

# # Instrument samples only:
# echo "Generating visualizations for instrument samples only..."
# uv run python scripts/visualize_embeddings.py --use_class_means --aug time_stretching --center_projection --class-filter is
# uv run python scripts/visualize_embeddings.py --use_class_means --aug gain --center_projection --class-filter is
# uv run python scripts/visualize_embeddings.py --use_class_means --aug pitch_shifting --center_projection --class-filter is

# # Speech only:
# echo "Generating visualizations for speech only..."
# uv run python scripts/visualize_embeddings.py --use_class_means --aug time_stretching --center_projection --class-filter sp
# uv run python scripts/visualize_embeddings.py --use_class_means --aug gain --center_projection --class-filter sp
# uv run python scripts/visualize_embeddings.py --use_class_means --aug pitch_shifting --center_projection --class-filter sp

# # Sound effects only:
# echo "Generating visualizations for sound effects only..."
# uv run python scripts/visualize_embeddings.py --use_class_means --aug time_stretching --center_projection --class-filter fx
# uv run python scripts/visualize_embeddings.py --use_class_means --aug gain --center_projection --class-filter fx
# uv run python scripts/visualize_embeddings.py --use_class_means --aug pitch_shifting --center_projection --class-filter fx

# # Soundscapes only:
# echo "Generating visualizations for soundscapes only..."
# uv run python scripts/visualize_embeddings.py --use_class_means --aug time_stretching --center_projection --class-filter ss
# uv run python scripts/visualize_embeddings.py --use_class_means --aug gain --center_projection --class-filter ss
# uv run python scripts/visualize_embeddings.py --use_class_means --aug pitch_shifting --center_projection --class-filter ss  