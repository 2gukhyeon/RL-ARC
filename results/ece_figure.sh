# math
# RLAR
# python ece_figure.py \
# --data_paths ./Math-models-fresh/RLAR/amc23/RLAR_outputs.json ./Math-models-fresh/RLAR/aime24/RLAR_outputs.json ./Math-models-fresh/RLAR/big-math-digits/RLAR_outputs.json ./Math-models-fresh/RLAR/gsm8k/RLAR_outputs.json ./Math-models-fresh/RLAR/math-500/RLAR_outputs.json \
# --output_path ./Math-models-fresh/RLAR/ece_graph_math.png \
# --title "Math (RLAR)"

# RLCR
# python ece_figure.py \
# --data_paths ./Math-models-fresh/RLCR/amc23/RLCR_outputs.json ./Math-models-fresh/RLCR/aime24/RLCR_outputs.json ./Math-models-fresh/RLCR/big-math-digits/RLCR_outputs.json ./Math-models-fresh/RLCR/gsm8k/RLCR_outputs.json ./Math-models-fresh/RLCR/math-500/RLCR_outputs.json \
# --output_path ./Math-models-fresh/RLCR/ece_graph_math.png \
# --title "Math (RLCR)"

# RLVR
python ece_figure.py \
--data_paths ./Math-models-fresh/RLVR/amc23/RLVR_outputs.json ./Math-models-fresh/RLVR/aime24/RLVR_outputs.json ./Math-models-fresh/RLVR/big-math-digits/RLVR_outputs.json ./Math-models-fresh/RLVR/gsm8k/RLVR_outputs.json ./Math-models-fresh/RLVR/math-500/RLVR_outputs.json \
--output_path ./Math-models-fresh/RLVR/ece_graph_math.png \
--title "Math (RLVR)"

# OOD
# RLAR
# python ece_figure.py \
# --data_paths ./Math-models-fresh/RLAR/gpqa/RLAR_outputs.json ./Math-models-fresh/RLAR/hotpot-vanilla-eval-em/RLAR_outputs.json ./Math-models-fresh/RLAR/nq-open/RLAR_outputs.json ./Math-models-fresh/RLAR/simpleqa/RLAR_outputs.json ./Math-models-fresh/RLAR/strategyqa/RLAR_outputs.json ./Math-models-fresh/RLAR/trivia/RLAR_outputs.json \
# --output_path ./Math-models-fresh/RLAR/ece_graph_ood.png \
# --title "OOD (RLAR)"

# RLCR
# python ece_figure.py \
# --data_paths ./Math-models-fresh/RLCR/gpqa/RLCR_outputs.json ./Math-models-fresh/RLCR/hotpot-vanilla-eval-em/RLCR_outputs.json ./Math-models-fresh/RLCR/nq-open/RLCR_outputs.json ./Math-models-fresh/RLCR/simpleqa/RLCR_outputs.json ./Math-models-fresh/RLCR/strategyqa/RLCR_outputs.json ./Math-models-fresh/RLCR/trivia/RLCR_outputs.json \
# --output_path ./Math-models-fresh/RLCR/ece_graph_ood.png \
# --title "OOD (RLCR)"

# RLVR
python ece_figure.py \
--data_paths ./Math-models-fresh/RLVR/gpqa/RLVR_outputs.json ./Math-models-fresh/RLVR/hotpot-vanilla-eval-em/RLVR_outputs.json ./Math-models-fresh/RLVR/nq-open/RLVR_outputs.json ./Math-models-fresh/RLVR/simpleqa/RLVR_outputs.json ./Math-models-fresh/RLVR/strategyqa/RLVR_outputs.json ./Math-models-fresh/RLVR/trivia/RLVR_outputs.json \
--output_path ./Math-models-fresh/RLVR/ece_graph_ood.png \
--title "OOD (RLVR)"