git clone https://github.com/metavoiceio/metavoice-src.git
cd metavoice-src && git checkout de3fa21 && cd ..
# replace "threshold_s=30" with "threshold_s=1" in the file "metavoice-src/fam/llm/utils.py"
sed -i 's/threshold_s=30/threshold_s=1/g' metavoice-src/fam/llm/utils.py
# metavoice-src/fam/llm/fast_inference_utils.py has the following lines 
# 51 torch._inductor.config.fx_graph_cache = (
# 52    True  # Experimental feature to reduce compilation times, will be on by default in future
# 53 )
# we set the value to False
sed -i '51s/.*/#torch._inductor.config.fx_graph_cache = (/' metavoice-src/fam/llm/fast_inference_utils.py
sed -i '52s/.*/#    True/' metavoice-src/fam/llm/fast_inference_utils.py
sed -i '53s/.*/#)/' metavoice-src/fam/llm/fast_inference_utils.py
# replace compile=True with compile=False in the file "metavoice-src/fam/llm/fast_inference.py"
sed -i 's/compile=True/compile=False/g' metavoice-src/fam/llm/fast_inference.py
# replace "return dtype" with "return 'float16'" in the file "metavoice-src/fam/llm/utils.py"
sed -i 's/return dtype/return "float16"/g' metavoice-src/fam/llm/utils.py
# replace "model = model.to(device=device, dtype=torch.bfloat16)" with "model = model.to(device=device, dtype=precision)" in the file "metavoice-src/fam/llm/fast_inference_utils.py"
sed -i 's/model = model.to(device=device, dtype=torch.bfloat16)/model = model.to(device=device, dtype=precision)/g' metavoice-src/fam/llm/fast_inference_utils.py
cd ..
touch .setup_done