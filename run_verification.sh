#!/bin/bash

python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.01_0.01.dnnp --network N models/NetS_64.onnx --neurify
python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.02_0.01.dnnp --network N models/NetS_64.onnx --neurify
python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.04_0.01.dnnp --network N models/NetS_64.onnx --neurify
python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.08_0.01.dnnp --network N models/NetS_64.onnx --neurify

python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.01_0.01.dnnp --network N models/NetM_64.onnx --neurify
python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.02_0.01.dnnp --network N models/NetM_64.onnx --neurify
python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.04_0.01.dnnp --network N models/NetM_64.onnx --neurify
python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.08_0.01.dnnp --network N models/NetM_64.onnx --neurify

python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.01_0.01.dnnp --network N models/NetL_64.onnx --neurify
python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.02_0.01.dnnp --network N models/NetL_64.onnx --neurify
python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.04_0.01.dnnp --network N models/NetL_64.onnx --neurify
python3 DNNV/tools/resmonitor.py -T 300 ./run_DNNV.sh properties/robustness_0.08_0.01.dnnp --network N models/NetL_64.onnx --neurify
