#!/usr/bin/env bash
set -e
# adjust entrypoint as needed
python src/qg/train.py scenario=bc1_noslip_all__dgyre
python src/qg/train.py scenario=bc1_noslip_all__fgyre
python src/qg/train.py scenario=bc1_noslip_all__jet
python src/qg/train.py scenario=bc2_sponge_all__dgyre
python src/qg/train.py scenario=bc2_sponge_all__fgyre
python src/qg/train.py scenario=bc2_sponge_all__jet
python src/qg/train.py scenario=bc3_noslipLR_spongeTB__dgyre
python src/qg/train.py scenario=bc3_noslipLR_spongeTB__fgyre
python src/qg/train.py scenario=bc3_noslipLR_spongeTB__jet
python src/qg/train.py scenario=bc4_noslipL_spongeTRB__dgyre
python src/qg/train.py scenario=bc4_noslipL_spongeTRB__fgyre
python src/qg/train.py scenario=bc4_noslipL_spongeTRB__jet
python src/qg/train.py scenario=bc5_freeslip_all__dgyre
python src/qg/train.py scenario=bc5_freeslip_all__fgyre
python src/qg/train.py scenario=bc5_freeslip_all__jet
