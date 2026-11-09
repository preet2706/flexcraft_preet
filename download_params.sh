mkdir params
cd params
wget https://zenodo.org/records/14711580/files/salad_params.tar.gz
tar -xzf salad_params.tar.gz
mv params salad
mkdir pmpnn
cd pmpnn
for noise in 05 10 20 30; do
    wget https://github.com/sokrypton/ColabDesign/raw/refs/heads/main/colabdesign/mpnn/weights/v_48_0${noise}.pkl
done
cd ..
mkdir solmpnn
cd solmpnn
for noise in 05 10 20 30; do
    wget https://github.com/sokrypton/ColabDesign/raw/refs/heads/main/colabdesign/mpnn/weights/v_48_0${noise}.pkl
done
cd ..
# LigandMPNN. The released weights are PyTorch state dicts, so they are converted into
# flexcraft's format (see scripts/convert_ligandmpnn_params.py). The _25 suffix is the number
# of ligand context atoms per residue; v_32 means 32 neighbours, NOT ProteinMPNN's 48.
mkdir -p ligmpnn/raw
cd ligmpnn/raw
for noise in 005 010 020 030; do
    wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_${noise}_25.pt
done
cd ../../..
for noise in 005 010 020 030; do
    python scripts/convert_ligandmpnn_params.py \
        --in_path params/ligmpnn/raw/ligandmpnn_v_32_${noise}_25.pt \
        --out_path params/ligmpnn/v_32_${noise}_25.pkl
done
cd params
mkdir af
cd af
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar | tar x -C params
cd ../..
mkdir dependencies
cd dependencies
wget https://github.com/martinpacesa/BindCraft/raw/refs/heads/main/functions/DAlphaBall.gcc
chmod +x DAlphaBall.gcc
cd ..